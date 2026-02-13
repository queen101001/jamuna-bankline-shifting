"""
FastAPI application for Jamuna River bankline shift prediction.

Endpoints
---------
GET  /health                          — model status and metadata
POST /predict                         — TFT forecast with quantile intervals
POST /predict/baseline                — named baseline model forecast
GET  /evaluate                        — latest evaluation metrics
GET  /anomaly/changepoints            — PELT change-point detection results
GET  /series/{reach_id}/{bank_side}   — historical series + latest forecast
POST /train                           — trigger background retraining

All prediction endpoints use synchronous def (not async def).
PyTorch inference is CPU-bound; async would block the event loop.
Background retraining uses ThreadPoolExecutor to avoid blocking.
"""

from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import math

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Path as PathParam, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pytorch_forecasting import TemporalFusionTransformer

from src.anomaly.changepoint import (
    ChangepointResult,
    changepoints_to_dataframe,
    detect_changepoints_pelt,
)
from src.config import Settings, get_settings
from src.data.dataset import build_prediction_dataset, make_dataloaders
from src.data.loader import get_series_id, load_long_dataframe
from src.data.preprocessing import run_preprocessing_pipeline, temporal_split
from src.models.tft_wrapper import load_best_checkpoint, load_tft_from_checkpoint
from src.serving.schemas import (
    AnomalyInfo,
    BaselineForecastResponse,
    BaselineRequest,
    ChangepointResponse,
    EvaluationResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    QuantileForecast,
    SeriesForecast,
    SeriesHistoryResponse,
    TrainRequest,
    TrainResponse,
    YearPointForecast,
    YearPredictionResponse,
)

# Maximum forecast year supported by the rolling prediction engine
_MAX_FORECAST_YEAR = 2099


# ── Application state ─────────────────────────────────────────────────────────


class AppState:
    """Holds all model and data artifacts loaded at startup."""

    settings: Settings | None = None
    model: TemporalFusionTransformer | None = None
    train_dataset: object = None
    df_full: pd.DataFrame | None = None
    changepoint_df: pd.DataFrame | None = None
    model_version: str | None = None
    n_series: int = 0
    last_training_date: str | None = None
    vae_loaded: bool = False
    predictions_cache: dict[int, dict] = {}  # year → serialised YearPredictionResponse
    cache_ready: bool = False
    _executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
    _train_jobs: dict[str, dict] = {}  # job_id → {status, phase, logs}


state = AppState()


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all artifacts at startup. Release resources at shutdown.

    Startup sequence
    ----------------
    1. Load and validate settings from configs/config.yaml
    2. Load and preprocess the full dataset (cached in memory)
    3. Load best TFT checkpoint from models/tft/
    4. Build train dataset to share normalizer with inference
    5. Run change-point detection (fast, ~1 second)
    6. Mark app as ready
    """
    logger.info("Starting Jamuna Bankline Prediction API...")

    try:
        settings = get_settings()
        state.settings = settings

        # Load and preprocess data
        logger.info("Loading dataset...")
        raw_df = load_long_dataframe(settings)
        df = run_preprocessing_pipeline(raw_df, settings)
        state.df_full = df
        state.n_series = df[["reach_id", "bank_side"]].drop_duplicates().shape[0]

        # Build train dataset (needed for prediction dataset normalizer)
        train_df, val_df, _ = temporal_split(df, settings)
        train_dataset, _, _ = make_dataloaders(train_df, val_df, settings)
        state.train_dataset = train_dataset

        # Load TFT model
        try:
            ckpt_path = load_best_checkpoint(settings)
            state.model = load_tft_from_checkpoint(ckpt_path)
            state.model.eval()
            state.model_version = Path(ckpt_path).stem
            state.last_training_date = str(Path(ckpt_path).stat().st_mtime)
            logger.info(f"TFT loaded: {state.model_version}")
        except FileNotFoundError:
            logger.warning(
                "No trained model found. API will start but /predict will return 503. "
                "Run: python -m src.training.train"
            )

        # Load or build prediction cache
        if state.model is not None:
            cache_path = Path(settings.paths.models_dir) / "predictions_cache.json"
            if cache_path.exists():
                raw = json.loads(cache_path.read_text())
                state.predictions_cache = {int(k): v for k, v in raw.items()}
                state.cache_ready = True
                logger.info(
                    f"Prediction cache loaded from disk ({len(state.predictions_cache)} years)"
                )
            else:
                logger.info(
                    f"Building prediction cache for years "
                    f"{settings.api.cache_start_year}–{settings.api.cache_end_year}…"
                )
                state.predictions_cache = _build_predictions_cache(settings)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(state.predictions_cache))
                state.cache_ready = True
                logger.info(
                    f"Prediction cache built and saved ({len(state.predictions_cache)} years)"
                )
        else:
            logger.info("Skipping cache build — no model loaded yet")

        # Run change-point detection
        logger.info("Running change-point detection...")
        cp_results = detect_changepoints_pelt(df, settings)
        state.changepoint_df = changepoints_to_dataframe(cp_results)
        logger.info(
            f"Change-point detection complete: "
            f"{len(state.changepoint_df)} breakpoints detected"
        )

        logger.info(
            f"API ready | {state.n_series} series | "
            f"model={'loaded' if state.model else 'not loaded'}"
        )

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield  # Application runs

    # Shutdown
    logger.info("Shutting down API...")
    state._executor.shutdown(wait=False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── FastAPI app ───────────────────────────────────────────────────────────────


app = FastAPI(
    title="Jamuna River Bankline Shift Prediction API",
    description=(
        "Temporal Fusion Transformer model for predicting bankline migration "
        "across 50 reaches of the Jamuna River (1991–2020 training data)."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health endpoint ───────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Check API readiness and model loading status."""
    return HealthResponse(
        status="ok" if state.model is not None else "model_not_loaded",
        model_loaded=state.model is not None,
        model_version=state.model_version,
        n_series=state.n_series,
        last_training_date=state.last_training_date,
        vae_loaded=state.vae_loaded,
        changepoints_loaded=state.changepoint_df is not None,
        cache_ready=state.cache_ready,
        cached_years=sorted(state.predictions_cache.keys()) if state.cache_ready else [],
    )


# ── Prediction endpoint ───────────────────────────────────────────────────────


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Generate TFT bankline shift forecasts with quantile uncertainty intervals.

    Returns q50 (median) forecast and optionally q10/q90 bounds.
    Anomaly flags indicate reaches with unusual behaviour patterns.
    """
    _require_model()
    assert state.settings is not None
    assert state.df_full is not None
    assert state.train_dataset is not None

    sides = (
        ["right", "left"] if request.bank_sides == "both" else [request.bank_sides]
    )

    # Filter to requested series
    mask = (
        state.df_full["reach_id"].isin(request.reach_ids)
        & state.df_full["bank_side"].isin(sides)
    )
    df_subset = state.df_full[mask].copy()
    if df_subset.empty:
        raise HTTPException(status_code=400, detail="No matching series found")

    # Build prediction dataset (last encoder_length steps per series)
    pred_dataset = build_prediction_dataset(df_subset, state.train_dataset, state.settings)
    pred_loader = pred_dataset.to_dataloader(
        train=False,
        batch_size=len(request.reach_ids) * len(sides),
        num_workers=0,
    )

    # Run inference
    assert state.model is not None
    with torch.no_grad():
        raw_preds = state.model.predict(
            pred_loader,
            mode="raw",
            return_x=False,
        )

    predictions_tensor = raw_preds["prediction"]  # (N, pred_len, n_quantiles)

    # Build structured response
    series_forecasts = _build_series_forecasts(
        predictions_tensor=predictions_tensor,
        df_subset=df_subset,
        request=request,
        settings=state.settings,
    )

    anomaly_info = _build_anomaly_info(
        reach_ids=request.reach_ids,
        sides=sides,
        changepoint_df=state.changepoint_df,
    )

    return PredictionResponse(
        model_version=state.model_version or "unknown",
        predictions=series_forecasts,
        anomaly_info=anomaly_info,
    )


# ── Year-based prediction (all 100 series for a single year) ──────────────────


@app.get("/predict/year/{year}", response_model=YearPredictionResponse, tags=["Prediction"])
def predict_year(
    year: Annotated[
        int,
        PathParam(
            ge=2021,
            le=_MAX_FORECAST_YEAR,
            description=(
                f"Forecast year (2021–{_MAX_FORECAST_YEAR}). "
                "Years 2021–2025 use a single direct TFT pass (highest accuracy). "
                "Years 2026+ use iterative rolling forecasts with compounding uncertainty."
            ),
        ),
    ],
) -> YearPredictionResponse:
    """
    Predict bankline positions for ALL 100 series (50 reaches × left + right)
    at a single user-specified forecast year (2021–2099).

    Serves from pre-computed cache for years 2021–2040 (instantaneous).
    Falls back to live inference for years outside the cache range.
    """
    # Serve from cache if available
    if state.cache_ready and year in state.predictions_cache:
        cached = state.predictions_cache[year]
        return YearPredictionResponse(
            year=cached["year"],
            last_observed_year=cached["last_observed_year"],
            n_steps=cached["n_steps"],
            n_points=cached["n_points"],
            forecast_type=cached["forecast_type"],
            accuracy_warning=cached.get("accuracy_warning"),
            source="cache",
            predictions=[YearPointForecast(**p) for p in cached["predictions"]],
        )
    # Fall back to live inference
    result = _predict_year_internal(year)
    result.source = "live"
    return result


def _predict_year_internal(year: int) -> YearPredictionResponse:
    """Compute year prediction via live TFT inference (no cache)."""
    _require_model()
    assert state.df_full is not None
    assert state.settings is not None

    last_observed_year = int(state.df_full["year"].max())

    if year <= last_observed_year:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Year {year} is within the observed period "
                f"(last observed year: {last_observed_year}). "
                f"Forecast years must be > {last_observed_year}."
            ),
        )

    n_steps = year - last_observed_year
    pred_len = state.settings.tft.prediction_length
    is_rolling = n_steps > pred_len

    if is_rolling:
        all_step_results = _rolling_forecast(
            model=state.model,
            df_full=state.df_full,
            train_dataset=state.train_dataset,
            settings=state.settings,
            n_steps_total=n_steps,
        )
        target_results = [r for r in all_step_results if r["step"] == n_steps]
        points: list[YearPointForecast] = [
            YearPointForecast(
                reach_id=r["reach_id"],
                bank_side=r["bank_side"],
                series_id=r["series_id"],
                q50=r["q50"],
                q10=r["q10"],
                q90=r["q90"],
            )
            for r in target_results
        ]
        n_rounds = math.ceil(n_steps / pred_len)
        accuracy_warning = (
            f"Rolling forecast: {n_steps} steps ahead using {n_rounds} iterative "
            f"prediction rounds. Each round compounds prediction error. "
            f"Uncertainty intervals (q10–q90) reflect growing forecast uncertainty. "
            f"Treat as trend indication only beyond ~{last_observed_year + 10}."
        )
        forecast_type = "rolling"
    else:
        full_request = PredictionRequest(
            reach_ids=list(range(1, 51)),
            bank_sides="both",
            n_steps=n_steps,
            return_quantiles=True,
        )
        pred_response = predict(full_request)
        points = [
            YearPointForecast(
                reach_id=sf.reach_id,
                bank_side=sf.bank_side,
                series_id=sf.series_id,
                q50=sf.forecasts[-1].q50,
                q10=sf.forecasts[-1].q10,
                q90=sf.forecasts[-1].q90,
            )
            for sf in pred_response.predictions
        ]
        accuracy_warning = None
        forecast_type = "direct"

    return YearPredictionResponse(
        year=year,
        last_observed_year=last_observed_year,
        n_steps=n_steps,
        n_points=len(points),
        forecast_type=forecast_type,
        accuracy_warning=accuracy_warning,
        source="live",
        predictions=points,
    )


# ── Baseline prediction ───────────────────────────────────────────────────────


@app.post("/predict/baseline", response_model=BaselineForecastResponse, tags=["Prediction"])
def predict_baseline(request: BaselineRequest) -> BaselineForecastResponse:
    """
    Run a named baseline model for a single (reach_id, bank_side) series.

    Available models: persistence, linear, arima, random_forest.
    Uses training split for fitting and returns n_steps forecasts.
    """
    _require_data()
    assert state.df_full is not None
    assert state.settings is not None

    from src.models.baselines import (
        ARIMABaseline,
        LinearExtrapolationBaseline,
        PersistenceBaseline,
        RandomForestBaseline,
    )

    model_map = {
        "persistence": PersistenceBaseline(),
        "linear": LinearExtrapolationBaseline(),
        "arima": ARIMABaseline(),
        "random_forest": RandomForestBaseline(),
    }
    baseline = model_map[request.model_name]

    series = state.df_full[
        (state.df_full["reach_id"] == request.reach_id)
        & (state.df_full["bank_side"] == request.bank_side)
    ].sort_values("year")

    if series.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Series not found: reach_id={request.reach_id}, bank_side={request.bank_side}",
        )

    splits = state.settings.data.splits
    train_series = series[series["year"] <= splits.train_end_year]

    try:
        baseline.fit(train_series)
        preds = baseline.predict(train_series, n_steps=request.n_steps)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Baseline prediction failed: {e}")

    last_year = int(series["year"].max())
    forecasts = [
        QuantileForecast(
            step=i + 1,
            estimated_year=last_year + i + 1,
            q50=float(preds[i]),
        )
        for i in range(len(preds))
    ]

    return BaselineForecastResponse(
        model_name=request.model_name,
        reach_id=request.reach_id,
        bank_side=request.bank_side,
        forecasts=forecasts,
    )


# ── Evaluation metrics ────────────────────────────────────────────────────────


@app.get("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
def get_evaluation(
    split: str = Query(default="test", description="'val' or 'test'"),
) -> EvaluationResponse:
    """
    Return latest evaluation metrics for the loaded model.

    Reads pre-computed metrics from data/processed/eval_results/ if available.
    Run python -m src.training.evaluate --checkpoint <path> to generate them.
    """
    _require_data()
    assert state.settings is not None

    metrics_path = (
        Path(state.settings.paths.processed_dir)
        / "eval_results"
        / f"tft_metrics_{split}.csv"
    )
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Evaluation metrics not found at {metrics_path}. "
                "Run: python -m src.training.evaluate --checkpoint <path>"
            ),
        )

    df = pd.read_csv(metrics_path)
    row = df.iloc[0].to_dict() if len(df) > 0 else {}

    return EvaluationResponse(
        split=split,
        nse=row.get("NSE"),
        rmse=row.get("RMSE"),
        mae=row.get("MAE"),
        kge=row.get("KGE"),
        quantile_coverage_80=row.get("quantile_coverage_80"),
        n_series=state.n_series,
    )


# ── Change-point / anomaly endpoints ─────────────────────────────────────────


@app.get("/anomaly/changepoints", response_model=ChangepointResponse, tags=["Anomaly"])
def get_changepoints(
    protected_only: bool = Query(
        default=False,
        description="If True, return only reaches with protection signature (variance reduction ≥ 70%)",
    ),
) -> ChangepointResponse:
    """
    Return PELT change-point detection results.

    Protection signature: sudden variance reduction ≥ 70% (CEGIS criterion).
    Likely indicates installation of bank protection works.
    """
    if state.changepoint_df is None or state.changepoint_df.empty:
        raise HTTPException(status_code=503, detail="Change-point detection not loaded")

    df = state.changepoint_df
    if protected_only:
        df = df[df["is_protection_signature"] == True]

    protected_count = (
        state.changepoint_df["is_protection_signature"].sum()
        if state.changepoint_df is not None
        else 0
    )

    return ChangepointResponse(
        total_changepoints=len(df),
        potentially_protected_reaches=int(protected_count),
        changepoints=df.to_dict(orient="records"),
    )


# ── Series history endpoint ───────────────────────────────────────────────────


@app.get(
    "/series/{reach_id}/{bank_side}",
    response_model=SeriesHistoryResponse,
    tags=["Data"],
)
def get_series(
    reach_id: Annotated[int, PathParam(ge=1, le=50)],
    bank_side: Annotated[str, PathParam(pattern="^(right|left)$")],
    include_forecast: bool = Query(default=True),
) -> SeriesHistoryResponse:
    """
    Return full historical bankline distance series for one (reach, bank_side).

    Optionally includes the latest 5-step TFT forecast.
    """
    _require_data()
    assert state.df_full is not None

    series = state.df_full[
        (state.df_full["reach_id"] == reach_id)
        & (state.df_full["bank_side"] == bank_side)
    ].sort_values("year")

    if series.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Reach {reach_id} {bank_side} not found",
        )

    observations = series[["year", "bank_distance"]].to_dict(orient="records")
    latest_forecast = None

    if include_forecast and state.model is not None:
        try:
            pred_request = PredictionRequest(
                reach_ids=[reach_id],
                bank_sides=bank_side,
                n_steps=5,
                return_quantiles=True,
            )
            pred_response = predict(pred_request)
            if pred_response.predictions:
                latest_forecast = pred_response.predictions[0].forecasts
        except Exception as e:
            logger.warning(f"Forecast failed for /series/{reach_id}/{bank_side}: {e}")

    return SeriesHistoryResponse(
        reach_id=reach_id,
        bank_side=bank_side,
        series_id=get_series_id(reach_id, bank_side),
        observations=observations,
        latest_forecast=latest_forecast,
    )


# ── Training trigger endpoint ─────────────────────────────────────────────────


@app.post("/train", response_model=TrainResponse, tags=["Training"])
def trigger_training(request: TrainRequest) -> TrainResponse:
    """
    Trigger model retraining in a background thread.

    Returns a job_id immediately. Poll /train/{job_id}/status for progress.
    The new model will be loaded automatically on the next startup.
    """
    assert state.settings is not None

    job_id = str(uuid.uuid4())
    state._train_jobs[job_id] = {"status": "started", "phase": "training", "logs": []}

    def _train_job() -> None:
        job: dict = state._train_jobs[job_id]
        job["status"] = "running"
        job["phase"] = "training"

        def _sink(message: object) -> None:
            record = message.record  # type: ignore[union-attr]
            line = f"{record['time'].strftime('%H:%M:%S')} | {record['level'].name:<8} | {record['message']}"
            job["logs"].append(line)

        sink_id = logger.add(_sink, format="{message}", level="DEBUG")
        try:
            job["logs"].append(
                f"[{__import__('datetime').datetime.now().strftime('%H:%M:%S')}] "
                f"Training job {job_id[:8]} started"
            )
            from src.training.train import train_tft

            run_id = train_tft(
                settings=state.settings,
                experiment_name=request.experiment_name,
                run_name=request.run_name or f"api_retrain_{job_id[:8]}",
                overrides=dict(request.overrides),
            )
            job["logs"].append(f"Training complete. MLflow run_id={run_id}")

            # Auto-reload the new best checkpoint
            job["phase"] = "model_reloading"
            logger.info("Training done — reloading model checkpoint…")
            try:
                new_ckpt = load_best_checkpoint(state.settings)
                state.model = load_tft_from_checkpoint(new_ckpt)
                state.model.eval()
                state.model_version = Path(new_ckpt).stem
                state.last_training_date = str(Path(new_ckpt).stat().st_mtime)
                logger.info(f"Model reloaded: {state.model_version}")
            except Exception as reload_err:
                logger.error(f"Model reload failed: {reload_err}")
                raise

            # Rebuild prediction cache
            job["phase"] = "cache_building"
            logger.info(
                f"Building prediction cache for years "
                f"{state.settings.api.cache_start_year}–{state.settings.api.cache_end_year}…"
            )
            state.cache_ready = False
            state.predictions_cache = _build_predictions_cache(state.settings)
            cache_path = Path(state.settings.paths.models_dir) / "predictions_cache.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(state.predictions_cache))
            state.cache_ready = True
            logger.success(
                f"Prediction cache ready — {len(state.predictions_cache)} years cached. "
                "All year queries now served from cache."
            )

            job["phase"] = "ready"
            job["status"] = f"completed:run_id={run_id}"
            logger.info(f"Background training job {job_id} completed: run_id={run_id}")
        except Exception as e:
            job["phase"] = "failed"
            job["status"] = f"failed:{e}"
            job["logs"].append(f"[ERROR] Job failed: {e}")
            logger.error(f"Background training job {job_id} failed: {e}")
        finally:
            logger.remove(sink_id)

    state._executor.submit(_train_job)

    return TrainResponse(
        job_id=job_id,
        status="started",
        message=(
            f"Training started in background. "
            f"Poll GET /train/{job_id}/status for updates."
        ),
    )


@app.get("/train/{job_id}/status", tags=["Training"])
def get_train_status(job_id: str) -> dict[str, str]:
    """Poll the status of a background training job."""
    if job_id not in state._train_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {"job_id": job_id, "status": state._train_jobs[job_id]["status"]}


@app.get("/train/{job_id}/logs", tags=["Training"])
def get_train_logs(
    job_id: str,
    since: int = Query(default=0, ge=0, description="Return log lines from this index onwards"),
) -> dict:
    """
    Stream training log lines for a background job.

    Poll this endpoint every 2 seconds. Pass `since=N` (where N is the
    total lines received so far) to get only new lines since last poll.
    """
    if job_id not in state._train_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job = state._train_jobs[job_id]
    all_logs: list[str] = job.get("logs", [])
    return {
        "job_id": job_id,
        "status": job["status"],
        "phase": job.get("phase", "training"),
        "logs": all_logs[since:],
        "total": len(all_logs),
    }


# ── Internal helpers ──────────────────────────────────────────────────────────


def _build_predictions_cache(settings: Settings) -> dict[int, dict]:
    """
    Pre-compute predictions for every year in [cache_start_year, cache_end_year].

    Returns a plain dict suitable for JSON serialisation:
      {year: {year, last_observed_year, n_steps, n_points, forecast_type,
              accuracy_warning, source, predictions: [...]}}
    """
    cache: dict[int, dict] = {}
    start = settings.api.cache_start_year
    end = settings.api.cache_end_year
    for year in range(start, end + 1):
        try:
            result = _predict_year_internal(year)
            cache[year] = result.model_dump()
        except Exception as e:
            logger.warning(f"Cache build skipped year {year}: {e}")
    return cache


def _rolling_forecast(
    model: "TemporalFusionTransformer",
    df_full: pd.DataFrame,
    train_dataset: object,
    settings: "Settings",
    n_steps_total: int,
) -> list[dict]:
    """
    Iterative multi-step forecast for horizons beyond prediction_length.

    Algorithm
    ---------
    Round 1 : feed last enc_len real observations → predict steps 1..pred_len
    Round 2 : append round-1 q50 as synthetic rows → predict steps pred_len+1..2*pred_len
    Round N : repeat until n_steps_total steps are covered

    For each synthetic row appended to the context window we recompute
    all time-varying features (erosion_indicator, rate_of_change, etc.)
    from the previous row, keeping the feature pipeline consistent with
    what the model saw during training.

    Returns
    -------
    list of dicts: {reach_id, bank_side, series_id, step, year, q50, q10, q90}
    covering steps 1..n_steps_total for all 100 series.
    """
    from src.data.dataset import build_prediction_dataset
    from src.data.loader import get_series_id as _get_series_id

    pred_len = settings.tft.prediction_length
    quantiles = settings.tft.quantiles
    q_med = quantiles.index(0.5)
    q_low = quantiles.index(0.1) if 0.1 in quantiles else None
    q_hi = quantiles.index(0.9) if 0.9 in quantiles else None

    last_observed_year = int(df_full["year"].max())
    last_time_idx = int(df_full["time_idx"].max())

    # Working context — grows with synthetic rows each round
    df_context = df_full.copy()
    all_results: list[dict] = []
    step_offset = 0

    while step_offset < n_steps_total:
        steps_this_round = min(pred_len, n_steps_total - step_offset)

        # Build prediction dataset from current context (last enc_len per series)
        pred_dataset = build_prediction_dataset(df_context, train_dataset, settings)
        pred_loader = pred_dataset.to_dataloader(
            train=False, batch_size=200, num_workers=0
        )

        with torch.no_grad():
            raw_preds = model.predict(pred_loader, mode="raw", return_x=False)

        preds_tensor = raw_preds["prediction"]  # (N, pred_len, n_quantiles)

        # Series order matches groupby sort on (reach_id_enc, bank_side_enc)
        series_keys = (
            df_context.groupby(["reach_id", "bank_side"])
            .first()
            .reset_index()[["reach_id", "bank_side"]]
            .sort_values(["reach_id", "bank_side"])
            .values.tolist()
        )

        # Collect q50/q10/q90 for all steps this round
        round_preds: dict[tuple, list[float]] = {}  # (reach_id, bank_side) → [q50 per step]
        for i, (reach_id, bank_side) in enumerate(series_keys):
            if i >= preds_tensor.shape[0]:
                break
            preds_i = preds_tensor[i].cpu().numpy()  # (pred_len, n_quantiles)
            key = (int(reach_id), str(bank_side))
            round_preds[key] = []

            for t in range(steps_this_round):
                step = step_offset + t + 1
                year = last_observed_year + step
                q50 = float(preds_i[t, q_med])
                q10 = float(preds_i[t, q_low]) if q_low is not None else None
                q90 = float(preds_i[t, q_hi]) if q_hi is not None else None
                all_results.append({
                    "reach_id": int(reach_id),
                    "bank_side": str(bank_side),
                    "series_id": _get_series_id(int(reach_id), str(bank_side)),
                    "step": step,
                    "year": year,
                    "q50": q50,
                    "q10": q10,
                    "q90": q90,
                })
                round_preds[key].append(q50)

        # Build synthetic rows for the next encoder window
        # First, compute net_channel_erosion per (reach_id, year) using q50 values
        new_rows: list[dict] = []
        for t in range(steps_this_round):
            step = step_offset + t + 1
            year = last_observed_year + step
            time_idx = last_time_idx + step

            # net_channel_erosion = left_bank_distance − right_bank_distance
            net_erosion: dict[int, float] = {}
            for reach_id in range(1, 51):
                left_bd = round_preds.get((reach_id, "left"), [None] * steps_this_round)[t]
                right_bd = round_preds.get((reach_id, "right"), [None] * steps_this_round)[t]
                if left_bd is not None and right_bd is not None:
                    net_erosion[reach_id] = left_bd - right_bd

            for reach_id, bank_side in series_keys:
                reach_id = int(reach_id)
                bank_side = str(bank_side)
                key = (reach_id, bank_side)
                if key not in round_preds or t >= len(round_preds[key]):
                    continue

                q50 = round_preds[key][t]

                # Previous row for computing deltas
                prev = df_context[
                    (df_context["reach_id"] == reach_id)
                    & (df_context["bank_side"] == bank_side)
                ].sort_values("time_idx").iloc[-1]

                prev_bd = float(prev["bank_distance"])
                prev_ei = float(prev["erosion_indicator"])
                # Rolling mean uses last 2 known + current predicted
                prev2 = df_context[
                    (df_context["reach_id"] == reach_id)
                    & (df_context["bank_side"] == bank_side)
                ].sort_values("time_idx")["bank_distance"].iloc[-2:].tolist()

                erosion_indicator = q50 if bank_side == "left" else -q50
                rate_of_change = q50 - prev_bd
                erosion_rate = erosion_indicator - prev_ei
                rolling_mean_3 = float(np.mean((prev2 + [q50])[-3:]))

                new_rows.append({
                    "reach_id": reach_id,
                    "bank_side": bank_side,
                    "year": year,
                    "bank_distance": q50,
                    "time_idx": time_idx,
                    "erosion_indicator": erosion_indicator,
                    "rate_of_change": rate_of_change,
                    "erosion_rate": erosion_rate,
                    "rolling_mean_3": rolling_mean_3,
                    "net_channel_erosion": net_erosion.get(reach_id, 0.0),
                    "reach_id_enc": reach_id - 1,
                    "bank_side_enc": 1 if bank_side == "right" else 0,
                    "series_id": _get_series_id(reach_id, bank_side),
                })

        if new_rows:
            df_context = pd.concat(
                [df_context, pd.DataFrame(new_rows)], ignore_index=True
            )

        step_offset += steps_this_round

    return all_results


def _require_model() -> None:
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Run: python -m src.training.train "
                "then restart the API."
            ),
        )


def _require_data() -> None:
    if state.df_full is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")


def _build_series_forecasts(
    predictions_tensor: torch.Tensor,
    df_subset: pd.DataFrame,
    request: PredictionRequest,
    settings: Settings,
) -> list[SeriesForecast]:
    """Convert raw TFT prediction tensor to SeriesForecast objects."""
    quantiles = settings.tft.quantiles
    q_med = quantiles.index(0.5)
    q_low = quantiles.index(0.1) if 0.1 in quantiles else None
    q_hi = quantiles.index(0.9) if 0.9 in quantiles else None

    results = []
    n_preds = predictions_tensor.shape[0]

    # Group predictions by series (predictions are ordered by group_ids)
    series_keys = (
        df_subset.groupby(["reach_id", "bank_side"])
        .first()
        .reset_index()[["reach_id", "bank_side"]]
        .values.tolist()
    )

    for i in range(min(n_preds, len(series_keys))):
        reach_id, bank_side = int(series_keys[i][0]), str(series_keys[i][1])

        series = df_subset[
            (df_subset["reach_id"] == reach_id)
            & (df_subset["bank_side"] == bank_side)
        ].sort_values("year")

        last_year = int(series["year"].max())
        last_value = float(series["bank_distance"].iloc[-1])

        preds_i = predictions_tensor[i, : request.n_steps, :].cpu().numpy()
        forecasts = []
        for t in range(request.n_steps):
            forecasts.append(
                QuantileForecast(
                    step=t + 1,
                    estimated_year=last_year + t + 1,
                    q50=float(preds_i[t, q_med]),
                    q10=float(preds_i[t, q_low]) if (request.return_quantiles and q_low is not None) else None,
                    q90=float(preds_i[t, q_hi]) if (request.return_quantiles and q_hi is not None) else None,
                )
            )

        results.append(
            SeriesForecast(
                reach_id=reach_id,
                bank_side=bank_side,
                series_id=get_series_id(reach_id, bank_side),
                last_observed_year=last_year,
                last_observed_value=last_value,
                forecasts=forecasts,
            )
        )

    return results


def _build_anomaly_info(
    reach_ids: list[int],
    sides: list[str],
    changepoint_df: pd.DataFrame | None,
) -> list[AnomalyInfo]:
    """Build anomaly info from changepoint detection results."""
    results = []
    for reach_id in reach_ids:
        for bank_side in sides:
            sid = get_series_id(reach_id, bank_side)
            cp_years: list[int] = []
            is_anomalous = False

            if changepoint_df is not None and not changepoint_df.empty:
                cp_rows = changepoint_df[
                    (changepoint_df["reach_id"] == reach_id)
                    & (changepoint_df["bank_side"] == bank_side)
                ]
                cp_years = cp_rows["changepoint_year"].tolist()
                is_anomalous = bool(cp_rows["is_protection_signature"].any())

            results.append(
                AnomalyInfo(
                    series_id=sid,
                    reach_id=reach_id,
                    bank_side=bank_side,
                    is_anomalous=is_anomalous,
                    anomaly_score=None,
                    changepoint_years=cp_years,
                )
            )
    return results


def main() -> None:
    """CLI entry point: python -m src.serving.api"""
    import argparse
    import os

    import uvicorn

    parser = argparse.ArgumentParser(description="Start the Jamuna Bankline Prediction API")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

    from src.config import load_settings

    settings = load_settings(args.config)
    host = args.host or settings.api.host
    port = args.port or settings.api.port
    reload = args.reload or settings.api.reload

    uvicorn.run(
        "src.serving.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
