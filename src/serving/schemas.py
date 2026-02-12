"""
Pydantic v2 request/response models for the FastAPI serving layer.

All models use strict validation. Field descriptions appear in the
auto-generated OpenAPI docs at /docs.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Request models ────────────────────────────────────────────────────────────


class PredictionRequest(BaseModel):
    """Request body for TFT bankline shift forecast."""

    reach_ids: list[Annotated[int, Field(ge=1, le=50)]] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of reach IDs to forecast (1–50)",
        examples=[[1, 5, 10]],
    )
    bank_sides: str = Field(
        default="both",
        description="Which bank(s) to forecast: 'right', 'left', or 'both'",
        examples=["both"],
    )
    n_steps: int = Field(
        default=5,
        ge=1,
        le=5,
        description="Forecast horizon in time steps (1–5, matching prediction_length)",
    )
    return_quantiles: bool = Field(
        default=True,
        description="If True, return q10 and q90 alongside the q50 median forecast",
    )

    @field_validator("bank_sides")
    @classmethod
    def validate_bank_sides(cls, v: str) -> str:
        allowed = {"right", "left", "both"}
        if v.lower() not in allowed:
            raise ValueError(f"bank_sides must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator("reach_ids")
    @classmethod
    def deduplicate_reach_ids(cls, v: list[int]) -> list[int]:
        return sorted(set(v))


class BaselineRequest(BaseModel):
    """Request body for a named baseline model forecast."""

    reach_id: Annotated[int, Field(ge=1, le=50)] = Field(
        ..., description="Reach ID (1–50)"
    )
    bank_side: str = Field(
        ..., description="'right' or 'left'"
    )
    model_name: str = Field(
        ...,
        description="Baseline model: 'persistence', 'linear', 'arima', or 'random_forest'",
        examples=["persistence"],
    )
    n_steps: int = Field(
        default=5, ge=1, le=10, description="Forecast horizon"
    )

    @field_validator("bank_side")
    @classmethod
    def validate_bank_side(cls, v: str) -> str:
        if v.lower() not in {"right", "left"}:
            raise ValueError("bank_side must be 'right' or 'left'")
        return v.lower()

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        allowed = {"persistence", "linear", "arima", "random_forest"}
        if v.lower() not in allowed:
            raise ValueError(f"model_name must be one of {allowed}")
        return v.lower()


class TrainRequest(BaseModel):
    """Request body for triggering model retraining."""

    experiment_name: str | None = Field(
        default=None,
        description="MLflow experiment name (uses config default if not set)",
    )
    run_name: str | None = Field(
        default=None,
        description="MLflow run name for this training job",
    )
    overrides: dict[str, float | int | str] = Field(
        default_factory=dict,
        description="Hyperparameter overrides as dot-notation dict, e.g. {'tft.hidden_size': 128}",
    )


# ── Response sub-models ───────────────────────────────────────────────────────


class QuantileForecast(BaseModel):
    """Single-step quantile forecast."""

    step: int = Field(..., description="Forecast step index (1-based)")
    estimated_year: int = Field(..., description="Estimated calendar year of this step")
    q50: float = Field(
        ...,
        description=(
            "Median forecast of bank_distance (meters). "
            "Sign convention: Left bank — positive = erosion, negative = deposition. "
            "Right bank — negative = erosion, positive = deposition."
        ),
    )
    q10: float | None = Field(None, description="10th percentile of bank_distance forecast (lower bound, same sign convention as q50)")
    q90: float | None = Field(None, description="90th percentile of bank_distance forecast (upper bound, same sign convention as q50)")


class SeriesForecast(BaseModel):
    """Complete forecast for one (reach_id, bank_side) series."""

    reach_id: int
    bank_side: str
    series_id: str
    last_observed_year: int
    last_observed_value: float = Field(
        ...,
        description=(
            "Last known bank_distance in meters. "
            "Left bank: positive = erosion, negative = deposition. "
            "Right bank: negative = erosion, positive = deposition."
        ),
    )
    forecasts: list[QuantileForecast]


class AnomalyInfo(BaseModel):
    """Anomaly detection result for one series."""

    series_id: str
    reach_id: int
    bank_side: str
    is_anomalous: bool = Field(
        ..., description="True if anomaly score exceeds threshold"
    )
    anomaly_score: float | None = Field(
        None, description="Raw VAE reconstruction error score"
    )
    changepoint_years: list[int] = Field(
        default_factory=list,
        description="Years where structural change was detected by PELT",
    )


# ── Response models ───────────────────────────────────────────────────────────


class PredictionResponse(BaseModel):
    """Response for POST /predict."""

    model_version: str
    predictions: list[SeriesForecast]
    anomaly_info: list[AnomalyInfo]
    status: str = "ok"


class BaselineForecastResponse(BaseModel):
    """Response for POST /predict/baseline."""

    model_name: str
    reach_id: int
    bank_side: str
    forecasts: list[QuantileForecast]
    metrics_on_test: dict[str, float] | None = Field(
        None,
        description="NSE, RMSE, MAE, KGE on test split if available",
    )


class EvaluationResponse(BaseModel):
    """Response for GET /evaluate."""

    split: str
    nse: float | None
    rmse: float | None
    mae: float | None
    kge: float | None
    quantile_coverage_80: float | None
    n_series: int
    per_series: list[dict] | None = None


class ChangepointResponse(BaseModel):
    """Response for GET /anomaly/changepoints."""

    total_changepoints: int
    potentially_protected_reaches: int
    changepoints: list[dict]


class SeriesHistoryResponse(BaseModel):
    """Response for GET /series/{reach_id}/{bank_side}."""

    reach_id: int
    bank_side: str
    series_id: str
    observations: list[dict] = Field(
        ..., description="List of {year, bank_distance} historical records"
    )
    latest_forecast: list[QuantileForecast] | None = None


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    model_loaded: bool
    model_version: str | None
    n_series: int
    last_training_date: str | None
    vae_loaded: bool
    changepoints_loaded: bool


class TrainResponse(BaseModel):
    """Response for POST /train."""

    job_id: str
    status: str = "started"
    message: str


class MetricsSummary(BaseModel):
    """Aggregated metric summary across series."""

    mean_nse: float | None
    mean_rmse: float | None
    mean_mae: float | None
    mean_kge: float | None
    n_series_evaluated: int


class YearPointForecast(BaseModel):
    """Predicted bankline position for one (reach_id, bank_side) at a specific year."""

    reach_id: int = Field(..., description="Reach ID (1–50)")
    bank_side: str = Field(..., description="'left' or 'right'")
    series_id: str = Field(..., description="Series label e.g. 'R01_left'")
    q50: float = Field(
        ...,
        description=(
            "Median predicted bank_distance (meters). "
            "Left bank: positive = erosion, negative = deposition. "
            "Right bank: negative = erosion, positive = deposition."
        ),
    )
    q10: float | None = Field(None, description="10th percentile (lower bound)")
    q90: float | None = Field(None, description="90th percentile (upper bound)")


class YearPredictionResponse(BaseModel):
    """Response for GET /predict/year/{year} — all 100 series for a single forecast year."""

    year: int = Field(..., description="Requested forecast year")
    last_observed_year: int = Field(..., description="Last year with actual observed data (2020)")
    n_steps: int = Field(..., description="Number of steps ahead from last observed year")
    n_points: int = Field(..., description="Number of series returned (always 100)")
    forecast_type: str = Field(
        ...,
        description=(
            "'direct' for years 2021–2025 (single TFT pass, highest accuracy). "
            "'rolling' for years 2026+ (iterative multi-step, accuracy degrades with horizon)."
        ),
    )
    accuracy_warning: str | None = Field(
        None,
        description=(
            "Present for rolling forecasts. Describes expected accuracy degradation. "
            "Use q10/q90 intervals to convey uncertainty to the end user."
        ),
    )
    predictions: list[YearPointForecast]
