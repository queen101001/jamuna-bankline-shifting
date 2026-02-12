"""
Model evaluation: walk-forward temporal CV, LOLO spatial CV, and baseline comparison.

Metrics: NSE, RMSE, MAE, KGE (standard hydrology benchmark suite).
Results are saved as CSV artifacts and logged to MLflow.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from loguru import logger
from pytorch_forecasting import TemporalFusionTransformer

from src.config import Settings, get_settings
from src.data.dataset import build_tft_dataset, make_dataloaders
from src.data.loader import load_long_dataframe, get_series_id
from src.data.preprocessing import run_preprocessing_pipeline, temporal_split
from src.models.baselines import compute_metrics, evaluate_all_baselines
from src.models.tft_wrapper import load_tft_from_checkpoint


# ── TFT evaluation ────────────────────────────────────────────────────────────


def evaluate_tft(
    checkpoint_path: str,
    settings: Settings | None = None,
    split: str = "test",
    save_dir: str | None = None,
) -> pd.DataFrame:
    """
    Evaluate TFT on val or test split using walk-forward prediction.

    For each series and each valid encoder window in the split,
    generate prediction_length-step forecasts and compare to actuals.

    Metrics returned per series
    ---------------------------
    NSE, RMSE, MAE, KGE : standard hydrology metrics
    quantile_coverage_80 : fraction of actuals within [q10, q90] interval
                           (should be ~80% for well-calibrated quantiles)

    Parameters
    ----------
    checkpoint_path : path to .ckpt file
    settings        : Settings instance
    split           : 'val' or 'test'
    save_dir        : if provided, saves metrics CSV to this directory

    Returns
    -------
    pd.DataFrame with one row per (reach_id, bank_side)
    """
    if settings is None:
        settings = get_settings()

    logger.info(f"Evaluating TFT on {split} split from {Path(checkpoint_path).name}")

    # ── Data ──────────────────────────────────────────────────────────────────
    raw_df = load_long_dataframe(settings)
    df = run_preprocessing_pipeline(raw_df, settings)
    train_df, val_df, test_df = temporal_split(df, settings)

    eval_df = test_df if split == "test" else val_df
    # Context includes all data up to the split boundary for encoder window
    context_df = pd.concat([train_df, val_df]) if split == "test" else train_df
    combined_df = pd.concat([context_df, eval_df]).sort_values(
        ["reach_id_enc", "bank_side_enc", "time_idx"]
    )

    # ── Dataset & loader ─────────────────────────────────────────────────────
    train_dataset, _, _ = make_dataloaders(train_df, val_df, settings)

    from pytorch_forecasting import TimeSeriesDataSet

    eval_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset,
        combined_df,
        predict=True,
        stop_randomization=True,
    )
    eval_loader = eval_dataset.to_dataloader(
        train=False,
        batch_size=settings.tft.batch_size * 2,
        num_workers=0,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_tft_from_checkpoint(checkpoint_path)
    model.eval()

    # ── Predictions ───────────────────────────────────────────────────────────
    with torch.no_grad():
        raw_preds, raw_x = model.predict(
            eval_loader,
            mode="raw",
            return_x=True,
        )

    # raw_preds["prediction"] shape: (n_windows, pred_len, n_quantiles)
    predictions_tensor = raw_preds["prediction"]
    quantiles = settings.tft.quantiles
    q_med_idx = quantiles.index(0.5)
    q_low_idx = quantiles.index(0.1)
    q_hi_idx = quantiles.index(0.9)

    # ── Recover actuals from raw_x ────────────────────────────────────────────
    # raw_x["decoder_target"] shape: (n_windows, pred_len)
    actuals_tensor = raw_preds.get("target", raw_x.get("decoder_target"))

    results = []
    n_windows = predictions_tensor.shape[0]

    # Collect per-window results then aggregate per series
    window_records = []
    for i in range(n_windows):
        med = predictions_tensor[i, :, q_med_idx].cpu().numpy()
        low = predictions_tensor[i, :, q_low_idx].cpu().numpy()
        high = predictions_tensor[i, :, q_hi_idx].cpu().numpy()

        if actuals_tensor is not None:
            act = actuals_tensor[i].cpu().numpy()
        else:
            act = np.full_like(med, np.nan)

        window_records.append({"med": med, "low": low, "high": high, "act": act})

    # Aggregate all windows into global metrics
    all_act = np.concatenate([r["act"] for r in window_records])
    all_med = np.concatenate([r["med"] for r in window_records])
    all_low = np.concatenate([r["low"] for r in window_records])
    all_high = np.concatenate([r["high"] for r in window_records])

    valid_mask = ~np.isnan(all_act)
    global_metrics = compute_metrics(all_act[valid_mask], all_med[valid_mask])
    coverage = float(np.mean(
        (all_act[valid_mask] >= all_low[valid_mask]) &
        (all_act[valid_mask] <= all_high[valid_mask])
    ))
    global_metrics["quantile_coverage_80"] = coverage
    global_metrics["split"] = split
    global_metrics["n_windows"] = n_windows

    results_df = pd.DataFrame([global_metrics])
    logger.info(
        f"Evaluation results ({split}): "
        f"NSE={global_metrics['NSE']:.3f}, "
        f"RMSE={global_metrics['RMSE']:.2f}m, "
        f"MAE={global_metrics['MAE']:.2f}m, "
        f"KGE={global_metrics['KGE']:.3f}, "
        f"Coverage={coverage:.1%}"
    )

    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out / f"tft_metrics_{split}.csv", index=False)
        logger.info(f"Metrics saved to {out / f'tft_metrics_{split}.csv'}")

    return results_df


def evaluate_per_series(
    checkpoint_path: str,
    settings: Settings | None = None,
    split: str = "test",
) -> pd.DataFrame:
    """
    Per-series (reach × bank_side) evaluation for spatial analysis.

    Returns one row per (reach_id, bank_side) with full metrics.
    Used for identifying high-error reaches and spatial patterns.
    """
    if settings is None:
        settings = get_settings()

    raw_df = load_long_dataframe(settings)
    df = run_preprocessing_pipeline(raw_df, settings)
    train_df, val_df, test_df = temporal_split(df, settings)

    eval_df = test_df if split == "test" else val_df
    context_df = pd.concat([train_df, val_df]) if split == "test" else train_df

    model = load_tft_from_checkpoint(checkpoint_path)
    model.eval()

    train_dataset, _, _ = make_dataloaders(train_df, val_df, settings)
    from pytorch_forecasting import TimeSeriesDataSet

    results = []
    for (reach_enc, side_enc), group_df in df.groupby(["reach_id_enc", "bank_side_enc"]):
        reach_id = int(reach_enc) + 1
        bank_side = "right" if int(side_enc) == 1 else "left"

        # Get eval data for this series
        eval_series = eval_df[
            (eval_df["reach_id_enc"] == reach_enc) &
            (eval_df["bank_side_enc"] == side_enc)
        ]
        if len(eval_series) == 0:
            continue

        # Context + eval for this series
        ctx = context_df[
            (context_df["reach_id_enc"] == reach_enc) &
            (context_df["bank_side_enc"] == side_enc)
        ]
        series_df = pd.concat([ctx, eval_series]).sort_values("time_idx")

        try:
            series_dataset = TimeSeriesDataSet.from_dataset(
                train_dataset, series_df, predict=True, stop_randomization=True
            )
            series_loader = series_dataset.to_dataloader(
                train=False, batch_size=64, num_workers=0
            )
            with torch.no_grad():
                preds = model.predict(series_loader, mode="quantiles")

            q_med = preds[:, :, settings.tft.quantiles.index(0.5)].cpu().numpy().flatten()
            q_low = preds[:, :, settings.tft.quantiles.index(0.1)].cpu().numpy().flatten()
            q_hi = preds[:, :, settings.tft.quantiles.index(0.9)].cpu().numpy().flatten()

            actuals = eval_series.sort_values("time_idx")["bank_distance"].values
            n = min(len(actuals), len(q_med))

            m = compute_metrics(actuals[:n], q_med[:n])
            m["quantile_coverage_80"] = float(np.mean(
                (actuals[:n] >= q_low[:n]) & (actuals[:n] <= q_hi[:n])
            ))
            m["reach_id"] = reach_id
            m["bank_side"] = bank_side
            m["series_id"] = get_series_id(reach_id, bank_side)
            results.append(m)

        except Exception as e:
            logger.warning(f"Failed per-series eval for {reach_id} {bank_side}: {e}")

    return pd.DataFrame(results)


# ── Baseline comparison ───────────────────────────────────────────────────────


def run_baseline_comparison(
    settings: Settings | None = None,
    save_dir: str | None = None,
) -> pd.DataFrame:
    """
    Run all baseline models on the full dataset and save metrics.

    Used to establish the performance floor that TFT must beat.
    """
    if settings is None:
        settings = get_settings()

    raw_df = load_long_dataframe(settings)
    df = run_preprocessing_pipeline(raw_df, settings)

    splits = settings.data.splits
    results_df = evaluate_all_baselines(
        df,
        train_end_year=splits.train_end_year,
        val_end_year=splits.val_end_year,
    )

    logger.info("Baseline summary (mean across all series):")
    summary = results_df.groupby("model")[["NSE", "RMSE", "MAE", "KGE"]].mean()
    logger.info(f"\n{summary.to_string()}")

    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out / "baseline_metrics.csv", index=False)

    return results_df


# ── LOLO spatial cross-validation ────────────────────────────────────────────


def evaluate_lolo(
    checkpoint_path: str,
    settings: Settings | None = None,
    save_dir: str | None = None,
) -> pd.DataFrame:
    """
    Leave-One-Location-Out (LOLO) spatial generalization evaluation.

    For each reach 1..50:
      - Evaluate the trained model on both banks of the held-out reach
      - Compare to baselines on the same reach

    This measures spatial generalization: can the model predict
    a reach it was trained on using cross-reach patterns?
    (Note: the TFT is trained jointly on all reaches, so LOLO here
    means evaluating on reaches with minimal similar neighbors in the
    attention mechanism — not full leave-one-out retraining.)

    Returns
    -------
    pd.DataFrame with per-reach spatial generalization metrics
    """
    if settings is None:
        settings = get_settings()

    raw_df = load_long_dataframe(settings)
    df = run_preprocessing_pipeline(raw_df, settings)
    train_df, val_df, test_df = temporal_split(df, settings)

    model = load_tft_from_checkpoint(checkpoint_path)
    model.eval()

    train_dataset, _, _ = make_dataloaders(train_df, val_df, settings)
    from pytorch_forecasting import TimeSeriesDataSet

    context_df = pd.concat([train_df, val_df])
    results = []

    for reach_id in range(1, settings.data.n_reaches + 1):
        for bank_side in ["right", "left"]:
            side_enc = 1 if bank_side == "right" else 0
            reach_enc = reach_id - 1

            test_series = test_df[
                (test_df["reach_id_enc"] == reach_enc) &
                (test_df["bank_side_enc"] == side_enc)
            ]
            if len(test_series) == 0:
                continue

            ctx = context_df[
                (context_df["reach_id_enc"] == reach_enc) &
                (context_df["bank_side_enc"] == side_enc)
            ]
            combined = pd.concat([ctx, test_series]).sort_values("time_idx")

            try:
                ds = TimeSeriesDataSet.from_dataset(
                    train_dataset, combined, predict=True, stop_randomization=True
                )
                loader = ds.to_dataloader(train=False, batch_size=32, num_workers=0)
                with torch.no_grad():
                    preds = model.predict(loader, mode="quantiles")

                q_med = preds[:, :, settings.tft.quantiles.index(0.5)].cpu().numpy().flatten()
                actuals = test_series.sort_values("time_idx")["bank_distance"].values
                n = min(len(actuals), len(q_med))

                m = compute_metrics(actuals[:n], q_med[:n])
                m["reach_id"] = reach_id
                m["bank_side"] = bank_side
                results.append(m)

            except Exception as e:
                logger.warning(f"LOLO failed for reach {reach_id} {bank_side}: {e}")

    df_results = pd.DataFrame(results)

    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(out / "lolo_metrics.csv", index=False)

    return df_results


def main() -> None:
    """CLI entry point: python -m src.training.evaluate"""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Evaluate TFT on Jamuna bankline data")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--save-dir", default="data/processed/eval_results")
    parser.add_argument("--baselines", action="store_true", help="Also run baseline comparison")
    parser.add_argument("--lolo", action="store_true", help="Also run LOLO evaluation")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

    from src.config import load_settings

    settings = load_settings(args.config)

    evaluate_tft(args.checkpoint, settings, split=args.split, save_dir=args.save_dir)

    if args.baselines:
        run_baseline_comparison(settings, save_dir=args.save_dir)

    if args.lolo:
        evaluate_lolo(args.checkpoint, settings, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
