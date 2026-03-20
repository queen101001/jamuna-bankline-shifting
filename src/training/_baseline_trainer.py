"""
Shared utility for training baseline models across all 100 series.

Each per-algorithm training script delegates to `train_single_baseline()`
to avoid duplicating the load → loop → fit → save logic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.config import Settings
    from src.models.baselines import BaseBaseline


def _configure_logging(model_name: str) -> int:
    """Add a file sink for this model's training log. Returns sink id."""
    log_dir = Path("logs/training")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{model_name}.log"
    sink_id = logger.add(
        str(log_file),
        rotation="10 MB",
        level="DEBUG",
        format="{time:HH:mm:ss} | {level:<8} | {message}",
    )
    return sink_id


def train_single_baseline(
    model_cls: type[BaseBaseline],
    settings: Settings,
    output_dir: str | None = None,
) -> int:
    """
    Load data, iterate all 100 series, fit the model, save .joblib files.

    Parameters
    ----------
    model_cls : The baseline class to instantiate and train (e.g. ARIMABaseline).
    settings  : Project settings (paths, splits, etc.)
    output_dir: Override output directory. Defaults to settings.paths.baselines_dir/{name}.

    Returns
    -------
    int : Number of models successfully saved.
    """
    from src.data.loader import load_long_dataframe
    from src.data.preprocessing import run_preprocessing_pipeline

    model_name = model_cls.name  # type: ignore[attr-defined]
    sink_id = _configure_logging(model_name)

    logger.info(f"Training {model_name} baseline for all series...")

    # Load and preprocess data
    raw_df = load_long_dataframe(settings)
    df = run_preprocessing_pipeline(raw_df, settings)

    splits = settings.data.splits
    save_dir = Path(output_dir) if output_dir else Path(settings.paths.baselines_dir) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    series_groups = df.groupby(["reach_id", "bank_side"])
    total = len(series_groups)
    saved = 0

    for i, ((reach_id, bank_side), _group) in enumerate(series_groups):
        series = df[
            (df["reach_id"] == reach_id) & (df["bank_side"] == bank_side)
        ].sort_values("year")
        train_series = series[series["year"] <= splits.train_end_year]

        if len(train_series) < 2:
            logger.warning(
                f"Skipping reach {reach_id} {bank_side}: only {len(train_series)} train points"
            )
            continue

        try:
            model = model_cls()
            model.fit(train_series)
            out_path = save_dir / f"{reach_id}_{bank_side}.joblib"
            model.save(out_path)
            saved += 1
            if (i + 1) % 20 == 0 or (i + 1) == total:
                logger.info(f"  [{model_name}] {i + 1}/{total} series fitted")
        except Exception as e:
            logger.warning(f"  [{model_name}] Failed for reach {reach_id} {bank_side}: {e}")

    logger.success(f"{model_name}: saved {saved}/{total} models to {save_dir}")
    logger.remove(sink_id)
    return saved
