"""
Train ALL baseline models for all 100 series.

Entry point: python -m src.training.train_all_baselines

Iterates over all 10 baseline algorithms, training each one on every
(reach_id, bank_side) combination and saving the fitted models to
models/baselines/{model_name}/{reach_id}_{bank_side}.joblib.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.config import get_settings
from src.models.baselines import ALL_BASELINE_CLASSES
from src.training._baseline_trainer import train_single_baseline


def main() -> None:
    log_dir = Path("logs/training")
    log_dir.mkdir(parents=True, exist_ok=True)
    sink_id = logger.add(
        str(log_dir / "all_baselines.log"),
        rotation="10 MB",
        level="INFO",
        format="{time:HH:mm:ss} | {level:<8} | {message}",
    )

    settings = get_settings()
    total_saved = 0

    logger.info(f"Training {len(ALL_BASELINE_CLASSES)} baseline algorithms...")

    for cls in ALL_BASELINE_CLASSES:
        try:
            saved = train_single_baseline(cls, settings)
            total_saved += saved
        except Exception as e:
            logger.error(f"Failed to train {cls.name}: {e}")  # type: ignore[attr-defined]

    logger.success(f"All baselines complete: {total_saved} models saved total")
    logger.remove(sink_id)


if __name__ == "__main__":
    main()
