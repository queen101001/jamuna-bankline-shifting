"""
Full TFT training pipeline with MLflow experiment tracking.

Entry point: train_tft() — called directly or from tune.py for each Optuna trial.
"""

from __future__ import annotations

import src  # Applies torch.load patch (must be before other imports)

import os
from pathlib import Path

import torch

import mlflow
import mlflow.pytorch
import lightning.pytorch as pl
from loguru import logger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import MLFlowLogger

from src.config import Settings, get_settings
from src.data.dataset import make_dataloaders
from src.data.loader import load_long_dataframe
from src.data.preprocessing import run_preprocessing_pipeline, temporal_split
from src.models.tft_wrapper import build_tft_model


def train_tft(
    settings: Settings | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
    overrides: dict[str, object] | None = None,
) -> str:
    """
    Full TFT training pipeline.

    Steps
    -----
    1. Load and preprocess data from xlsx
    2. Temporal split into train/val/test
    3. Build PyTorch Forecasting DataLoaders
    4. Build TFT model (with optional hyperparameter overrides)
    5. Train with Lightning Trainer (EarlyStopping, ModelCheckpoint)
    6. Log all artifacts to MLflow
    7. Return MLflow run_id (used by Optuna to fetch val_loss)

    Parameters
    ----------
    settings        : Settings instance; loads from config.yaml if None
    experiment_name : MLflow experiment name (overrides config)
    run_name        : MLflow run name (e.g. "trial_42" for Optuna)
    overrides       : flat dict of hyperparameter overrides, e.g.
                      {"tft.hidden_size": 128, "tft.dropout": 0.2}
                      Applied by creating a modified copy of settings.

    Returns
    -------
    str : MLflow run_id
    """
    if settings is None:
        settings = get_settings()
    if overrides:
        settings = _apply_overrides(settings, overrides)

    exp_name = experiment_name or settings.mlflow.experiment_name

    # Reproducibility
    pl.seed_everything(settings.training.random_seed, workers=True)

    # ── Data pipeline ─────────────────────────────────────────────────────────
    logger.info("Loading data...")
    raw_df = load_long_dataframe(settings)
    df = run_preprocessing_pipeline(raw_df, settings)
    train_df, val_df, _ = temporal_split(df, settings)

    has_val = len(val_df) > 0
    train_dataset, train_loader, val_loader = make_dataloaders(train_df, val_df, settings)
    if not has_val:
        logger.info("No validation split — training on all data")

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Building TFT model...")
    model = build_tft_model(train_dataset, settings)

    # When no validation set, patch the LR scheduler to monitor train_loss_epoch
    # instead of val_loss (which doesn't exist without a val split).
    if not has_val:
        _original_configure_optimizers = model.configure_optimizers

        def _patched_configure_optimizers():
            result = _original_configure_optimizers()
            if "lr_scheduler" in result and isinstance(result["lr_scheduler"], dict):
                result["lr_scheduler"]["monitor"] = "train_loss_epoch"
            return result

        model.configure_optimizers = _patched_configure_optimizers

    # ── Callbacks ────────────────────────────────────────────────────────────
    ckpt_dir = settings.tft_checkpoints_dir()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = []

    if has_val:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=settings.training.early_stopping_patience,
                mode="min",
                verbose=True,
            )
        )
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="tft-{epoch:02d}-val_loss={val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
            )
        )
    else:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="tft-{epoch:02d}-train_loss={train_loss_epoch:.4f}",
                monitor="train_loss_epoch",
                mode="min",
                save_top_k=3,
                save_last=True,
            )
        )

    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(exp_name)

    mlflow_logger = MLFlowLogger(
        experiment_name=exp_name,
        run_name=run_name,
        tracking_uri=settings.mlflow.tracking_uri,
        log_model=False,  # we handle artifact logging manually
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=settings.tft.max_epochs,
        accelerator=settings.training.accelerator,
        gradient_clip_val=settings.tft.gradient_clip_val,
        callbacks=callbacks,
        logger=mlflow_logger,
        enable_progress_bar=True,
        deterministic="warn",
        log_every_n_steps=1,
    )

    # ── Learning rate finder ──────────────────────────────────────────────
    # Use a separate loader with num_workers=0 to avoid deadlock when running
    # inside a ThreadPoolExecutor (background training from the API).
    try:
        from lightning.pytorch.tuner import Tuner

        lr_loader = train_dataset.to_dataloader(
            train=True, batch_size=settings.tft.batch_size, num_workers=0, drop_last=False,
        )
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(
            model,
            train_dataloaders=lr_loader,
            min_lr=1e-5,
            max_lr=1.0,
            num_training=100,
        )
        optimal_lr = lr_finder.suggestion()
        if optimal_lr is not None:
            logger.info(f"LR finder suggests: {optimal_lr:.6f}")
            model.hparams.learning_rate = optimal_lr
        else:
            logger.warning("LR finder returned None, using config learning_rate")
    except Exception as e:
        logger.warning(f"LR finder failed ({e}), using config learning_rate")

    # ── Training ──────────────────────────────────────────────────────────────
    logger.info("Starting training...")
    with mlflow.start_run(
        run_name=run_name,
        experiment_id=mlflow.get_experiment_by_name(exp_name).experiment_id,
    ) as run:
        _log_settings_to_mlflow(settings)

        fit_kwargs: dict = {"train_dataloaders": train_loader}
        if val_loader is not None:
            fit_kwargs["val_dataloaders"] = val_loader

        trainer.fit(model, **fit_kwargs)

        # Log best checkpoint
        best_ckpt = trainer.checkpoint_callback.best_model_path
        best_score = trainer.checkpoint_callback.best_model_score

        if best_ckpt:
            mlflow.log_artifact(best_ckpt, artifact_path="checkpoints")

        metric_name = "best_val_loss" if has_val else "best_train_loss"
        if best_score is not None:
            mlflow.log_metric(metric_name, float(best_score))

        logger.info(
            f"Training complete | {metric_name}={float(best_score):.4f} | "
            f"checkpoint={Path(best_ckpt).name if best_ckpt else 'none'}"
        )

    return run.info.run_id


def _apply_overrides(settings: Settings, overrides: dict[str, object]) -> Settings:
    """
    Apply flat dot-notation overrides to a copy of Settings.

    Example: {"tft.hidden_size": 128} sets settings.tft.hidden_size = 128.
    Only supports one level of nesting (section.param).
    """
    # Convert to dict, apply overrides, re-validate
    data = settings.model_dump()
    for key, val in overrides.items():
        parts = key.split(".", 1)
        if len(parts) == 2:
            section, param = parts
            if section in data and param in data[section]:
                data[section][param] = val
            else:
                logger.warning(f"Override key not found in settings: {key}")
        else:
            logger.warning(f"Override key has no section prefix: {key}")
    return Settings.model_validate(data)


def _log_settings_to_mlflow(settings: Settings) -> None:
    """Log all scalar settings as MLflow params."""
    def _flatten(d: dict, prefix: str = "") -> dict[str, object]:
        result = {}
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(_flatten(v, full_key))
            elif isinstance(v, (int, float, str, bool)):
                result[full_key] = v
        return result

    flat = _flatten(settings.model_dump())
    for k, v in flat.items():
        try:
            mlflow.log_param(k, v)
        except Exception:
            pass  # param key length limits etc.


def main() -> None:
    """CLI entry point: python -m src.training.train"""
    import argparse

    parser = argparse.ArgumentParser(description="Train TFT on Jamuna bankline data")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to config.yaml"
    )
    parser.add_argument("--run-name", default=None, help="MLflow run name")
    args = parser.parse_args()

    # Change to project root so relative paths work
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

    from src.config import load_settings

    settings = load_settings(args.config)
    run_id = train_tft(settings=settings, run_name=args.run_name)
    logger.info(f"Training complete. MLflow run_id: {run_id}")


if __name__ == "__main__":
    main()
