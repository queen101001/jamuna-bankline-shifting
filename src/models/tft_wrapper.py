"""
Temporal Fusion Transformer model construction and checkpoint management.

The TFT is built from a fitted TimeSeriesDataSet using from_dataset(),
which automatically infers input sizes, embedding dimensions, and
normalizer configurations from the dataset.
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from src.config import Settings, get_settings


def build_tft_model(
    train_dataset: object,
    settings: Settings | None = None,
    *,
    hidden_size: int | None = None,
    attention_head_size: int | None = None,
    dropout: float | None = None,
    hidden_continuous_size: int | None = None,
    lstm_layers: int | None = None,
    learning_rate: float | None = None,
) -> TemporalFusionTransformer:
    """
    Instantiate a TFT from a fitted TimeSeriesDataSet.

    Keyword overrides allow Optuna to pass trial-specific hyperparameters
    without modifying the global config.

    Architecture notes
    ------------------
    - QuantileLoss([0.1, 0.5, 0.9]): the 0.5 quantile is the point forecast;
      0.1/0.9 give 80% prediction intervals for uncertainty quantification.
    - output_size=[3]: must be a list (PF 1.4+ requirement for multi-quantile).
    - log_interval=10: enables attention weight logging every 10 batches,
      which is visualized in MLflow for interpretability analysis.
    - reduce_on_plateau_patience: inner LR scheduler inside TFT;
      complementary to the Lightning EarlyStopping callback.
    - Small hidden_size (32-128) prevents overfitting on the ~1500 data points.

    Parameters
    ----------
    train_dataset        : fitted TimeSeriesDataSet (from dataset.make_dataloaders)
    settings             : Settings instance
    hidden_size          : override config value (for Optuna tuning)
    attention_head_size  : override config value
    dropout              : override config value
    hidden_continuous_size: override config value
    lstm_layers          : override config value
    learning_rate        : override config value

    Returns
    -------
    TemporalFusionTransformer ready for pl.Trainer.fit()
    """
    if settings is None:
        settings = get_settings()

    tft_cfg = settings.tft
    tr_cfg = settings.training

    model = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=learning_rate or tft_cfg.learning_rate,
        hidden_size=hidden_size or tft_cfg.hidden_size,
        attention_head_size=attention_head_size or tft_cfg.attention_head_size,
        dropout=dropout or tft_cfg.dropout,
        hidden_continuous_size=hidden_continuous_size or tft_cfg.hidden_continuous_size,
        lstm_layers=lstm_layers or tft_cfg.lstm_layers,
        output_size=[len(tft_cfg.quantiles)],   # [3] for [q10, q50, q90]
        loss=QuantileLoss(quantiles=tft_cfg.quantiles),
        log_interval=10,
        reduce_on_plateau_patience=tr_cfg.reduce_lr_patience,
        log_val_interval=1,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"TFT model built | trainable params: {n_params:,}")
    return model


def load_best_checkpoint(
    settings: Settings | None = None,
) -> str:
    """
    Find the best (lowest val_loss) checkpoint in models/tft/.

    ModelCheckpoint saves files as:
      tft-epoch=05-val_loss=0.1234.ckpt

    We sort by the val_loss value embedded in the filename.

    Returns
    -------
    str path to the best checkpoint file

    Raises
    ------
    FileNotFoundError if no checkpoints exist
    """
    if settings is None:
        settings = get_settings()

    ckpt_dir = settings.tft_checkpoints_dir()
    checkpoints = list(ckpt_dir.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {ckpt_dir.absolute()}. "
            "Run training first: python -m src.training.train"
        )

    def _extract_val_loss(path: Path) -> float:
        try:
            part = path.stem.split("val_loss=")[-1]
            return float(part)
        except (IndexError, ValueError):
            return float("inf")

    best = min(checkpoints, key=_extract_val_loss)
    logger.info(f"Best checkpoint: {best.name}")
    return str(best)


def load_tft_from_checkpoint(
    checkpoint_path: str,
    map_location: str = "cpu",
) -> TemporalFusionTransformer:
    """
    Load a TFT model from a Lightning checkpoint.

    The checkpoint contains the full model architecture via hyperparameters,
    so no dataset reference is needed (unlike the from_dataset constructor).

    Parameters
    ----------
    checkpoint_path : path to .ckpt file
    map_location    : torch device ('cpu', 'cuda', 'cuda:0', etc.)
    """
    model = TemporalFusionTransformer.load_from_checkpoint(
        checkpoint_path,
        map_location=map_location,
    )
    model.eval()
    logger.info(f"TFT loaded from {Path(checkpoint_path).name}")
    return model


def get_feature_importance(
    model: TemporalFusionTransformer,
    val_dataloader: object,
) -> dict[str, float]:
    """
    Extract TFT variable selection weights as feature importance scores.

    The TFT's Variable Selection Networks produce a weight per input feature
    per time step, averaged to a single importance score. These are the
    interpretability outputs unique to TFT.

    Returns
    -------
    dict mapping feature name → mean selection weight (0..1)
    """
    interpretation = model.interpret_output(
        model.predict(val_dataloader, mode="raw", return_x=True)[0],
        reduction="sum",
    )
    encoder_importance: dict[str, float] = {}

    if "encoder_variables" in interpretation:
        weights = interpretation["encoder_variables"]
        for name, w in zip(
            model.hparams.x_reals + model.hparams.x_categoricals,
            weights.tolist(),
        ):
            encoder_importance[name] = float(w)

    return encoder_importance
