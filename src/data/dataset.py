"""
PyTorch Forecasting TimeSeriesDataSet construction for TFT training.

Key design decisions
--------------------
1. group_ids = ['reach_id_enc', 'bank_side_enc']
   Creates 50×2 = 100 independent series. PF uses these to split
   the DataFrame and apply per-group normalization.

2. GroupNormalizer with transformation='softplus'
   Normalizes each (reach, bank_side) series independently.
   'softplus' handles both positive and negative bank_distance values
   gracefully (unlike 'log' which requires positive values).

3. time_idx must be gapless integers (0..N-1).
   This is ensured by preprocessing.assign_time_idx().
   PF raises ValueError if any group has time_idx gaps.

4. Val/test datasets use TimeSeriesDataSet.from_dataset(train_ds, ...)
   This shares the normalizer fitted on training data, preventing leakage.

5. allow_missing_timesteps=True handles series that may not have all
   time steps in val/test splits (e.g., a reach with fewer observations).
"""

from __future__ import annotations

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from torch.utils.data import DataLoader

from src.config import Settings, get_settings

# Features that are observed but not known in the future
_TIME_VARYING_UNKNOWN_REALS = [
    "bank_distance",
    "erosion_indicator",   # unified: positive = erosion regardless of bank side
    "erosion_rate",        # YoY change in erosion_indicator
    "rate_of_change",      # raw YoY change in bank_distance (bank-sign preserved)
    "rolling_mean_3",
    "net_channel_erosion", # left - right: positive = both banks eroding
]

# Static categorical features (per-series, time-invariant)
_STATIC_CATEGORICALS = ["reach_id_enc", "bank_side_enc"]


def build_tft_dataset(
    df: pd.DataFrame,
    settings: Settings | None = None,
    predict_mode: bool = False,
    min_encoder_length: int | None = None,
) -> TimeSeriesDataSet:
    """
    Construct a TimeSeriesDataSet for TFT training or prediction.

    Parameters
    ----------
    df            : processed DataFrame with all required columns
    settings      : Settings instance
    predict_mode  : if True, last encoder_length steps per group only
                    (used by the FastAPI serving layer)
    min_encoder_length : minimum context window; defaults to encoder_length // 2

    Returns
    -------
    TimeSeriesDataSet ready for .to_dataloader()
    """
    if settings is None:
        settings = get_settings()

    tft = settings.tft
    min_enc = min_encoder_length if min_encoder_length is not None else tft.encoder_length // 2

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="bank_distance",
        group_ids=_STATIC_CATEGORICALS,
        # Encoder: context window
        max_encoder_length=tft.encoder_length,
        min_encoder_length=min_enc,
        # Decoder: forecast horizon
        max_prediction_length=tft.prediction_length,
        min_prediction_length=1,
        # Static features (time-invariant per series)
        static_categoricals=_STATIC_CATEGORICALS,
        static_reals=[],
        # Known future inputs (only the time index itself is known)
        time_varying_known_categoricals=[],
        time_varying_known_reals=["time_idx"],
        # Unknown future inputs (all observed features)
        time_varying_unknown_reals=_TIME_VARYING_UNKNOWN_REALS,
        # Per-series normalization (critical for heterogeneous reach scales)
        target_normalizer=GroupNormalizer(
            groups=_STATIC_CATEGORICALS,
            transformation="softplus",
        ),
        # Helper features for TFT's temporal self-attention
        add_relative_time_idx=True,    # adds 0..1 normalized position
        add_target_scales=True,         # adds mean/scale as known inputs
        add_encoder_length=True,        # handles variable context windows
        # Allow series with fewer than max_encoder_length observations
        # (happens in val/test splits near the boundary)
        allow_missing_timesteps=True,
        predict_mode=predict_mode,
    )
    return dataset


def make_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    settings: Settings | None = None,
) -> tuple[TimeSeriesDataSet, DataLoader, DataLoader]:
    """
    Build train TimeSeriesDataSet and derive val dataset from it.

    The val dataset MUST be created via TimeSeriesDataSet.from_dataset()
    to share the GroupNormalizer fitted on training data.
    Creating a fresh val dataset would re-fit the normalizer on val data,
    causing data leakage.

    Returns
    -------
    (train_dataset, train_loader, val_loader)
    """
    if settings is None:
        settings = get_settings()

    tft = settings.tft
    tr = settings.training

    train_dataset = build_tft_dataset(train_df, settings)

    val_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset,
        val_df,
        predict=True,
        stop_randomization=True,
    )

    train_loader = train_dataset.to_dataloader(
        train=True,
        batch_size=tft.batch_size,
        num_workers=tr.num_workers,
        drop_last=False,
    )
    val_loader = val_dataset.to_dataloader(
        train=False,
        batch_size=tft.batch_size * 2,
        num_workers=tr.num_workers,
        drop_last=False,
    )

    return train_dataset, train_loader, val_loader


def build_prediction_dataset(
    df: pd.DataFrame,
    train_dataset: TimeSeriesDataSet,
    settings: Settings | None = None,
) -> TimeSeriesDataSet:
    """
    Build a dataset for inference using the last encoder_length observations
    per series. Shares the normalizer from train_dataset.

    Parameters
    ----------
    df            : full preprocessed DataFrame (all years)
    train_dataset : fitted training dataset (provides normalizer)
    settings      : Settings instance

    Returns
    -------
    TimeSeriesDataSet in predict_mode=True
    """
    if settings is None:
        settings = get_settings()

    enc_len = settings.tft.encoder_length

    # Keep only the last enc_len time steps per group
    max_idx = df.groupby(["reach_id_enc", "bank_side_enc"])["time_idx"].transform("max")
    df_tail = df[df["time_idx"] > (max_idx - enc_len)].copy()

    return TimeSeriesDataSet.from_dataset(
        train_dataset,
        df_tail,
        predict=True,
        stop_randomization=True,
    )
