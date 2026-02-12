"""
Preprocessing pipeline: imputation → feature engineering → time_idx → encoding → splits.

All transformations are deterministic given the same input DataFrame.
Scalers are fitted on training data only and saved to disk for reuse.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import RobustScaler

from src.config import Settings, get_settings
from src.data.loader import get_series_id


# ── Public pipeline entry point ───────────────────────────────────────────────


def run_preprocessing_pipeline(
    df: pd.DataFrame,
    settings: Settings | None = None,
    fit_scaler: bool = False,
) -> pd.DataFrame:
    """
    End-to-end preprocessing pipeline.

    Steps (in order)
    ----------------
    1. Impute missing bank_distance values (linear interpolation per series)
    2. Compute derived features: rate_of_change, rolling_mean_3, channel_width
    3. Assign gapless time_idx (0..N-1 for N observed years)
    4. Encode static categoricals as integers
    5. Optionally fit + apply RobustScaler on continuous features

    Parameters
    ----------
    df        : raw long-format DataFrame from loader.load_long_dataframe()
    settings  : Settings instance; loads from config.yaml if None
    fit_scaler: if True, fit a new RobustScaler on the full df and save it

    Returns
    -------
    Fully preprocessed DataFrame ready for dataset.py
    """
    if settings is None:
        settings = get_settings()

    logger.info("Starting preprocessing pipeline")
    df = impute_missing(df)
    df = compute_features(df)
    df = assign_time_idx(df, settings)
    df = encode_categoricals(df)
    logger.info(
        f"Preprocessing complete: {len(df)} rows, "
        f"columns={list(df.columns)}"
    )
    return df


# ── Step 1: Imputation ────────────────────────────────────────────────────────


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Linear interpolation of NaN bank_distance values per (reach_id, bank_side) series.

    For isolated missing points (e.g. a single year within a series),
    pandas interpolate(method='linear', limit_direction='both') fills
    by linear interpolation between surrounding values, and forward/backward
    fills edge NaNs.

    The xlsx dataset has both banks for all years, so missing values
    are only occasional point nulls, not structural gaps.
    """
    df = df.copy()
    n_before = df["bank_distance"].isna().sum()

    def _fill_series(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.sort_values("year")
        grp["bank_distance"] = grp["bank_distance"].interpolate(
            method="linear", limit_direction="both"
        )
        return grp

    df = df.groupby(["reach_id", "bank_side"], group_keys=False).apply(_fill_series)
    n_after = df["bank_distance"].isna().sum()
    logger.debug(f"Imputation: {n_before} → {n_after} missing values")
    return df.reset_index(drop=True)


# ── Step 2: Feature engineering ───────────────────────────────────────────────


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to the long DataFrame.

    Sign convention (from dataset definition)
    ------------------------------------------
    Left bank  : positive bank_distance = EROSION,  negative = DEPOSITION
    Right bank : negative bank_distance = EROSION,  positive = DEPOSITION

    New columns
    -----------
    erosion_indicator : unified direction feature — always POSITIVE = erosion,
                        NEGATIVE = deposition, regardless of bank side.
                        Left bank  : erosion_indicator =  bank_distance
                        Right bank : erosion_indicator = -bank_distance
                        This is the primary physically-consistent feature for
                        the model and for frontend display.

    erosion_rate      : year-over-year change in erosion_indicator (positive =
                        erosion accelerating, negative = erosion decelerating
                        or deposition increasing). Consistent across both banks.

    rate_of_change    : raw year-over-year delta in bank_distance (bank-side
                        sign preserved; used internally by the model).

    rolling_mean_3    : 3-point rolling mean of bank_distance per series.

    net_channel_erosion: left_bank_distance − right_bank_distance per reach/year.
                        Positive when both banks are eroding simultaneously
                        (channel widening under erosion pressure).
                        Formula: left(+erosion) + right(+erosion as -right_dist)
                               = left_bank_distance - right_bank_distance

    series_id         : string label "R01_right", etc.
    """
    df = df.copy().sort_values(["reach_id", "bank_side", "year"])

    # erosion_indicator: flip right bank so positive always means erosion
    df["erosion_indicator"] = np.where(
        df["bank_side"] == "left",
        df["bank_distance"],          # left:  +ve = erosion (no flip)
        -df["bank_distance"],         # right: -ve = erosion → flip sign
    )

    # rate_of_change (raw first-difference within each series)
    df["rate_of_change"] = (
        df.groupby(["reach_id", "bank_side"])["bank_distance"]
        .diff()
        .fillna(0.0)
    )

    # erosion_rate: first-difference of erosion_indicator
    # positive = erosion accelerating, consistent across both banks
    df["erosion_rate"] = (
        df.groupby(["reach_id", "bank_side"])["erosion_indicator"]
        .diff()
        .fillna(0.0)
    )

    # rolling_mean_3 (3-point rolling mean of bank_distance per series)
    df["rolling_mean_3"] = (
        df.groupby(["reach_id", "bank_side"])["bank_distance"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )

    # net_channel_erosion: cross-bank erosion pressure per reach/year
    df = _add_net_channel_erosion(df)

    # series_id
    df["series_id"] = df.apply(
        lambda r: get_series_id(int(r["reach_id"]), str(r["bank_side"])), axis=1
    )

    return df.reset_index(drop=True)


def _add_net_channel_erosion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute net_channel_erosion = left_bank_distance − right_bank_distance
    per (reach_id, year) and broadcast to all rows of that reach/year.

    Physical meaning (given sign convention)
    -----------------------------------------
    Left bank  erosion = +left_bank_distance
    Right bank erosion = -right_bank_distance

    net_channel_erosion = left_bank_distance - right_bank_distance
                        = (left erosion signal) + (right erosion signal)

    Interpretation
    --------------
    Large positive : both banks eroding — channel under severe pressure
    Near zero      : banks in balance (both stable, or one eroding / one depositing)
    Large negative : both banks depositing — reach accumulating sediment

    This replaces the old `channel_width = right - left` which had the
    wrong sign interpretation under the actual dataset convention.
    """
    right = (
        df[df["bank_side"] == "right"][["reach_id", "year", "bank_distance"]]
        .rename(columns={"bank_distance": "_rb"})
    )
    left = (
        df[df["bank_side"] == "left"][["reach_id", "year", "bank_distance"]]
        .rename(columns={"bank_distance": "_lb"})
    )
    merged = right.merge(left, on=["reach_id", "year"], how="inner")
    # left(+erosion) - right(+deposition) → positive = both banks eroding
    merged["net_channel_erosion"] = merged["_lb"] - merged["_rb"]
    merged = merged[["reach_id", "year", "net_channel_erosion"]]

    df = df.merge(merged, on=["reach_id", "year"], how="left")
    df["net_channel_erosion"] = df["net_channel_erosion"].fillna(0.0)
    return df


# ── Step 3: time_idx assignment ───────────────────────────────────────────────


def assign_time_idx(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """
    Assign a monotonic gapless integer time_idx to each row.

    PyTorch Forecasting requires time_idx to be:
    - Unique per (group_id, time_idx)
    - Monotonically increasing
    - Gapless within a series (any gap raises a DataSet error)

    We map each observed calendar year to its 0-based rank in the
    sorted observed_years list. This gives indices 0..N-1 even though
    the calendar years are irregular (1991, 1993, 1995, ...).

    The mapping is stored in settings.data.observed_years and is
    deterministic — year 1991 → 0, year 1993 → 1, etc.
    """
    year_map = settings.year_to_time_idx()
    df = df.copy()
    df["time_idx"] = df["year"].map(year_map)

    unmapped = df["time_idx"].isna()
    if unmapped.any():
        bad_years = df.loc[unmapped, "year"].unique().tolist()
        raise ValueError(
            f"Years not in observed_years config: {bad_years}. "
            "Update configs/config.yaml data.observed_years."
        )

    df["time_idx"] = df["time_idx"].astype(int)
    return df


# ── Step 4: Categorical encoding ─────────────────────────────────────────────


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Integer-encode static categorical features.

    Columns added
    -------------
    reach_id_enc  : reach_id − 1  (0-based, 0..49)
    bank_side_enc : 0 = left, 1 = right

    PyTorch Forecasting needs string or integer group_ids.
    We keep both the original columns and the encoded ones.
    """
    df = df.copy()
    df["reach_id_enc"] = df["reach_id"].astype(int) - 1
    df["bank_side_enc"] = (df["bank_side"] == "right").astype(int)
    return df


# ── Step 5: Temporal splits ───────────────────────────────────────────────────


def temporal_split(
    df: pd.DataFrame,
    settings: Settings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward temporal split into train / val / test.

    Default splits (from config)
    ----------------------------
    Train : year <= 2010  (years: 1991-2010, 17 observed years)
    Val   : 2010 < year <= 2015  (years: 2012-2015, 4 observed years)
    Test  : year > 2015  (years: 2016-2020, 5 observed years)

    The test split has exactly prediction_length=5 steps, matching
    the TFT's output horizon for a clean hold-out evaluation.

    No data is shuffled; temporal order is strictly preserved.
    """
    if settings is None:
        settings = get_settings()
    splits = settings.data.splits

    train = df[df["year"] <= splits.train_end_year].copy()
    val = df[(df["year"] > splits.train_end_year) & (df["year"] <= splits.val_end_year)].copy()
    test = df[df["year"] > splits.val_end_year].copy()

    logger.info(
        f"Split sizes — train: {train['year'].nunique()} years, "
        f"val: {val['year'].nunique()} years, "
        f"test: {test['year'].nunique()} years"
    )
    return train, val, test


# ── Scaler utilities ──────────────────────────────────────────────────────────

SCALE_FEATURES = [
    "bank_distance",
    "erosion_indicator",
    "erosion_rate",
    "rate_of_change",
    "rolling_mean_3",
    "net_channel_erosion",
]


def fit_and_save_scaler(
    train_df: pd.DataFrame,
    settings: Settings,
    feature_cols: list[str] | None = None,
) -> RobustScaler:
    """
    Fit a RobustScaler on training data and save to disk.

    RobustScaler uses median and IQR, making it robust to the extreme
    erosion events in the dataset (values up to ±9800 m).

    Scaler is fitted on all training rows jointly (not per-series),
    because PyTorch Forecasting's GroupNormalizer handles per-series
    normalization inside the model. This scaler is used for the
    LSTM-VAE anomaly detector which operates outside PF.
    """
    if feature_cols is None:
        feature_cols = SCALE_FEATURES

    scaler = RobustScaler()
    valid_rows = train_df[feature_cols].dropna()
    scaler.fit(valid_rows.values)

    scaler_path = Path(settings.paths.models_dir) / "scaler.joblib"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    return scaler


def load_scaler(settings: Settings) -> RobustScaler:
    """Load a previously fitted scaler from disk."""
    scaler_path = Path(settings.paths.models_dir) / "scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run training first.")
    return joblib.load(scaler_path)


def apply_scaler(
    df: pd.DataFrame,
    scaler: RobustScaler,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Apply a pre-fitted scaler to specified feature columns."""
    if feature_cols is None:
        feature_cols = SCALE_FEATURES
    df = df.copy()
    valid_mask = df[feature_cols].notna().all(axis=1)
    df.loc[valid_mask, feature_cols] = scaler.transform(
        df.loc[valid_mask, feature_cols].values
    )
    return df
