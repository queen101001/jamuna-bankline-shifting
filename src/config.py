"""
Typed configuration via pydantic-settings.

All modules import `get_settings()` instead of parsing raw YAML dicts.
Settings are cached after first load via @lru_cache.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Sub-models ────────────────────────────────────────────────────────────────


class PathsConfig(BaseModel):
    raw_xlsx: str = "data/raw/Distances (1).xlsx"
    processed_dir: str = "data/processed"
    models_dir: str = "models"


class SplitsConfig(BaseModel):
    train_end_year: int = 2010
    val_end_year: int = 2015
    test_start_year: int = 2016


class DataConfig(BaseModel):
    sheet_name: str = "Data"
    n_reaches: int = 50
    observed_years: list[int] = Field(
        default=[
            1991, 1993, 1995, 1996, 1997, 1998, 1999, 2000,
            2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
            2009, 2010, 2012, 2013, 2014, 2015, 2016, 2017,
            2018, 2019, 2020,
        ]
    )
    splits: SplitsConfig = Field(default_factory=SplitsConfig)

    @field_validator("observed_years")
    @classmethod
    def years_sorted(cls, v: list[int]) -> list[int]:
        return sorted(v)


class TFTConfig(BaseModel):
    encoder_length: int = 10
    prediction_length: int = 5
    quantiles: list[float] = Field(default=[0.1, 0.5, 0.9])
    hidden_size: int = 64
    attention_head_size: int = 4
    dropout: float = 0.1
    hidden_continuous_size: int = 32
    lstm_layers: int = 2
    batch_size: int = 64
    max_epochs: int = 100
    learning_rate: float = 0.001
    gradient_clip_val: float = 0.1


class TrainingConfig(BaseModel):
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 5
    accelerator: str = "auto"
    num_workers: int = 0
    random_seed: int = 42


class MLflowConfig(BaseModel):
    tracking_uri: str = "mlruns"
    experiment_name: str = "jamuna_tft"


class OptunaConfig(BaseModel):
    n_trials: int = 50
    timeout_seconds: int = 7200
    study_name: str = "jamuna_tft_study"
    storage: str = "sqlite:///optuna.db"


class ChangepointConfig(BaseModel):
    model: str = "rbf"
    penalty: float = 5.0
    min_size: int = 2


class AutoencoderConfig(BaseModel):
    latent_dim: int = 16
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    kl_weight: float = 0.1
    epochs: int = 100
    threshold_percentile: float = 95.0


class AnomalyConfig(BaseModel):
    changepoint: ChangepointConfig = Field(default_factory=ChangepointConfig)
    autoencoder: AutoencoderConfig = Field(default_factory=AutoencoderConfig)


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


# ── Root settings ─────────────────────────────────────────────────────────────


class Settings(BaseModel):
    """
    Root configuration model.

    Loaded from configs/config.yaml via `load_settings(path)`.
    All fields are typed and validated by Pydantic v2.
    """

    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    tft: TFTConfig = Field(default_factory=TFTConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    def raw_xlsx_path(self) -> Path:
        return Path(self.paths.raw_xlsx)

    def models_dir(self) -> Path:
        return Path(self.paths.models_dir)

    def tft_checkpoints_dir(self) -> Path:
        return Path(self.paths.models_dir) / "tft"

    def year_to_time_idx(self) -> dict[int, int]:
        """Map calendar year → 0-based gapless time_idx."""
        return {year: idx for idx, year in enumerate(self.data.observed_years)}

    def time_idx_to_year(self) -> dict[int, int]:
        """Map 0-based time_idx → calendar year."""
        return {idx: year for idx, year in enumerate(self.data.observed_years)}


def load_settings(config_path: str | Path = "configs/config.yaml") -> Settings:
    """Load Settings from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.absolute()}")
    with path.open() as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return Settings.model_validate(data)


@lru_cache(maxsize=1)
def get_settings(config_path: str = "configs/config.yaml") -> Settings:
    """
    Cached settings loader.

    Usage in all modules:
        from src.config import get_settings
        settings = get_settings()
    """
    return load_settings(config_path)
