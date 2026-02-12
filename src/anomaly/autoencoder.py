"""
LSTM Variational Autoencoder for unsupervised bankline anomaly detection.

The VAE is trained on unprotected (normal) bankline series. Protected reaches
or intervention events produce high reconstruction errors because their
patterns deviate from the learned distribution of natural erosion dynamics.

Architecture
------------
Encoder: LSTM → mean (μ) + log-variance (log σ²) of latent distribution
Reparameterization: z = μ + ε·σ  (ε ~ N(0,1))
Decoder: Linear projection of z → LSTM → reconstruction

Loss: MSE reconstruction + β·KL(N(μ,σ²) || N(0,1))
Anomaly score: per-timestep MSE reconstruction error
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from src.config import Settings, get_settings
from src.data.preprocessing import SCALE_FEATURES, apply_scaler, load_scaler


# ── LSTM-VAE model ────────────────────────────────────────────────────────────


class LSTMVariationalAutoencoder(pl.LightningModule):
    """
    LSTM Variational Autoencoder for time series anomaly detection.

    Input/Output shapes
    -------------------
    Input  : (batch, seq_len, n_features)
    Output : (batch, seq_len, n_features) [reconstruction]

    Usage
    -----
    1. Train on normal (unprotected) reaches
    2. Call anomaly_score(x) on all reaches
    3. Flag reaches exceeding threshold from fit_threshold()
    """

    def __init__(
        self,
        n_features: int = 4,
        seq_len: int = 27,
        hidden_size: int = 64,
        latent_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        kl_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.mu_proj = nn.Linear(hidden_size, latent_dim)
        self.logvar_proj = nn.Linear(hidden_size, latent_dim)

        # Decoder
        self.z_to_hidden = nn.Linear(latent_dim, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=n_features + latent_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_size, n_features)

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, F) → μ, log_σ²: (B, latent_dim)"""
        _, (h, _) = self.encoder_lstm(x)
        h_last = h[-1]  # last layer: (B, hidden_size)
        return self.mu_proj(h_last), self.logvar_proj(h_last)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick: z = μ + ε·exp(0.5·log_σ²)"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at eval time

    def decode(
        self, z: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """z: (B, latent_dim), x: (B, T, F) → recon: (B, T, F)"""
        z_expanded = z.unsqueeze(1).expand(-1, self.seq_len, -1)  # (B, T, latent)
        decoder_input = torch.cat([x, z_expanded], dim=-1)  # (B, T, F+latent)
        out, _ = self.decoder_lstm(decoder_input)
        return self.output_proj(out)  # (B, T, F)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, mu, logvar)"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x)
        return recon, mu, logvar

    def _vae_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
        # KL divergence: −½ Σ(1 + log_σ² − μ² − σ²)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.kl_weight * kl_loss

    def training_step(
        self, batch: torch.Tensor | tuple, batch_idx: int
    ) -> torch.Tensor:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        recon, mu, logvar = self(x)
        loss = self._vae_loss(x, recon, mu, logvar)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self, batch: torch.Tensor | tuple, batch_idx: int
    ) -> None:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        recon, mu, logvar = self(x)
        loss = self._vae_loss(x, recon, mu, logvar)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> np.ndarray:
        """
        Compute per-timestep reconstruction MSE as anomaly score.

        Higher score = more anomalous (deviates from learned normal patterns).

        Returns
        -------
        np.ndarray of shape (batch, seq_len)
        """
        self.eval()
        recon, _, _ = self(x)
        score = ((x - recon) ** 2).mean(dim=-1)  # (B, T)
        return score.cpu().numpy()

    @torch.no_grad()
    def series_anomaly_score(self, x: torch.Tensor) -> np.ndarray:
        """Mean anomaly score per series (mean over time steps)."""
        scores = self.anomaly_score(x)
        return scores.mean(axis=1)  # (B,)


# ── Training and inference helpers ────────────────────────────────────────────


def _build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
) -> torch.Tensor:
    """
    Build 3D tensor of shape (n_series, seq_len, n_features) from long DataFrame.

    For series shorter than seq_len, pad with zeros on the left.
    Returns float32 tensor.
    """
    sequences = []
    for _, grp in df.groupby(["reach_id", "bank_side"]):
        grp = grp.sort_values("time_idx")
        vals = grp[feature_cols].values.astype(np.float32)
        # Pad or truncate to seq_len
        if len(vals) < seq_len:
            pad = np.zeros((seq_len - len(vals), len(feature_cols)), dtype=np.float32)
            vals = np.concatenate([pad, vals], axis=0)
        else:
            vals = vals[-seq_len:]  # keep most recent
        sequences.append(vals)

    return torch.tensor(np.stack(sequences), dtype=torch.float32)


def train_anomaly_detector(
    df: pd.DataFrame,
    settings: Settings | None = None,
    save_path: str | None = None,
) -> LSTMVariationalAutoencoder:
    """
    Train LSTM-VAE on all bankline series (or a subset of unprotected reaches).

    Parameters
    ----------
    df        : preprocessed long-format DataFrame (scaled features)
    settings  : Settings instance
    save_path : if provided, save model checkpoint here

    Returns
    -------
    Trained LSTMVariationalAutoencoder
    """
    if settings is None:
        settings = get_settings()

    ae_cfg = settings.anomaly.autoencoder
    seq_len = len(settings.data.observed_years)
    feature_cols = ["bank_distance", "rate_of_change", "rolling_mean_3", "channel_width"]

    # Scale features for VAE (separate from TFT's GroupNormalizer)
    try:
        scaler = load_scaler(settings)
        df_scaled = apply_scaler(df, scaler, feature_cols)
    except FileNotFoundError:
        logger.warning("Scaler not found; using unscaled data for VAE")
        df_scaled = df.copy()

    X = _build_sequences(df_scaled, feature_cols, seq_len)
    logger.info(f"VAE training tensor shape: {X.shape}")  # (100, 27, 4)

    # 80/20 train/val split of series (not temporal — VAE is unsupervised)
    n = X.shape[0]
    n_train = int(n * 0.8)
    indices = torch.randperm(n)
    train_tensor = X[indices[:n_train]]
    val_tensor = X[indices[n_train:]]

    train_loader = DataLoader(
        TensorDataset(train_tensor), batch_size=16, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor), batch_size=16, shuffle=False
    )

    model = LSTMVariationalAutoencoder(
        n_features=len(feature_cols),
        seq_len=seq_len,
        hidden_size=ae_cfg.hidden_size,
        latent_dim=ae_cfg.latent_dim,
        num_layers=ae_cfg.num_layers,
        dropout=ae_cfg.dropout,
        kl_weight=ae_cfg.kl_weight,
    )

    trainer = pl.Trainer(
        max_epochs=ae_cfg.epochs,
        accelerator=settings.training.accelerator,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ],
        enable_progress_bar=True,
        log_every_n_steps=1,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if save_path:
        trainer.save_checkpoint(save_path)
        logger.info(f"VAE checkpoint saved to {save_path}")

    return model


def compute_anomaly_scores(
    df: pd.DataFrame,
    model: LSTMVariationalAutoencoder,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """
    Compute anomaly scores for all 100 series.

    Returns
    -------
    DataFrame with columns: reach_id, bank_side, series_id, anomaly_score,
                             is_anomalous (True if score > threshold)
    """
    if settings is None:
        settings = get_settings()

    ae_cfg = settings.anomaly.autoencoder
    seq_len = len(settings.data.observed_years)
    feature_cols = ["bank_distance", "rate_of_change", "rolling_mean_3", "channel_width"]

    try:
        scaler = load_scaler(settings)
        df_scaled = apply_scaler(df, scaler, feature_cols)
    except FileNotFoundError:
        df_scaled = df.copy()

    X = _build_sequences(df_scaled, feature_cols, seq_len)

    model.eval()
    scores = model.series_anomaly_score(X)  # (100,)

    threshold = float(np.percentile(scores, ae_cfg.threshold_percentile))

    rows = []
    from src.data.loader import get_series_id

    for i, ((reach_id, bank_side), _) in enumerate(
        df.groupby(["reach_id", "bank_side"])
    ):
        rows.append(
            {
                "reach_id": int(reach_id),
                "bank_side": str(bank_side),
                "series_id": get_series_id(int(reach_id), str(bank_side)),
                "anomaly_score": float(scores[i]),
                "is_anomalous": bool(scores[i] > threshold),
            }
        )

    return pd.DataFrame(rows)


def save_anomaly_threshold(threshold: float, path: str | Path) -> None:
    """Save threshold scalar to disk for API use."""
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump({"threshold": threshold}, f)
    logger.info(f"Anomaly threshold {threshold:.4f} saved to {p}")


def load_anomaly_threshold(path: str | Path) -> float:
    """Load threshold scalar from disk."""
    import json

    with Path(path).open() as f:
        return float(json.load(f)["threshold"])
