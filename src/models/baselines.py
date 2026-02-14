"""
Baseline models for bankline shift prediction.

All baselines implement the BaseBaseline ABC with:
  fit(series_df)                 — fit on a single (reach, bank_side) training series
  predict(history_df, n_steps)   — produce n_steps-ahead point forecasts

Baselines are evaluated against the TFT to establish performance floors.
A TFT that cannot beat persistence is not learning useful patterns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor


# ── Metric computation ────────────────────────────────────────────────────────


def compute_metrics(
    actuals: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """
    Compute NSE, RMSE, MAE, KGE for a single series forecast.

    NSE (Nash-Sutcliffe Efficiency)
        1 − Σ(obs−sim)² / Σ(obs−mean(obs))²
        Range: (−∞, 1]. 1=perfect, 0=mean-only baseline, <0=worse than mean.

    KGE (Kling-Gupta Efficiency, Gupta et al. 2009)
        1 − √[(r−1)² + (α−1)² + (β−1)²]
        r = Pearson correlation, α = std_sim/std_obs, β = mean_sim/mean_obs
        KGE=1 is perfect; decomposes into correlation, bias, variability errors.

    RMSE and MAE are in the original units (meters).
    """
    obs = np.asarray(actuals, dtype=float)
    sim = np.asarray(predictions, dtype=float)

    mae = float(np.mean(np.abs(obs - sim)))
    rmse = float(np.sqrt(np.mean((obs - sim) ** 2)))

    obs_mean = np.mean(obs)
    ss_res = np.sum((obs - sim) ** 2)
    ss_tot = np.sum((obs - obs_mean) ** 2)
    nse = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else float("nan")

    # KGE components
    if len(obs) > 1 and np.std(obs) > 1e-10:
        r = float(np.corrcoef(obs, sim)[0, 1])
        alpha = float(np.std(sim) / np.std(obs))
        beta = float(np.mean(sim) / np.mean(obs)) if abs(np.mean(obs)) > 1e-10 else 1.0
        kge = float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    else:
        kge = float("nan")

    return {"NSE": nse, "RMSE": rmse, "MAE": mae, "KGE": kge}


# ── Abstract base ─────────────────────────────────────────────────────────────


class BaseBaseline(ABC):
    """Abstract interface for all baseline models."""

    name: str = "baseline"

    @abstractmethod
    def fit(self, series_df: pd.DataFrame) -> None:
        """
        Fit on a single series DataFrame.

        Parameters
        ----------
        series_df : DataFrame for one (reach_id, bank_side) sorted by year,
                    with columns [year, bank_distance, time_idx, ...]
        """

    @abstractmethod
    def predict(self, history_df: pd.DataFrame, n_steps: int) -> np.ndarray:
        """
        Generate n_steps-ahead point forecasts.

        Parameters
        ----------
        history_df : historical data up to and including the last training year
        n_steps    : number of future steps to forecast

        Returns
        -------
        np.ndarray of shape (n_steps,)
        """

    def evaluate(
        self,
        full_df: pd.DataFrame,
        reach_id: int,
        bank_side: str,
        train_end_year: int,
        val_end_year: int,
    ) -> dict[str, float]:
        """
        Fit-predict-score one series. Returns metric dict with reach metadata.
        """
        series = (
            full_df[(full_df["reach_id"] == reach_id) & (full_df["bank_side"] == bank_side)]
            .sort_values("year")
            .copy()
        )
        train = series[series["year"] <= train_end_year]
        val = series[(series["year"] > train_end_year) & (series["year"] <= val_end_year)]

        if len(train) < 2 or len(val) == 0:
            logger.warning(f"Skipping reach {reach_id} {bank_side}: insufficient data")
            return {"NSE": float("nan"), "RMSE": float("nan"), "MAE": float("nan"), "KGE": float("nan")}

        self.fit(train)
        preds = self.predict(train, n_steps=len(val))
        actuals = val["bank_distance"].values

        metrics = compute_metrics(actuals, preds)
        metrics["reach_id"] = reach_id
        metrics["bank_side"] = bank_side
        metrics["model"] = self.name
        return metrics


# ── Baseline implementations ──────────────────────────────────────────────────


class PersistenceBaseline(BaseBaseline):
    """
    Naive baseline: carry forward the last observed value indefinitely.

    This is the hardest baseline to beat for slowly evolving systems.
    If the TFT cannot outperform this, it is not learning useful patterns.
    """

    name = "persistence"
    _last_value: float = 0.0

    def fit(self, series_df: pd.DataFrame) -> None:
        s = series_df.sort_values("year")
        self._last_value = float(s["bank_distance"].iloc[-1])

    def predict(self, history_df: pd.DataFrame, n_steps: int) -> np.ndarray:
        return np.full(n_steps, self._last_value)


class LinearExtrapolationBaseline(BaseBaseline):
    """
    Fit a linear regression on (year, bank_distance) and extrapolate.

    Captures long-term monotonic trends (e.g. Jamuna's secular westward drift).
    Fails when the trend is nonlinear or has structural breaks.
    """

    name = "linear"
    _slope: float = 0.0
    _intercept: float = 0.0
    _last_year: int = 0

    def fit(self, series_df: pd.DataFrame) -> None:
        s = series_df.sort_values("year").dropna(subset=["bank_distance"])
        x = s["year"].values.astype(float)
        y = s["bank_distance"].values
        coeffs = np.polyfit(x, y, deg=1)
        self._slope = float(coeffs[0])
        self._intercept = float(coeffs[1])
        self._last_year = int(s["year"].iloc[-1])

    def predict(self, history_df: pd.DataFrame, n_steps: int) -> np.ndarray:
        last_year = float(history_df["year"].max())
        future_years = np.arange(last_year + 1, last_year + n_steps + 1)
        return self._slope * future_years + self._intercept


class ARIMABaseline(BaseBaseline):
    """
    ARIMA via statsmodels. Auto-selects (p, d=1, q) order by AIC over a
    small grid: p in {0,1,2,3}, q in {0,1,2}.  No external dependencies
    beyond statsmodels (already a direct project dependency).
    """

    name = "arima"
    _model: object = None

    def fit(self, series_df: pd.DataFrame) -> None:
        import warnings
        from statsmodels.tsa.arima.model import ARIMA

        values = (
            series_df.sort_values("year")["bank_distance"].dropna().to_numpy()
        )

        best_aic = float("inf")
        best_order = (1, 1, 1)
        for p in range(0, 4):
            for q in range(0, 3):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = ARIMA(values, order=(p, 1, q)).fit()
                    if m.aic < best_aic:
                        best_aic = m.aic
                        best_order = (p, 1, q)
                except Exception:
                    continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = ARIMA(values, order=best_order).fit()

    def predict(self, history_df: pd.DataFrame, n_steps: int) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict()")
        return np.asarray(self._model.forecast(steps=n_steps))


class RandomForestBaseline(BaseBaseline):
    """
    Random Forest with recursive multi-step forecasting using lag features.

    Features per step: lag_1, lag_2, lag_3, rate_of_change, rolling_mean_3.
    Multi-step: predicted value at t is fed back as lag_1 for t+1.
    """

    name = "random_forest"
    _model: RandomForestRegressor | None = None
    _n_lags: int = 3

    def _make_features(
        self, values: list[float]
    ) -> np.ndarray:
        """Build feature vector from most recent values."""
        n = len(values)
        lag1 = values[-1] if n >= 1 else 0.0
        lag2 = values[-2] if n >= 2 else lag1
        lag3 = values[-3] if n >= 3 else lag1
        roc = (values[-1] - values[-2]) if n >= 2 else 0.0
        roll3 = float(np.mean(values[-3:])) if n >= 3 else float(np.mean(values))
        return np.array([[lag1, lag2, lag3, roc, roll3]])

    def fit(self, series_df: pd.DataFrame) -> None:
        s = series_df.sort_values("year")["bank_distance"].dropna().tolist()
        if len(s) < self._n_lags + 2:
            logger.warning("Not enough data for RF baseline, using persistence fallback")
            self._model = None
            self._fallback = s[-1] if s else 0.0
            return

        X_list, y_list = [], []
        for i in range(self._n_lags, len(s)):
            X_list.append(self._make_features(s[:i]).flatten())
            y_list.append(s[i])

        X = np.array(X_list)
        y = np.array(y_list)

        self._model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X, y)
        self._history = s.copy()

    def predict(self, history_df: pd.DataFrame, n_steps: int) -> np.ndarray:
        if self._model is None:
            return np.full(n_steps, self._fallback)

        history = history_df.sort_values("year")["bank_distance"].dropna().tolist()
        preds = []
        for _ in range(n_steps):
            x = self._make_features(history)
            p = float(self._model.predict(x)[0])
            preds.append(p)
            history.append(p)
        return np.array(preds)


# ── Batch evaluation helper ───────────────────────────────────────────────────


def evaluate_all_baselines(
    full_df: pd.DataFrame,
    train_end_year: int = 2010,
    val_end_year: int = 2015,
) -> pd.DataFrame:
    """
    Run all 4 baselines on all 100 (reach, bank_side) series.

    Returns
    -------
    pd.DataFrame with one row per (model, reach_id, bank_side)
    and columns [model, reach_id, bank_side, NSE, RMSE, MAE, KGE]
    """
    baselines: list[BaseBaseline] = [
        PersistenceBaseline(),
        LinearExtrapolationBaseline(),
        ARIMABaseline(),
        RandomForestBaseline(),
    ]
    results = []
    series_groups = full_df.groupby(["reach_id", "bank_side"])
    total = len(series_groups)

    for i, (key, _) in enumerate(series_groups):
        reach_id, bank_side = key
        logger.debug(f"Evaluating baselines for reach {reach_id} {bank_side} ({i+1}/{total})")
        for model in baselines:
            try:
                m = model.evaluate(full_df, reach_id, bank_side, train_end_year, val_end_year)
                results.append(m)
            except Exception as e:
                logger.warning(f"{model.name} failed for reach {reach_id} {bank_side}: {e}")
                results.append({
                    "model": model.name, "reach_id": reach_id, "bank_side": bank_side,
                    "NSE": float("nan"), "RMSE": float("nan"),
                    "MAE": float("nan"), "KGE": float("nan"),
                })

    return pd.DataFrame(results)
