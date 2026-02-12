"""
Change-point detection using PELT (Pruned Exact Linear Time) via ruptures.

Detects sudden shifts in bankline migration rates that may indicate:
  - Bank protection works (revetments, hard points, guide bunds)
  - Natural avulsion events
  - Measurement regime changes

A protected reach signature: sudden variance reduction (>70%) and
biased prediction residuals (model predicts erosion but reach is stable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt
from loguru import logger

from src.config import Settings, get_settings
from src.data.loader import get_series_id


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class ChangepointResult:
    reach_id: int
    bank_side: str
    series_id: str
    changepoint_years: list[int]
    changepoint_indices: list[int]
    n_changepoints: int
    variance_before: list[float]   # variance of each segment
    variance_after: list[float]    # for protection signature detection
    variance_reduction: list[float]  # (before - after) / before per breakpoint


# ── Public API ────────────────────────────────────────────────────────────────


def detect_changepoints_pelt(
    df: pd.DataFrame,
    settings: Settings | None = None,
) -> list[ChangepointResult]:
    """
    Run PELT change-point detection on each (reach_id, bank_side) series.

    Algorithm: PELT with RBF (Radial Basis Function) cost function.
    - RBF detects both mean shifts and variance changes, unlike L2 (mean only)
    - This is appropriate for bankline migration which can show both:
      sudden avulsions (mean shifts) and protection works (variance reduction)

    Parameters
    ----------
    df       : preprocessed long-format DataFrame
    settings : Settings instance

    Returns
    -------
    List of ChangepointResult, one per (reach_id, bank_side) series
    """
    if settings is None:
        settings = get_settings()

    cp_cfg = settings.anomaly.changepoint
    results = []

    for (reach_id, bank_side), grp in df.groupby(["reach_id", "bank_side"]):
        s = grp.sort_values("year").dropna(subset=["bank_distance"])
        values = s["bank_distance"].values
        years = s["year"].values

        if len(values) < cp_cfg.min_size * 2 + 1:
            logger.debug(
                f"Skipping {reach_id} {bank_side}: "
                f"too few observations ({len(values)})"
            )
            continue

        cp_indices, cp_years, var_before, var_after, var_reduction = _run_pelt(
            values=values,
            years=years,
            model=cp_cfg.model,
            penalty=cp_cfg.penalty,
            min_size=cp_cfg.min_size,
        )

        results.append(
            ChangepointResult(
                reach_id=int(reach_id),
                bank_side=str(bank_side),
                series_id=get_series_id(int(reach_id), str(bank_side)),
                changepoint_years=cp_years,
                changepoint_indices=cp_indices,
                n_changepoints=len(cp_indices),
                variance_before=var_before,
                variance_after=var_after,
                variance_reduction=var_reduction,
            )
        )

    logger.info(
        f"Change-point detection complete | "
        f"{sum(r.n_changepoints for r in results)} total breakpoints across "
        f"{len(results)} series"
    )
    return results


def flag_protected_reaches(
    results: list[ChangepointResult],
    variance_reduction_threshold: float = 0.70,
) -> list[ChangepointResult]:
    """
    Filter to reaches with protection signature:
    variance reduction > threshold at some breakpoint.

    A reach is flagged as potentially protected if its bankline variance
    dropped by more than 70% at some point in time (CEGIS criterion).

    Returns subset of results with at least one such breakpoint.
    """
    flagged = [
        r for r in results
        if any(vr >= variance_reduction_threshold for vr in r.variance_reduction)
    ]
    logger.info(
        f"Flagged {len(flagged)} series as potentially protected "
        f"(variance reduction ≥ {variance_reduction_threshold:.0%})"
    )
    return flagged


def changepoints_to_dataframe(
    results: list[ChangepointResult],
) -> pd.DataFrame:
    """
    Convert list of ChangepointResults to flat DataFrame for API responses
    and CSV export.

    Returns
    -------
    DataFrame with columns:
        reach_id, bank_side, series_id, changepoint_year,
        changepoint_idx, variance_before, variance_after,
        variance_reduction, is_protection_signature
    """
    rows = []
    for r in results:
        for i, (year, idx, vb, va, vr) in enumerate(
            zip(
                r.changepoint_years,
                r.changepoint_indices,
                r.variance_before,
                r.variance_after,
                r.variance_reduction,
            )
        ):
            rows.append(
                {
                    "reach_id": r.reach_id,
                    "bank_side": r.bank_side,
                    "series_id": r.series_id,
                    "changepoint_year": year,
                    "changepoint_idx": idx,
                    "variance_before": vb,
                    "variance_after": va,
                    "variance_reduction": vr,
                    "is_protection_signature": vr >= 0.70,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "reach_id", "bank_side", "series_id", "changepoint_year",
                "changepoint_idx", "variance_before", "variance_after",
                "variance_reduction", "is_protection_signature",
            ]
        )
    return pd.DataFrame(rows)


def save_changepoints(
    results: list[ChangepointResult],
    save_dir: str | Path,
) -> Path:
    """Save changepoint results to CSV."""
    df = changepoints_to_dataframe(results)
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "changepoints.csv"
    df.to_csv(path, index=False)
    logger.info(f"Changepoints saved to {path}")
    return path


# ── Internal helpers ──────────────────────────────────────────────────────────


def _run_pelt(
    values: np.ndarray,
    years: np.ndarray,
    model: str = "rbf",
    penalty: float = 5.0,
    min_size: int = 2,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]:
    """
    Run PELT on a 1D signal and compute variance statistics per segment.

    Returns
    -------
    cp_indices     : 0-based indices of breakpoints (into values array)
    cp_years       : corresponding calendar years
    var_before     : variance of each segment before the breakpoint
    var_after      : variance of each segment after the breakpoint
    var_reduction  : (var_before - var_after) / var_before per breakpoint
    """
    algo = rpt.Pelt(model=model, min_size=min_size, jump=1)
    algo.fit(values)
    breakpoints = algo.predict(pen=penalty)
    # ruptures returns endpoint indices; last is always len(signal), drop it
    cp_indices_raw = [bp - 1 for bp in breakpoints[:-1]]

    if not cp_indices_raw:
        return [], [], [], [], []

    cp_years = [int(years[i]) for i in cp_indices_raw if i < len(years)]

    var_before: list[float] = []
    var_after: list[float] = []
    var_reduction: list[float] = []

    # Create segment boundaries: [0, cp1, cp2, ..., len(values)]
    boundaries = [0] + [i + 1 for i in cp_indices_raw] + [len(values)]

    for j, cp_idx in enumerate(cp_indices_raw):
        seg_start = boundaries[j]
        seg_end = boundaries[j + 1]
        next_end = boundaries[j + 2]

        before = values[seg_start:seg_end]
        after = values[seg_end:next_end]

        vb = float(np.var(before)) if len(before) > 1 else 0.0
        va = float(np.var(after)) if len(after) > 1 else 0.0
        vr = float((vb - va) / vb) if vb > 1e-10 else 0.0

        var_before.append(vb)
        var_after.append(va)
        var_reduction.append(vr)

    return cp_indices_raw, cp_years, var_before, var_after, var_reduction
