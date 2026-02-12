"""
Data loader: reads the Distances xlsx file and produces a tidy long-format DataFrame.

The xlsx has a 2-row hierarchical header:
  Row 1: "Reaches", "Distance(1991)", "Distance(1991)", "Distance(1993)", ...
  Row 2: "",         "Right Bank (m)", "Left Bank (m)", "Right Bank (m)", ...

pandas read_excel with header=[0,1] produces a MultiIndex column.
We flatten it to "Distance(YEAR)_Right Bank (m)" / "Distance(YEAR)_Left Bank (m)".
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.config import Settings, get_settings


# ── Constants ─────────────────────────────────────────────────────────────────

_REACHES_COL = "Reaches"
_YEAR_PATTERN = re.compile(r"Distance\((\d{4})\)")
_BANK_PATTERN = re.compile(r"(Right|Left) Bank \(m\)", re.IGNORECASE)


# ── Public API ────────────────────────────────────────────────────────────────


def load_long_dataframe(settings: Settings | None = None) -> pd.DataFrame:
    """
    Full pipeline: xlsx → validated tidy long-format DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        reach_id      : int  (1..50)
        bank_side     : str  ('right' | 'left')
        year          : int  (calendar year)
        bank_distance : float (meters, can be negative)

    This is the single public entry point used by all downstream modules.
    """
    if settings is None:
        settings = get_settings()
    raw_path = Path(settings.paths.raw_xlsx)
    logger.info(f"Loading xlsx from {raw_path.absolute()}")
    wide_df = load_xlsx(raw_path, settings.data.sheet_name)
    long_df = wide_to_long(wide_df)
    logger.info(
        f"Loaded {len(long_df)} rows | "
        f"{long_df['reach_id'].nunique()} reaches | "
        f"{long_df['year'].nunique()} years"
    )
    return long_df


def load_xlsx(path: str | Path, sheet_name: str = "Data") -> pd.DataFrame:
    """
    Read the xlsx file and return a clean wide-format DataFrame.

    Steps
    -----
    1. Read with header=[0,1] to capture the 2-row header
    2. Flatten MultiIndex columns
    3. Drop trailing empty columns (cols 56-59 in the xlsx)
    4. Drop trailing empty rows (rows 53-57)
    5. Rename 'Reaches' column, cast to int
    """
    df = pd.read_excel(
        path,
        sheet_name=sheet_name,
        header=[0, 1],
        engine="openpyxl",
    )

    # Flatten MultiIndex columns
    df.columns = _flatten_multiindex(df.columns)

    # Drop columns where the flattened name is empty or NaN
    df = df.loc[:, [c for c in df.columns if _is_valid_column(c)]]

    # Drop rows where Reaches is null (trailing empty rows)
    df = df[df[_REACHES_COL].notna()].copy()
    df[_REACHES_COL] = df[_REACHES_COL].astype(int)

    logger.debug(f"Wide DataFrame: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape from wide (one row per reach, one col per year×bank) to long format.

    Output columns: [reach_id, bank_side, year, bank_distance]
    """
    rows: list[dict[str, object]] = []

    for _, rec in df.iterrows():
        reach_id = int(rec[_REACHES_COL])
        for col in df.columns:
            if col == _REACHES_COL:
                continue
            parsed = _parse_column(col)
            if parsed is None:
                continue
            year, side = parsed
            val = rec[col]
            rows.append(
                {
                    "reach_id": reach_id,
                    "bank_side": side,
                    "year": year,
                    "bank_distance": float(val) if (val is not None and not _is_na(val)) else np.nan,
                }
            )

    result = pd.DataFrame(rows, columns=["reach_id", "bank_side", "year", "bank_distance"])
    result = result.sort_values(["reach_id", "bank_side", "year"]).reset_index(drop=True)
    return result


def get_series_id(reach_id: int, bank_side: str) -> str:
    """Canonical series identifier: 'R01_right', 'R50_left', etc."""
    return f"R{reach_id:02d}_{bank_side}"


# ── Internal helpers ──────────────────────────────────────────────────────────


def _flatten_multiindex(cols: pd.MultiIndex) -> list[str]:
    """
    Flatten a 2-level MultiIndex from pd.read_excel(header=[0,1]).

    Rules
    -----
    - Level 0 = year label like "Distance(1991)" or "Reaches"
    - Level 1 = bank label like "Right Bank (m)" or NaN/empty

    Result: "Distance(1991)_Right Bank (m)", "Distance(1991)_Left Bank (m)", "Reaches"
    For unnamed upper-level cols pandas fills with "Unnamed: N_level_0" — skip those.
    """
    flattened: list[str] = []
    for top, bottom in cols:
        top_str = str(top).strip()
        bottom_str = str(bottom).strip() if not _is_na(bottom) else ""

        # pandas uses "Unnamed: N_level_0" for merged cells in row 1
        if top_str.startswith("Unnamed:") and not bottom_str:
            flattened.append("")
            continue

        if top_str == _REACHES_COL:
            flattened.append(_REACHES_COL)
            continue

        if bottom_str:
            flattened.append(f"{top_str}_{bottom_str}")
        else:
            flattened.append(top_str)

    return flattened


def _is_valid_column(col: str) -> bool:
    """Keep only 'Reaches' and properly named distance columns."""
    if not col or col.isspace():
        return False
    if col == _REACHES_COL:
        return True
    # Must match Distance(YEAR)_Right Bank (m) or Distance(YEAR)_Left Bank (m)
    return bool(_YEAR_PATTERN.search(col) and _BANK_PATTERN.search(col))


def _parse_column(col: str) -> tuple[int, str] | None:
    """
    Parse a distance column name into (year, bank_side).

    Returns None for unrecognised columns.
    """
    year_match = _YEAR_PATTERN.search(col)
    bank_match = _BANK_PATTERN.search(col)
    if not year_match or not bank_match:
        return None
    year = int(year_match.group(1))
    side = bank_match.group(1).lower()  # 'right' or 'left'
    return year, side


def _is_na(val: object) -> bool:
    """Robust NA check that handles str, float, None."""
    if val is None:
        return True
    try:
        return bool(pd.isna(val))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
