"""
Factor Calculation & Normalization
------------------------------------
Takes a raw factor DataFrame (from DataFetcher.build_factor_dataframe)
and produces clean, normalized factor scores ready for composite scoring.

Key concepts (also in CFA L1 Quantitative Methods):
  - Cross-sectional z-score: (x - mean) / std within the universe
  - Percentile rank: where does this stock sit vs peers
  - Winsorizing: clip outliers at 5th/95th percentile before scoring
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Factor definitions
# Direction: +1 = higher raw value is BETTER, -1 = lower raw value is BETTER
# ---------------------------------------------------------------------------

FACTOR_DIRECTION: dict[str, int] = {
    # Value — lower is better (cheaper)
    "pe_ratio":       -1,
    "pb_ratio":       -1,
    "ps_ratio":       -1,
    "ev_ebitda":      -1,

    # Quality — higher is better
    "roe":            +1,
    "roa":            +1,
    "gross_margin":   +1,
    "net_margin":     +1,
    "current_ratio":  +1,

    # Risk — lower debt is better
    "debt_to_equity": -1,

    # Income — higher yield is better
    "dividend_yield": +1,

    # Growth — higher is better
    "revenue_growth":  +1,
    "earnings_growth": +1,

    # Momentum — higher is better
    "mom_1m":  +1,
    "mom_3m":  +1,
    "mom_6m":  +1,
    "mom_12m": +1,
}

# Factor groups — used by style presets in scorer.py
FACTOR_GROUPS: dict[str, list[str]] = {
    "value":    ["pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda"],
    "quality":  ["roe", "roa", "gross_margin", "net_margin", "debt_to_equity"],
    "momentum": ["mom_1m", "mom_3m", "mom_6m", "mom_12m"],
    "income":   ["dividend_yield"],
    "growth":   ["revenue_growth", "earnings_growth"],
}


# ---------------------------------------------------------------------------
# Core normalization functions
# ---------------------------------------------------------------------------

def winsorize(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
    """Clip values at [lower, upper] percentile to remove outlier distortion."""
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def zscore(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score: (x - mean) / std. Returns NaN if std == 0."""
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(np.nan, index=series.index)
    return (series - series.mean()) / std


def percentile_rank(series: pd.Series) -> pd.Series:
    """Rank each value as percentile [0, 1] within the series."""
    return series.rank(pct=True, na_option="keep")


# ---------------------------------------------------------------------------
# Main: compute normalized factor scores
# ---------------------------------------------------------------------------

def compute_factor_scores(
    df: pd.DataFrame,
    winsorize_clip: bool = True,
) -> pd.DataFrame:
    """
    Given a raw factor DataFrame, return a new DataFrame of the same shape
    where each factor column is replaced by its SIGNED z-score:
      - Winsorized to remove outlier distortion
      - Z-scored cross-sectionally (within the universe)
      - Multiplied by direction (+1/-1) so higher score ALWAYS = better

    Non-factor columns (ticker, name, sector) are preserved unchanged.

    Args:
        df:             Output from DataFetcher.build_factor_dataframe()
        winsorize_clip: If True, winsorize before z-scoring (recommended)

    Returns:
        pd.DataFrame with same index, factor columns replaced by z-scores.
    """
    factor_cols = [c for c in df.columns if c in FACTOR_DIRECTION]
    meta_cols   = [c for c in df.columns if c not in FACTOR_DIRECTION]

    result = df[meta_cols].copy()

    for col in factor_cols:
        raw = df[col].copy()
        direction = FACTOR_DIRECTION[col]

        # 1. Drop rows with NaN for this factor (only for calculation)
        valid_mask = raw.notna()
        if valid_mask.sum() < 3:
            result[f"z_{col}"] = np.nan
            continue

        s = raw.copy()

        # 2. Winsorize
        if winsorize_clip:
            s[valid_mask] = winsorize(s[valid_mask])

        # 3. Z-score
        z = zscore(s)

        # 4. Apply direction so higher z = always better
        result[f"z_{col}"] = z * direction

    return result


def available_factors(df: pd.DataFrame) -> list[str]:
    """Return factor column names present in df with sufficient non-null data."""
    out = []
    for col in FACTOR_DIRECTION:
        if col in df.columns and df[col].notna().sum() >= 3:
            out.append(col)
    return out
