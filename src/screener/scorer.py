"""
Multi-Factor Composite Scorer
------------------------------
Combines individual factor z-scores into a single composite score
using configurable style weights.

Investment styles (for CFA context):
  Value    — buy cheap stocks (low P/E, P/B)
  Quality  — buy high-profitability, low-debt stocks
  Momentum — buy recent winners (trend-following)
  Income   — maximize dividend yield
  Blend    — balanced across all dimensions
  Custom   — user-defined weights
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .factors import FACTOR_GROUPS, compute_factor_scores

# ---------------------------------------------------------------------------
# Style preset weight configs
# Weights are per-factor-GROUP, then distributed equally within group
# ---------------------------------------------------------------------------

STYLE_PRESETS: dict[str, dict[str, float]] = {
    "value": {
        "value":    0.70,
        "quality":  0.20,
        "momentum": 0.05,
        "income":   0.05,
        "growth":   0.00,
    },
    "quality": {
        "value":    0.20,
        "quality":  0.60,
        "momentum": 0.10,
        "income":   0.05,
        "growth":   0.05,
    },
    "momentum": {
        "value":    0.10,
        "quality":  0.10,
        "momentum": 0.75,
        "income":   0.05,
        "growth":   0.00,
    },
    "income": {
        "value":    0.20,
        "quality":  0.20,
        "momentum": 0.05,
        "income":   0.50,
        "growth":   0.05,
    },
    "growth": {
        "value":    0.10,
        "quality":  0.30,
        "momentum": 0.20,
        "income":   0.00,
        "growth":   0.40,
    },
    "blend": {
        "value":    0.25,
        "quality":  0.30,
        "momentum": 0.20,
        "income":   0.10,
        "growth":   0.15,
    },
}


# ---------------------------------------------------------------------------
# CompositeScorer
# ---------------------------------------------------------------------------

@dataclass
class CompositeScorer:
    """
    Turns a raw factor DataFrame into a ranked composite score.

    Usage:
        scorer = CompositeScorer(style="blend")
        ranked_df = scorer.score(raw_factor_df)
        top10 = ranked_df.head(10)
    """
    style: str = "blend"
    custom_weights: Optional[dict[str, float]] = None  # per-group weights

    def __post_init__(self) -> None:
        if self.style == "custom" and not self.custom_weights:
            raise ValueError("custom_weights required when style='custom'")
        if self.style != "custom" and self.style not in STYLE_PRESETS:
            raise ValueError(f"Unknown style '{self.style}'. Choose: {list(STYLE_PRESETS)}")

    @property
    def group_weights(self) -> dict[str, float]:
        if self.style == "custom":
            return self.custom_weights  # type: ignore[return-value]
        return STYLE_PRESETS[self.style]

    def score(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute composite score for each stock.

        Steps:
          1. Normalize factors to z-scores (compute_factor_scores)
          2. Average z-scores within each factor group
          3. Weight group scores by style preset
          4. Sum → composite score
          5. Sort descending, add rank column

        Returns:
            DataFrame with original columns + group scores +
            'composite_score' + 'rank', sorted best→worst.
        """
        scored = compute_factor_scores(raw_df)

        group_scores: dict[str, pd.Series] = {}

        for group, factors in FACTOR_GROUPS.items():
            z_cols = [f"z_{f}" for f in factors if f"z_{f}" in scored.columns]
            if not z_cols:
                continue
            # Average non-null z-scores within group
            group_scores[f"score_{group}"] = scored[z_cols].mean(axis=1, skipna=True)

        # Build result DataFrame
        result = raw_df.copy()
        for col, series in group_scores.items():
            result[col] = series

        # Composite = weighted sum of group scores
        composite = pd.Series(0.0, index=result.index)
        total_weight = 0.0

        for group, weight in self.group_weights.items():
            col = f"score_{group}"
            if col in result.columns and weight > 0:
                valid = result[col].notna()
                composite[valid] += result.loc[valid, col] * weight
                total_weight += weight

        if total_weight > 0:
            composite /= total_weight

        result["composite_score"] = composite.round(4)
        result = result.sort_values("composite_score", ascending=False)
        result["rank"] = range(1, len(result) + 1)

        return result.reset_index(drop=True)

    def top_n(self, raw_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Score and return top N stocks only."""
        return self.score(raw_df).head(n)
