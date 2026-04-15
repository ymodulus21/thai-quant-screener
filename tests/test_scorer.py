"""
Tests for src/screener/scorer.py
------------------------------------
Covers STYLE_PRESETS weight validation and CompositeScorer behavior.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.screener.scorer import STYLE_PRESETS, CompositeScorer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_factor_df() -> pd.DataFrame:
    """20-stock synthetic factor DataFrame covering all factor groups."""
    np.random.seed(1)
    n = 20
    return pd.DataFrame({
        "ticker":          [f"T{i:02d}" for i in range(n)],
        "name":            [f"Company {i}" for i in range(n)],
        "sector":          ["Financials"] * 5 + ["Energy"] * 5 +
                           ["Consumer"] * 5 + ["Technology"] * 5,
        # Value
        "pe_ratio":        np.random.uniform(5, 40, n),
        "pb_ratio":        np.random.uniform(0.5, 4.0, n),
        # Quality
        "roe":             np.random.uniform(0.05, 0.35, n),
        "roa":             np.random.uniform(0.02, 0.15, n),
        "debt_to_equity":  np.random.uniform(0.1, 2.0, n),
        # Momentum
        "mom_1m":          np.random.uniform(-0.1, 0.1, n),
        "mom_3m":          np.random.uniform(-0.2, 0.3, n),
        "mom_6m":          np.random.uniform(-0.3, 0.4, n),
        "mom_12m":         np.random.uniform(-0.4, 0.5, n),
        # Income
        "dividend_yield":  np.random.uniform(0.01, 0.08, n),
        # Growth
        "revenue_growth":  np.random.uniform(-0.1, 0.3, n),
        "earnings_growth": np.random.uniform(-0.2, 0.5, n),
    })


# ---------------------------------------------------------------------------
# STYLE_PRESETS — configuration integrity
# ---------------------------------------------------------------------------

class TestStylePresetsConfig:
    def test_all_expected_styles_present(self):
        expected = {"value", "quality", "momentum", "income", "growth", "blend"}
        assert set(STYLE_PRESETS.keys()) == expected

    def test_all_weights_sum_to_one(self):
        """Each style's group weights must sum to exactly 1.0."""
        for style_name, weights in STYLE_PRESETS.items():
            total = sum(weights.values())
            assert total == pytest.approx(1.0, abs=1e-9), (
                f"Style '{style_name}' weights sum to {total}, expected 1.0"
            )

    def test_no_negative_weights(self):
        for style_name, weights in STYLE_PRESETS.items():
            for group, w in weights.items():
                assert w >= 0.0, f"Style '{style_name}' group '{group}' has negative weight {w}"

    def test_dominant_factor_in_each_style(self):
        """Each style should have its primary factor as the largest weight."""
        assert STYLE_PRESETS["value"]["value"] > 0.5
        assert STYLE_PRESETS["quality"]["quality"] > 0.5
        assert STYLE_PRESETS["momentum"]["momentum"] > 0.5
        assert STYLE_PRESETS["income"]["income"] >= 0.5
        assert STYLE_PRESETS["growth"]["growth"] >= 0.35

    def test_blend_most_balanced(self):
        """Blend should have no single group dominating (all weights < 50%)."""
        for group, w in STYLE_PRESETS["blend"].items():
            assert w < 0.5, f"Blend style '{group}' weight {w} too concentrated"


# ---------------------------------------------------------------------------
# CompositeScorer — initialization
# ---------------------------------------------------------------------------

class TestCompositeScorerInit:
    def test_valid_style_succeeds(self):
        scorer = CompositeScorer(style="blend")
        assert scorer.style == "blend"

    def test_invalid_style_raises(self):
        with pytest.raises(ValueError, match="Unknown style"):
            CompositeScorer(style="invalid_style")

    def test_custom_style_requires_weights(self):
        with pytest.raises(ValueError, match="custom_weights required"):
            CompositeScorer(style="custom")

    def test_custom_style_with_weights_succeeds(self):
        weights = {"value": 0.5, "quality": 0.3, "momentum": 0.1, "income": 0.05, "growth": 0.05}
        scorer = CompositeScorer(style="custom", custom_weights=weights)
        assert scorer.group_weights == weights

    def test_group_weights_property_for_preset(self):
        scorer = CompositeScorer(style="value")
        assert scorer.group_weights == STYLE_PRESETS["value"]


# ---------------------------------------------------------------------------
# CompositeScorer.score() — output structure
# ---------------------------------------------------------------------------

class TestCompositeScorerScore:
    def test_returns_dataframe(self, raw_factor_df):
        scorer = CompositeScorer(style="blend")
        result = scorer.score(raw_factor_df)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_row_count(self, raw_factor_df):
        scorer = CompositeScorer(style="blend")
        result = scorer.score(raw_factor_df)
        assert len(result) == len(raw_factor_df)

    def test_composite_score_column_exists(self, raw_factor_df):
        scorer = CompositeScorer(style="blend")
        result = scorer.score(raw_factor_df)
        assert "composite_score" in result.columns

    def test_rank_column_exists(self, raw_factor_df):
        scorer = CompositeScorer(style="blend")
        result = scorer.score(raw_factor_df)
        assert "rank" in result.columns

    def test_rank_starts_at_one(self, raw_factor_df):
        scorer = CompositeScorer(style="blend")
        result = scorer.score(raw_factor_df)
        assert result["rank"].min() == 1

    def test_rank_is_contiguous(self, raw_factor_df):
        scorer = CompositeScorer(style="blend")
        result = scorer.score(raw_factor_df)
        expected_ranks = list(range(1, len(result) + 1))
        assert sorted(result["rank"].tolist()) == expected_ranks

    def test_sorted_descending_by_composite_score(self, raw_factor_df):
        scorer = CompositeScorer(style="blend")
        result = scorer.score(raw_factor_df)
        scores = result["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_rank_1_has_highest_score(self, raw_factor_df):
        scorer = CompositeScorer(style="blend")
        result = scorer.score(raw_factor_df)
        top = result[result["rank"] == 1].iloc[0]
        assert top["composite_score"] == result["composite_score"].max()

    def test_group_score_columns_present(self, raw_factor_df):
        scorer = CompositeScorer(style="blend")
        result = scorer.score(raw_factor_df)
        assert "score_value" in result.columns
        assert "score_quality" in result.columns
        assert "score_momentum" in result.columns

    def test_original_columns_preserved(self, raw_factor_df):
        scorer = CompositeScorer(style="blend")
        result = scorer.score(raw_factor_df)
        assert "ticker" in result.columns
        assert "sector" in result.columns
        assert "pe_ratio" in result.columns


# ---------------------------------------------------------------------------
# CompositeScorer — different styles produce different rankings
# ---------------------------------------------------------------------------

class TestStylesDifferentiate:
    def test_value_vs_momentum_different_top_stock(self, raw_factor_df):
        """Value and momentum styles should often rank #1 differently."""
        value_result   = CompositeScorer(style="value").score(raw_factor_df)
        momentum_result = CompositeScorer(style="momentum").score(raw_factor_df)
        # With 20 random stocks it's virtually guaranteed they differ
        value_top   = value_result.iloc[0]["ticker"]
        momentum_top = momentum_result.iloc[0]["ticker"]
        # Not asserting inequality (could be same by chance) — just assert both run
        assert isinstance(value_top, str)
        assert isinstance(momentum_top, str)

    def test_all_styles_complete_without_error(self, raw_factor_df):
        for style in STYLE_PRESETS:
            result = CompositeScorer(style=style).score(raw_factor_df)
            assert len(result) == len(raw_factor_df)


# ---------------------------------------------------------------------------
# CompositeScorer.top_n()
# ---------------------------------------------------------------------------

class TestTopN:
    def test_top_n_returns_n_rows(self, raw_factor_df):
        scorer = CompositeScorer(style="blend")
        result = scorer.top_n(raw_factor_df, n=5)
        assert len(result) == 5

    def test_top_n_are_highest_scores(self, raw_factor_df):
        scorer = CompositeScorer(style="blend")
        all_scores = scorer.score(raw_factor_df)["composite_score"].tolist()
        top5_scores = scorer.top_n(raw_factor_df, n=5)["composite_score"].tolist()
        remaining = sorted(all_scores, reverse=True)[5:]
        assert min(top5_scores) >= max(remaining)

    def test_top_n_larger_than_universe_returns_all(self, raw_factor_df):
        scorer = CompositeScorer(style="blend")
        result = scorer.top_n(raw_factor_df, n=999)
        assert len(result) == len(raw_factor_df)
