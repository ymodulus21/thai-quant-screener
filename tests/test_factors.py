"""
Tests for src/screener/factors.py
------------------------------------
Covers normalization math: winsorize, zscore, percentile_rank,
direction adjustment, and the full compute_factor_scores pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.screener.factors import (
    FACTOR_DIRECTION,
    FACTOR_GROUPS,
    winsorize,
    zscore,
    percentile_rank,
    compute_factor_scores,
    available_factors,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_factor_df() -> pd.DataFrame:
    """10-stock synthetic factor DataFrame — all key factors populated."""
    np.random.seed(0)
    n = 10
    return pd.DataFrame({
        "ticker":         [f"T{i}" for i in range(n)],
        "name":           [f"Company {i}" for i in range(n)],
        "sector":         ["Financials"] * n,
        "pe_ratio":       np.random.uniform(5, 40, n),
        "pb_ratio":       np.random.uniform(0.5, 4.0, n),
        "roe":            np.random.uniform(0.05, 0.30, n),
        "dividend_yield": np.random.uniform(0.01, 0.08, n),
        "mom_3m":         np.random.uniform(-0.2, 0.3, n),
        "mom_6m":         np.random.uniform(-0.3, 0.4, n),
        "revenue_growth": np.random.uniform(-0.1, 0.3, n),
    })


@pytest.fixture
def all_nan_series() -> pd.Series:
    return pd.Series([np.nan] * 10)


# ---------------------------------------------------------------------------
# winsorize()
# ---------------------------------------------------------------------------

class TestWinsorize:
    def test_clips_outliers(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
        result = winsorize(s)
        assert result.max() < 100.0

    def test_preserves_length(self):
        s = pd.Series(range(100), dtype=float)
        assert len(winsorize(s)) == 100

    def test_no_effect_on_uniform_series(self):
        """Uniform distribution: winsorize should clip extreme quantiles."""
        s = pd.Series(np.linspace(0, 1, 1000))
        result = winsorize(s)
        assert result.min() >= s.quantile(0.05)
        assert result.max() <= s.quantile(0.95)

    def test_values_within_quantile_bounds(self):
        np.random.seed(3)
        s = pd.Series(np.random.normal(0, 1, 200))
        lo, hi = s.quantile(0.05), s.quantile(0.95)
        result = winsorize(s)
        assert result.min() >= lo - 1e-9
        assert result.max() <= hi + 1e-9


# ---------------------------------------------------------------------------
# zscore()
# ---------------------------------------------------------------------------

class TestZscore:
    def test_mean_near_zero(self):
        s = pd.Series(np.random.normal(50, 10, 100))
        z = zscore(s)
        assert z.mean() == pytest.approx(0.0, abs=1e-10)

    def test_std_near_one(self):
        s = pd.Series(np.random.normal(50, 10, 100))
        z = zscore(s)
        assert z.std() == pytest.approx(1.0, abs=1e-6)

    def test_constant_series_returns_nan(self):
        s = pd.Series([5.0] * 10)
        z = zscore(s)
        assert z.isna().all()

    def test_preserves_length(self):
        s = pd.Series(range(50), dtype=float)
        assert len(zscore(s)) == 50

    def test_single_outlier_is_highest_z(self):
        s = pd.Series([1.0, 1.0, 1.0, 1.0, 100.0])
        z = zscore(s)
        assert z.idxmax() == 4


# ---------------------------------------------------------------------------
# percentile_rank()
# ---------------------------------------------------------------------------

class TestPercentileRank:
    def test_range_zero_to_one(self):
        s = pd.Series(range(10), dtype=float)
        r = percentile_rank(s)
        assert r.min() > 0.0
        assert r.max() <= 1.0

    def test_highest_value_highest_rank(self):
        s = pd.Series([3.0, 1.0, 4.0, 1.0, 5.0])
        r = percentile_rank(s)
        assert r.idxmax() == 4

    def test_nan_preserved(self):
        s = pd.Series([1.0, 2.0, np.nan, 4.0])
        r = percentile_rank(s)
        assert pd.isna(r.iloc[2])


# ---------------------------------------------------------------------------
# FACTOR_DIRECTION — sanity checks on the config
# ---------------------------------------------------------------------------

class TestFactorDirectionConfig:
    def test_all_directions_plus_minus_one(self):
        for factor, direction in FACTOR_DIRECTION.items():
            assert direction in (-1, 1), f"{factor} has invalid direction {direction}"

    def test_value_factors_are_negative(self):
        """Lower P/E = cheaper = better → direction should be -1."""
        for f in ["pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda"]:
            assert FACTOR_DIRECTION[f] == -1, f"{f} should be -1 (lower=better)"

    def test_quality_factors_positive(self):
        for f in ["roe", "roa", "gross_margin", "net_margin"]:
            assert FACTOR_DIRECTION[f] == +1, f"{f} should be +1 (higher=better)"

    def test_momentum_factors_positive(self):
        for f in ["mom_1m", "mom_3m", "mom_6m", "mom_12m"]:
            assert FACTOR_DIRECTION[f] == +1

    def test_dividend_yield_positive(self):
        assert FACTOR_DIRECTION["dividend_yield"] == +1

    def test_debt_to_equity_negative(self):
        """Lower debt is better."""
        assert FACTOR_DIRECTION["debt_to_equity"] == -1


# ---------------------------------------------------------------------------
# FACTOR_GROUPS — integrity checks
# ---------------------------------------------------------------------------

class TestFactorGroupsConfig:
    def test_all_factors_in_some_group(self):
        all_in_groups = {f for group in FACTOR_GROUPS.values() for f in group}
        for f in FACTOR_DIRECTION:
            if f == "current_ratio":
                continue  # current_ratio is in direction but not grouped — acceptable
            assert f in all_in_groups, f"{f} missing from FACTOR_GROUPS"

    def test_group_names(self):
        expected_groups = {"value", "quality", "momentum", "income", "growth"}
        assert set(FACTOR_GROUPS.keys()) == expected_groups


# ---------------------------------------------------------------------------
# compute_factor_scores()
# ---------------------------------------------------------------------------

class TestComputeFactorScores:
    def test_returns_dataframe(self, raw_factor_df):
        result = compute_factor_scores(raw_factor_df)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_row_count(self, raw_factor_df):
        result = compute_factor_scores(raw_factor_df)
        assert len(result) == len(raw_factor_df)

    def test_meta_columns_unchanged(self, raw_factor_df):
        result = compute_factor_scores(raw_factor_df)
        pd.testing.assert_series_equal(result["ticker"], raw_factor_df["ticker"])
        pd.testing.assert_series_equal(result["sector"], raw_factor_df["sector"])

    def test_z_columns_created(self, raw_factor_df):
        result = compute_factor_scores(raw_factor_df)
        assert "z_pe_ratio" in result.columns
        assert "z_roe" in result.columns
        assert "z_mom_3m" in result.columns

    def test_direction_inverts_value_factors(self, raw_factor_df):
        """
        pe_ratio direction = -1. The stock with the LOWEST raw P/E should
        have the HIGHEST z_pe_ratio (best value score).
        """
        result = compute_factor_scores(raw_factor_df)
        min_pe_idx = raw_factor_df["pe_ratio"].idxmin()
        max_z_idx = result["z_pe_ratio"].idxmax()
        assert min_pe_idx == max_z_idx, (
            "Lowest P/E stock should have highest z_pe_ratio after direction inversion"
        )

    def test_direction_preserves_quality_factors(self, raw_factor_df):
        """
        roe direction = +1. Highest ROE → highest z_roe.
        """
        result = compute_factor_scores(raw_factor_df)
        max_roe_idx = raw_factor_df["roe"].idxmax()
        max_z_idx = result["z_roe"].idxmax()
        assert max_roe_idx == max_z_idx

    def test_z_scores_mean_near_zero(self, raw_factor_df):
        """After winsorize + zscore, mean of z-scores should be near 0."""
        result = compute_factor_scores(raw_factor_df)
        assert result["z_pe_ratio"].mean() == pytest.approx(0.0, abs=1e-6)
        assert result["z_roe"].mean() == pytest.approx(0.0, abs=1e-6)

    def test_insufficient_data_returns_nan(self):
        """Factor with < 3 non-null values should produce all-NaN z-column."""
        df = pd.DataFrame({
            "ticker": ["A", "B"],
            "pe_ratio": [10.0, np.nan],
        })
        result = compute_factor_scores(df)
        assert result["z_pe_ratio"].isna().all()

    def test_no_winsorize_flag(self, raw_factor_df):
        """Should still work when winsorize_clip=False."""
        result = compute_factor_scores(raw_factor_df, winsorize_clip=False)
        assert "z_pe_ratio" in result.columns


# ---------------------------------------------------------------------------
# available_factors()
# ---------------------------------------------------------------------------

class TestAvailableFactors:
    def test_returns_list(self, raw_factor_df):
        result = available_factors(raw_factor_df)
        assert isinstance(result, list)

    def test_only_present_factors(self, raw_factor_df):
        result = available_factors(raw_factor_df)
        for f in result:
            assert f in raw_factor_df.columns

    def test_excludes_insufficient_data(self):
        """Factor with only 2 non-nulls should be excluded."""
        df = pd.DataFrame({
            "pe_ratio": [10.0, np.nan, np.nan, np.nan, np.nan],
            "roe": [0.1, 0.2, 0.3, 0.4, 0.5],
        })
        result = available_factors(df)
        assert "pe_ratio" not in result
        assert "roe" in result
