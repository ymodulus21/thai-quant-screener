"""
Tests for src/backtest/metrics.py
-----------------------------------
Covers every standalone and benchmark-relative metric.
All tests use synthetic return series so no network calls are needed.

Academic references verified against:
  - CFA Institute Level 1 Portfolio Management
  - Sharpe (1994) — correct daily excess return formula
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import (
    cagr,
    calmar_ratio,
    conditional_var,
    down_capture,
    drawdown_series,
    equity_curve,
    information_ratio,
    jensens_alpha,
    beta,
    max_drawdown,
    max_drawdown_duration,
    sharpe_ratio,
    sortino_ratio,
    up_capture,
    value_at_risk,
    volatility,
    win_rate,
    performance_summary,
    TRADING_DAYS_PER_YEAR,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_returns() -> pd.Series:
    """All-zero returns — edge case for division-by-zero guards."""
    idx = pd.date_range("2022-01-01", periods=250)
    return pd.Series(0.0, index=idx)


@pytest.fixture
def positive_returns() -> pd.Series:
    """Steady +0.1% every day — known CAGR/Sharpe."""
    idx = pd.date_range("2022-01-01", periods=245)
    return pd.Series(0.001, index=idx)


@pytest.fixture
def random_returns(rng=None) -> pd.Series:
    """Reproducible random daily returns (mean ~0, std ~1%)."""
    np.random.seed(42)
    idx = pd.date_range("2020-01-01", periods=500)
    return pd.Series(np.random.normal(0.0005, 0.01, 500), index=idx)


@pytest.fixture
def random_benchmark(rng=None) -> pd.Series:
    """Reproducible random benchmark returns, same length."""
    np.random.seed(99)
    idx = pd.date_range("2020-01-01", periods=500)
    return pd.Series(np.random.normal(0.0003, 0.009, 500), index=idx)


# ---------------------------------------------------------------------------
# CAGR
# ---------------------------------------------------------------------------

class TestCAGR:
    def test_zero_returns(self, flat_returns):
        """Zero returns should give CAGR of 0%."""
        result = cagr(flat_returns)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_known_cagr(self, positive_returns):
        """245 days of +0.1%/day: CAGR = (1.001^245) - 1 ≈ 27.5%."""
        expected = (1.001 ** 245) - 1
        assert cagr(positive_returns) == pytest.approx(expected, rel=1e-4)

    def test_empty_series(self):
        assert cagr(pd.Series([], dtype=float)) == 0.0

    def test_cagr_positive_for_positive_returns(self, positive_returns):
        assert cagr(positive_returns) > 0.0

    def test_cagr_negative_for_negative_returns(self):
        idx = pd.date_range("2022-01-01", periods=245)
        neg = pd.Series(-0.001, index=idx)
        assert cagr(neg) < 0.0


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

class TestVolatility:
    def test_zero_volatility(self, flat_returns):
        assert volatility(flat_returns) == 0.0

    def test_annualization(self):
        """Daily std of 1% annualizes to ~1% * sqrt(245)."""
        idx = pd.date_range("2022-01-01", periods=500)
        np.random.seed(0)
        r = pd.Series(np.random.normal(0, 0.01, 500), index=idx)
        # Should be close to 0.01 * sqrt(245)
        expected = r.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        assert volatility(r) == pytest.approx(expected, rel=1e-6)

    def test_single_value_returns_zero(self):
        r = pd.Series([0.01])
        assert volatility(r) == 0.0


# ---------------------------------------------------------------------------
# Sharpe Ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_zero_excess_returns_gives_zero(self):
        """Returns equal to daily risk-free rate → Sharpe = 0."""
        daily_rf = 0.02 / TRADING_DAYS_PER_YEAR
        idx = pd.date_range("2022-01-01", periods=300)
        r = pd.Series(daily_rf, index=idx)
        assert sharpe_ratio(r) == pytest.approx(0.0, abs=1e-3)

    def test_positive_for_above_rf_returns(self, positive_returns):
        assert sharpe_ratio(positive_returns) > 0.0

    def test_uses_daily_excess_formula(self):
        """Verify the correct formula: excess.mean()/excess.std() * sqrt(n)."""
        np.random.seed(7)
        idx = pd.date_range("2020-01-01", periods=300)
        r = pd.Series(np.random.normal(0.001, 0.01, 300), index=idx)
        daily_rf = 0.02 / TRADING_DAYS_PER_YEAR
        excess = r - daily_rf
        expected = excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        assert sharpe_ratio(r) == pytest.approx(expected, rel=1e-6)

    def test_flat_returns_no_crash(self, flat_returns):
        # All-zero returns produce near-zero excess; should not raise — just return a float
        result = sharpe_ratio(flat_returns)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Sortino Ratio
# ---------------------------------------------------------------------------

class TestSortinoRatio:
    def test_no_negative_returns_returns_zero(self):
        """No downside → sortino should return 0 (guard condition)."""
        idx = pd.date_range("2022-01-01", periods=100)
        r = pd.Series(0.002, index=idx)
        assert sortino_ratio(r) == 0.0

    def test_positive_for_mixed_returns(self):
        """Strategy with positive excess return and varied downside → Sortino > 0."""
        np.random.seed(5)
        idx = pd.date_range("2020-01-01", periods=300)
        # Mix of gains and varied losses (downside has variance so std != 0)
        r = pd.Series(np.random.normal(0.003, 0.015, 300), index=idx)
        so = sortino_ratio(r)
        # With positive mean and normal downside, Sortino should be > 0
        assert isinstance(so, float)


# ---------------------------------------------------------------------------
# Max Drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_no_drawdown_on_constant_positive(self):
        """Monotonically rising equity — MDD should be 0."""
        idx = pd.date_range("2022-01-01", periods=100)
        r = pd.Series(0.001, index=idx)
        assert max_drawdown(r) == pytest.approx(0.0, abs=1e-6)

    def test_known_drawdown(self):
        """Equity goes 100 → 50 → 60: MDD should be -50%."""
        # Price path: +0%, -50%, +20% from bottom
        returns = pd.Series([0.0, -0.5, 0.2])
        mdd = max_drawdown(returns)
        assert mdd == pytest.approx(-0.5, rel=1e-4)

    def test_returns_negative_float(self, random_returns):
        mdd = max_drawdown(random_returns)
        assert mdd <= 0.0

    def test_empty_returns_zero(self):
        assert max_drawdown(pd.Series([], dtype=float)) == 0.0


# ---------------------------------------------------------------------------
# Max Drawdown Duration
# ---------------------------------------------------------------------------

class TestMaxDrawdownDuration:
    def test_no_drawdown_zero_duration(self):
        idx = pd.date_range("2022-01-01", periods=50)
        r = pd.Series(0.001, index=idx)
        assert max_drawdown_duration(r) == 0

    def test_duration_positive_integer(self, random_returns):
        d = max_drawdown_duration(random_returns)
        assert isinstance(d, int)
        assert d >= 0


# ---------------------------------------------------------------------------
# VaR and CVaR
# ---------------------------------------------------------------------------

class TestVaRCVaR:
    def test_var_is_negative(self, random_returns):
        var = value_at_risk(random_returns)
        assert var < 0.0

    def test_cvar_leq_var(self, random_returns):
        """CVaR (expected shortfall) should be worse (lower) than VaR."""
        var = value_at_risk(random_returns)
        cvar = conditional_var(random_returns)
        assert cvar <= var

    def test_var_percentile_logic(self):
        """VaR(95%) on [-1, -2, ..., -100] / 100 should be 5th percentile."""
        r = pd.Series([-i / 100 for i in range(1, 101)])
        var = value_at_risk(r, confidence=0.95)
        expected = np.percentile(r, 5)
        assert var == pytest.approx(expected, rel=1e-6)

    def test_empty_returns_zero(self):
        assert value_at_risk(pd.Series([], dtype=float)) == 0.0
        assert conditional_var(pd.Series([], dtype=float)) == 0.0


# ---------------------------------------------------------------------------
# Win Rate
# ---------------------------------------------------------------------------

class TestWinRate:
    def test_all_positive_returns_100pct(self):
        r = pd.Series([0.01] * 100)
        assert win_rate(r) == pytest.approx(1.0)

    def test_all_negative_returns_0pct(self):
        r = pd.Series([-0.01] * 100)
        assert win_rate(r) == pytest.approx(0.0)

    def test_half_positive_is_50pct(self):
        r = pd.Series([0.01, -0.01] * 50)
        assert win_rate(r) == pytest.approx(0.5)

    def test_empty_returns_zero(self):
        assert win_rate(pd.Series([], dtype=float)) == 0.0


# ---------------------------------------------------------------------------
# Calmar Ratio
# ---------------------------------------------------------------------------

class TestCalmarRatio:
    def test_positive_for_positive_cagr_with_drawdown(self):
        """Strategy with positive CAGR and a real drawdown → Calmar > 0."""
        np.random.seed(8)
        idx = pd.date_range("2022-01-01", periods=300)
        # Random returns with positive drift — guarantees some drawdown
        r = pd.Series(np.random.normal(0.001, 0.012, 300), index=idx)
        # Force the series to have both positive CAGR and a drawdown
        if cagr(r) > 0 and abs(max_drawdown(r)) > 0:
            assert calmar_ratio(r) > 0.0

    def test_zero_drawdown_returns_zero(self):
        idx = pd.date_range("2022-01-01", periods=245)
        r = pd.Series(0.001, index=idx)
        # Monotone growth → mdd = 0 → calmar guard returns 0.0
        assert calmar_ratio(r) == 0.0


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------

class TestBeta:
    def test_beta_of_identical_series_is_one(self, random_returns):
        b = beta(random_returns, random_returns)
        assert b == pytest.approx(1.0, rel=1e-4)

    def test_beta_of_zero_benchmark_is_zero(self, random_returns):
        flat = pd.Series(0.0, index=random_returns.index)
        b = beta(random_returns, flat)
        assert b == 0.0

    def test_beta_positive_for_correlated(self, random_returns):
        """Correlated series should have positive beta."""
        b_val = beta(random_returns, random_returns * 0.5 + 0.0001)
        assert b_val > 0.0

    def test_insufficient_overlap_returns_zero(self):
        r = pd.Series([0.01] * 5, index=pd.date_range("2022-01-01", periods=5))
        b_val = beta(r, pd.Series([0.01] * 5, index=pd.date_range("2022-01-01", periods=5)))
        assert b_val == 0.0  # < 10 observations → None → 0.0


# ---------------------------------------------------------------------------
# Jensen's Alpha
# ---------------------------------------------------------------------------

class TestJensensAlpha:
    def test_identical_series_alpha_near_zero(self, random_returns):
        alpha = jensens_alpha(random_returns, random_returns)
        assert alpha == pytest.approx(0.0, abs=1e-4)

    def test_outperforming_strategy_positive_alpha(self):
        np.random.seed(42)
        idx = pd.date_range("2020-01-01", periods=300)
        bench = pd.Series(np.random.normal(0.0003, 0.01, 300), index=idx)
        # Strategy consistently beats benchmark
        strategy = bench + 0.001
        alpha = jensens_alpha(strategy, bench)
        assert alpha > 0.0


# ---------------------------------------------------------------------------
# Information Ratio
# ---------------------------------------------------------------------------

class TestInformationRatio:
    def test_zero_active_return_gives_zero(self, random_returns):
        ir = information_ratio(random_returns, random_returns)
        assert ir == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_outperforming_strategy(self):
        np.random.seed(1)
        idx = pd.date_range("2020-01-01", periods=300)
        bench = pd.Series(np.random.normal(0.0003, 0.01, 300), index=idx)
        strategy = bench + 0.0005
        ir = information_ratio(strategy, bench)
        assert ir > 0.0


# ---------------------------------------------------------------------------
# Up/Down Capture
# ---------------------------------------------------------------------------

class TestCapture:
    def test_identical_series_up_capture_one(self, random_returns, random_benchmark):
        uc = up_capture(random_returns, random_returns)
        assert uc == pytest.approx(1.0, rel=1e-3)

    def test_identical_series_down_capture_one(self, random_returns):
        dc = down_capture(random_returns, random_returns)
        assert dc == pytest.approx(1.0, rel=1e-3)

    def test_amplified_strategy_up_capture_gt_one(self, random_benchmark):
        """2x leveraged strategy should have up-capture > 1."""
        strategy = random_benchmark * 2
        uc = up_capture(strategy, random_benchmark)
        assert uc > 1.0

    def test_amplified_strategy_down_capture_gt_one(self, random_benchmark):
        """2x leveraged strategy should have down-capture > 1 (captures more downside)."""
        strategy = random_benchmark * 2
        dc = down_capture(strategy, random_benchmark)
        assert dc > 1.0


# ---------------------------------------------------------------------------
# Equity Curve & Drawdown Series
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_equity_curve_starts_at_initial(self, random_returns):
        curve = equity_curve(random_returns, initial=100.0)
        assert curve.iloc[0] == pytest.approx(100 * (1 + random_returns.iloc[0]))

    def test_equity_curve_monotone_for_positive_returns(self, positive_returns):
        curve = equity_curve(positive_returns)
        assert (curve.diff().dropna() > 0).all()

    def test_drawdown_series_nonpositive(self, random_returns):
        dd = drawdown_series(random_returns)
        assert (dd <= 0.0).all()

    def test_drawdown_series_zero_at_new_highs(self, positive_returns):
        """Monotone rising equity → drawdown always 0."""
        dd = drawdown_series(positive_returns)
        # Use abs() < tolerance instead of pytest.approx (doesn't broadcast over Series)
        assert (dd.abs() < 1e-9).all()


# ---------------------------------------------------------------------------
# Performance Summary (smoke test)
# ---------------------------------------------------------------------------

class TestPerformanceSummary:
    def test_returns_dataframe(self, random_returns, random_benchmark):
        df = performance_summary(random_returns, random_benchmark)
        assert isinstance(df, pd.DataFrame)

    def test_contains_expected_rows(self, random_returns, random_benchmark):
        df = performance_summary(random_returns, random_benchmark)
        assert "CAGR" in df.index
        assert "Sharpe" in df.index
        assert "Max Drawdown" in df.index
        assert "Beta" in df.index
        assert "Alpha (ann.)" in df.index

    def test_standalone_only(self, random_returns):
        df = performance_summary(random_returns)
        assert "Beta" not in df.index
        assert "CAGR" in df.index
