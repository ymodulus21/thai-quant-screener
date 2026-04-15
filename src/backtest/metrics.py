"""
Performance Metrics
--------------------
Complete tearsheet metrics used in:
  - CFA L1/L2 Portfolio Management
  - Quant finance strategy evaluation
  - Professional fund reporting

All functions accept a pd.Series of PERIODIC returns (not prices).

Metrics ported and extended from thai-stock-backtestv2:
  + Beta, Alpha (Jensen's), Information Ratio
  + VaR (95%), CVaR / Expected Shortfall (95%)
  + Up/Down Market Capture Ratio
  + Max Drawdown Duration
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Thai market: ~245 trading days/year
TRADING_DAYS_PER_YEAR = 245
RISK_FREE_RATE = 0.02   # ~2% Thai risk-free proxy (fixed deposit / T-bill)


# ---------------------------------------------------------------------------
# Standalone Return Metrics
# ---------------------------------------------------------------------------

def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Compound Annual Growth Rate.
    CFA: (ending_value / beginning_value)^(1/years) - 1
    """
    n = len(returns)
    if n == 0:
        return 0.0
    total = (1 + returns).prod()
    years = n / periods_per_year
    if years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1 / years) - 1)


def volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized standard deviation of returns."""
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Sharpe Ratio = (E[R] - Rf) / σ
    Uses DAILY excess returns then annualizes — correct formula.
    CFA: reward per unit of total risk.
    """
    daily_rf = risk_free / periods_per_year
    excess = returns - daily_rf
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Sortino = (E[R] - Rf) / downside_deviation
    Only penalizes negative returns — better for skewed distributions.
    CFA: relevant when return distribution is asymmetric.
    """
    daily_rf = risk_free / periods_per_year
    excess = returns - daily_rf
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = float(downside.std() * np.sqrt(periods_per_year))
    if downside_std == 0:
        return 0.0
    return float(excess.mean() * periods_per_year / downside_std)


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum peak-to-trough decline.
    Returns negative float e.g. -0.35 means -35%.
    CFA: critical tail-risk metric.
    """
    if len(returns) == 0:
        return 0.0
    cum = (1 + returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    return float(dd.min())


def max_drawdown_duration(returns: pd.Series) -> int:
    """
    Longest consecutive period spent underwater (in trading days).
    High duration = strategy takes long to recover — a hidden risk not shown by MDD alone.
    """
    if len(returns) == 0:
        return 0
    cum = (1 + returns).cumprod()
    underwater = (cum < cum.cummax()).astype(int)
    # Count consecutive underwater days
    groups = underwater * (underwater.groupby((underwater == 0).cumsum()).cumcount() + 1)
    return int(groups.max()) if len(groups) > 0 else 0


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Calmar = CAGR / |Max Drawdown|
    Higher = better return per unit of worst drawdown.
    """
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return float(cagr(returns, periods_per_year) / mdd)


def win_rate(returns: pd.Series) -> float:
    """Fraction of periods with positive return."""
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR at given confidence level.
    VaR(95%) = worst daily loss we expect to NOT exceed 95% of the time.
    CFA/FRM: standard market risk measure for regulatory reporting.
    Returns negative float e.g. -0.025 = 2.5% daily loss at 95% confidence.
    """
    if len(returns) == 0:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    CVaR / Expected Shortfall (ES) at given confidence level.
    Average loss GIVEN that we are beyond VaR threshold.
    CVaR > VaR: tells you HOW BAD the tail losses are, not just where they start.
    CFA: superior to VaR for non-normal distributions (fat tails).
    Returns negative float.
    """
    if len(returns) == 0:
        return 0.0
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else var


# ---------------------------------------------------------------------------
# Benchmark-Relative Metrics (require benchmark_returns)
# ---------------------------------------------------------------------------

def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Beta = Cov(R_portfolio, R_benchmark) / Var(R_benchmark)
    CFA: systematic (market) risk. Beta > 1 = amplified market moves.
    Beta < 1 = defensive. Beta < 0 = hedged / inverse.
    """
    aligned = _align(returns, benchmark_returns)
    if aligned is None:
        return 0.0
    r, b = aligned
    var_b = b.var()
    if var_b == 0:
        return 0.0
    return float(np.cov(r, b)[0, 1] / var_b)


def jensens_alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free: float = RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Jensen's Alpha (annualized) = R_p - [Rf + Beta * (R_m - Rf)]
    CFA: excess return AFTER adjusting for market risk taken.
    Positive alpha = genuine skill, not just market exposure.
    """
    aligned = _align(returns, benchmark_returns)
    if aligned is None:
        return 0.0
    r, b = aligned
    b_val = beta(r, b)
    daily_rf = risk_free / periods_per_year
    alpha_daily = (r.mean() - daily_rf) - b_val * (b.mean() - daily_rf)
    return float(alpha_daily * periods_per_year)


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    IR = Active Return / Tracking Error
    CFA: how much excess return per unit of active risk taken.
    IR > 0.5 = good active manager. IR > 1.0 = exceptional.
    """
    aligned = _align(returns, benchmark_returns)
    if aligned is None:
        return 0.0
    r, b = aligned
    active = r - b
    te = active.std() * np.sqrt(periods_per_year)
    if te == 0:
        return 0.0
    return float(active.mean() * periods_per_year / te)


def up_capture(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Up Capture Ratio = avg portfolio return on UP days / avg benchmark return on UP days.
    > 100% = outperforms benchmark when market rises.
    CFA: useful for understanding directional return profile.
    """
    aligned = _align(returns, benchmark_returns)
    if aligned is None:
        return 0.0
    r, b = aligned
    mask = b > 0
    if mask.sum() == 0 or b[mask].mean() == 0:
        return 0.0
    return float(r[mask].mean() / b[mask].mean())


def down_capture(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Down Capture Ratio = avg portfolio return on DOWN days / avg benchmark return on DOWN days.
    < 100% = loses less than benchmark when market falls. Lower is better.
    CFA: paired with up_capture to assess asymmetric return profile.
    """
    aligned = _align(returns, benchmark_returns)
    if aligned is None:
        return 0.0
    r, b = aligned
    mask = b < 0
    if mask.sum() == 0 or b[mask].mean() == 0:
        return 0.0
    return float(r[mask].mean() / b[mask].mean())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def equity_curve(returns: pd.Series, initial: float = 100.0) -> pd.Series:
    """Cumulative portfolio value starting at `initial`."""
    return initial * (1 + returns).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Full drawdown time series (negative values)."""
    cum = (1 + returns).cumprod()
    return (cum - cum.cummax()) / cum.cummax()


def _align(r: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series] | None:
    """Align two return series to common index, require at least 10 observations."""
    common = r.index.intersection(b.index)
    if len(common) < 10:
        return None
    return r.loc[common], b.loc[common]


# ---------------------------------------------------------------------------
# Full Tearsheet Summary
# ---------------------------------------------------------------------------

def performance_summary(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    label: str = "Strategy",
    risk_free: float = RISK_FREE_RATE,
) -> pd.DataFrame:
    """
    Professional performance tearsheet — two sections:
      1. Standalone metrics (apply to any return series)
      2. Benchmark-relative metrics (only if benchmark provided)

    Args:
        returns:           Strategy daily returns
        benchmark_returns: Benchmark daily returns (e.g. SET index)
        label:             Strategy display name
        risk_free:         Annual risk-free rate

    Returns:
        pd.DataFrame — rows = metrics, columns = [strategy, benchmark]
    """
    has_bench = benchmark_returns is not None

    def _standalone(r: pd.Series, name: str) -> dict:
        return {
            "name": name,
            # Returns
            "Total Return":       f"{(1 + r).prod() - 1:.2%}",
            "CAGR":               f"{cagr(r):.2%}",
            # Risk
            "Volatility":         f"{volatility(r):.2%}",
            "Max Drawdown":       f"{max_drawdown(r):.2%}",
            "Max DD Duration":    f"{max_drawdown_duration(r)} days",
            "VaR (95%)":          f"{value_at_risk(r):.2%}",
            "CVaR (95%)":         f"{conditional_var(r):.2%}",
            # Risk-adjusted
            "Sharpe":             f"{sharpe_ratio(r, risk_free):.2f}",
            "Sortino":            f"{sortino_ratio(r, risk_free):.2f}",
            "Calmar":             f"{calmar_ratio(r):.2f}",
            "Win Rate":           f"{win_rate(r):.2%}",
        }

    rows = [_standalone(returns, label)]
    if has_bench:
        rows.append(_standalone(benchmark_returns, "SET Index"))

    df = pd.DataFrame(rows).set_index("name").T
    df.index.name = "Metric"

    # Benchmark-relative section (only if benchmark provided)
    if has_bench:
        rel_data = {
            label: {
                "Beta":             f"{beta(returns, benchmark_returns):.2f}",
                "Alpha (ann.)":     f"{jensens_alpha(returns, benchmark_returns, risk_free):.2%}",
                "Info Ratio":       f"{information_ratio(returns, benchmark_returns):.2f}",
                "Up Capture":       f"{up_capture(returns, benchmark_returns):.1%}",
                "Down Capture":     f"{down_capture(returns, benchmark_returns):.1%}",
            },
            "SET Index": {
                "Beta":         "1.00",
                "Alpha (ann.)": "0.00%",
                "Info Ratio":   "—",
                "Up Capture":   "100.0%",
                "Down Capture": "100.0%",
            },
        }
        rel_df = pd.DataFrame(rel_data)
        rel_df.index.name = "Metric"
        df = pd.concat([df, rel_df])

    return df
