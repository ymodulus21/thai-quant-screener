"""
Performance Metrics
--------------------
All standard return-based metrics used in:
  - CFA L1/L2 Portfolio Management
  - Quant finance strategy evaluation

All functions accept a pd.Series of PERIODIC returns (not prices).
e.g. daily returns: [0.01, -0.005, 0.02, ...]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Thai market: ~245 trading days/year
TRADING_DAYS_PER_YEAR = 245
RISK_FREE_RATE = 0.02   # Thai risk-free proxy: ~2% (approximates T-bill / fixed deposit)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Compound Annual Growth Rate.
    CFA formula: (ending_value / beginning_value)^(1/years) - 1
    """
    n_periods = len(returns)
    if n_periods == 0:
        return 0.0
    total_return = (1 + returns).prod()
    years = n_periods / periods_per_year
    if years <= 0 or total_return <= 0:
        return 0.0
    return float(total_return ** (1 / years) - 1)


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
    Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Std Dev
    CFA: measures excess return per unit of total risk.
    """
    ann_return = cagr(returns, periods_per_year)
    ann_vol = volatility(returns, periods_per_year)
    if ann_vol == 0:
        return 0.0
    return float((ann_return - risk_free) / ann_vol)


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Sortino Ratio = (Portfolio Return - Risk-Free Rate) / Downside Deviation
    CFA: like Sharpe but only penalizes DOWNSIDE volatility.
    Better metric for asymmetric return distributions.
    """
    ann_return = cagr(returns, periods_per_year)
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = float(downside.std() * np.sqrt(periods_per_year))
    if downside_std == 0:
        return 0.0
    return float((ann_return - risk_free) / downside_std)


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum Drawdown = largest peak-to-trough decline.
    CFA: key risk metric for momentum / trend strategies.
    Returns negative float e.g. -0.35 = 35% max drawdown.
    """
    if len(returns) == 0:
        return 0.0
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Calmar Ratio = CAGR / |Max Drawdown|
    Measures return per unit of drawdown risk.
    Higher = better risk-adjusted return for trend-following strategies.
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


def equity_curve(returns: pd.Series, initial: float = 100.0) -> pd.Series:
    """Cumulative portfolio value starting at `initial`."""
    return initial * (1 + returns).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Full drawdown time series (negative values)."""
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    return (cumulative - rolling_max) / rolling_max


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def performance_summary(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    label: str = "Strategy",
) -> pd.DataFrame:
    """
    Produce a clean performance tearsheet as a DataFrame.

    Args:
        returns:            Strategy periodic returns
        benchmark_returns:  Optional benchmark (e.g. SET index returns)
        label:              Name of the strategy

    Returns:
        pd.DataFrame with metric | strategy | [benchmark] columns
    """
    def _metrics(r: pd.Series, name: str) -> dict:
        return {
            "name":        name,
            "CAGR":        f"{cagr(r):.2%}",
            "Volatility":  f"{volatility(r):.2%}",
            "Sharpe":      f"{sharpe_ratio(r):.2f}",
            "Sortino":     f"{sortino_ratio(r):.2f}",
            "Max Drawdown":f"{max_drawdown(r):.2%}",
            "Calmar":      f"{calmar_ratio(r):.2f}",
            "Win Rate":    f"{win_rate(r):.2%}",
            "Total Return":f"{(1 + returns).prod() - 1:.2%}",
        }

    rows = [_metrics(returns, label)]
    if benchmark_returns is not None:
        rows.append(_metrics(benchmark_returns, "SET Index"))

    df = pd.DataFrame(rows).set_index("name").T
    df.index.name = "Metric"
    return df
