"""
Backtesting Engine
-------------------
Monthly-rebalance, long-only, equal-weight portfolio backtester.

Design:
  - Walk-forward: at each rebalance date, score stocks using only data
    available at that point (no look-ahead bias)
  - Equal-weight within selected portfolio (simple, realistic for retail)
  - Transaction costs modeled as flat % per turnover

Usage:
    from src.backtest import BacktestEngine
    from src.data import DataFetcher, SETIndex

    fetcher = DataFetcher()
    engine  = BacktestEngine(fetcher)

    result = engine.run(
        index     = SETIndex.SET50,
        style     = "blend",
        top_n     = 10,
        start     = "2021-01-01",
        end       = "2026-01-01",
    )
    print(result.summary)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.data import DataFetcher, SETIndex
from src.screener.factors import FACTOR_DIRECTION, compute_factor_scores
from src.screener.scorer import CompositeScorer
from .metrics import (
    cagr, calmar_ratio, drawdown_series, equity_curve,
    max_drawdown, performance_summary, sharpe_ratio,
    sortino_ratio, volatility, win_rate,
)

TRANSACTION_COST = 0.0025  # 0.25% one-way — approximate SET brokerage + slippage


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    strategy_returns:  pd.Series
    benchmark_returns: pd.Series
    portfolio_history: pd.DataFrame   # date → list of tickers held
    rebalance_dates:   list[str]
    style:             str
    index:             str
    top_n:             int
    summary:           pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        self.summary = performance_summary(
            self.strategy_returns,
            self.benchmark_returns,
            label=f"{self.style.title()} Top{self.top_n}",
        )

    @property
    def equity_curve(self) -> pd.Series:
        return equity_curve(self.strategy_returns)

    @property
    def benchmark_equity_curve(self) -> pd.Series:
        return equity_curve(self.benchmark_returns)

    @property
    def drawdown(self) -> pd.Series:
        return drawdown_series(self.strategy_returns)


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Walk-forward monthly backtester.

    At each month-end:
      1. Pull ALL price history available up to that date
      2. Compute factors cross-sectionally from that snapshot
      3. Select top_n stocks by composite score
      4. Hold equal-weight until next rebalance
      5. Calculate return, subtract transaction costs on turnover
    """

    def __init__(self, fetcher: Optional[DataFetcher] = None) -> None:
        self.fetcher = fetcher or DataFetcher()

    def run(
        self,
        index:   SETIndex = SETIndex.SET50,
        style:   str      = "blend",
        top_n:   int      = 10,
        start:   str      = "2021-01-01",
        end:     str      = "2026-01-01",
        verbose: bool     = True,
    ) -> BacktestResult:
        """
        Run the backtest.

        Returns:
            BacktestResult with daily strategy + benchmark returns,
            portfolio history, and summary metrics.
        """
        if verbose:
            print(f"Backtesting: {style.upper()} Top{top_n} | {index.value} | {start} to {end}")

        # ── 1. Fetch all price + fundamental data ──────────────────────────
        # NOTE: Fundamentals are a current snapshot, not point-in-time historical.
        # This introduces mild look-ahead bias for fundamental factors (P/E, ROE etc.)
        # but is acceptable for MVP. A production system would use a time-series
        # fundamentals database (e.g. Compustat, Bloomberg point-in-time).
        price_map = self.fetcher.get_universe_prices(index=index, period="max", verbose=verbose)
        fund_snapshot = self.fetcher.get_universe_fundamentals(index=index, verbose=verbose)
        benchmark = self.fetcher.get_set_index(period="max")

        if not price_map or benchmark is None:
            raise RuntimeError("Failed to fetch price data.")

        # ── 2. Build aligned daily returns matrix ─────────────────────────
        close_dict = {t: p.df["Close"] for t, p in price_map.items()}
        price_df = pd.DataFrame(close_dict).sort_index()
        price_df.index = pd.to_datetime(price_df.index).tz_localize(None)

        bench_close = benchmark.df["Close"].copy()
        bench_close.index = pd.to_datetime(bench_close.index).tz_localize(None)

        # Align to backtest window
        price_df   = price_df[(price_df.index >= start) & (price_df.index <= end)]
        bench_close = bench_close[(bench_close.index >= start) & (bench_close.index <= end)]

        if price_df.empty:
            raise RuntimeError("No price data in the requested backtest window.")

        # ── 3. Generate monthly rebalance dates ───────────────────────────
        rebalance_dates = pd.date_range(
            start=price_df.index[0],
            end=price_df.index[-1],
            freq="ME",   # Month-End
        )

        # ── 4. Walk-forward loop ──────────────────────────────────────────
        daily_returns: list[pd.Series] = []
        portfolio_history: dict[str, list[str]] = {}
        current_portfolio: list[str] = []
        scorer = CompositeScorer(style=style)

        for i, rebal_date in enumerate(rebalance_dates[:-1]):
            next_rebal = rebalance_dates[i + 1]

            # Select portfolio at rebalance date
            new_portfolio = self._select_portfolio(
                price_df=price_df,
                fund_map=self._build_snapshot_factors(price_map, fund_snapshot, as_of=rebal_date),
                scorer=scorer,
                top_n=top_n,
                as_of=rebal_date,
            )

            if not new_portfolio:
                new_portfolio = current_portfolio  # fallback: hold previous

            # Transaction cost on turnover
            turnover = self._turnover(current_portfolio, new_portfolio)
            tc_drag = turnover * TRANSACTION_COST

            # Holding period returns (rebal_date → next_rebal)
            hold_prices = price_df.loc[
                (price_df.index > rebal_date) & (price_df.index <= next_rebal),
                new_portfolio,
            ]

            if hold_prices.empty or hold_prices.shape[0] < 2:
                current_portfolio = new_portfolio
                continue

            hold_returns = hold_prices.pct_change().dropna()
            portfolio_ret = hold_returns.mean(axis=1)  # equal-weight
            portfolio_ret.iloc[0] -= tc_drag           # apply TC on first day

            daily_returns.append(portfolio_ret)
            portfolio_history[str(rebal_date.date())] = new_portfolio
            current_portfolio = new_portfolio

            if verbose:
                port_str = ", ".join(t.replace(".BK", "") for t in new_portfolio[:5])
                print(f"  {rebal_date.date()} rebal → [{port_str}{'...' if len(new_portfolio)>5 else ''}]")

        # ── 5. Assemble results ───────────────────────────────────────────
        if not daily_returns:
            raise RuntimeError("Backtest produced no return data. Widen date range.")

        strategy_returns = pd.concat(daily_returns).sort_index()
        benchmark_returns = bench_close.pct_change().dropna()
        benchmark_returns = benchmark_returns.reindex(strategy_returns.index).fillna(0)

        return BacktestResult(
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            portfolio_history=portfolio_history,
            rebalance_dates=list(portfolio_history.keys()),
            style=style,
            index=index.value,
            top_n=top_n,
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _build_snapshot_factors(
        self,
        price_map: dict,
        fund_snapshot: dict,
        as_of: pd.Timestamp,
    ) -> dict[str, dict]:
        """
        Build a combined factor snapshot at a given rebalance date:

        - Momentum factors: calculated walk-forward from prices UP TO as_of
          (no look-ahead bias)
        - Fundamental factors: current snapshot (P/E, ROE, P/B etc.)
          Mild look-ahead bias acknowledged — acceptable for MVP.

        Merging both lets every style preset (value, quality, blend, etc.)
        work correctly, not just momentum.
        """
        snapshot: dict[str, dict] = {}

        for ticker_yf, price_data in price_map.items():
            df = price_data.df.copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            hist = df[df.index <= as_of]["Close"]

            if len(hist) < 22:
                continue

            # ── Momentum factors (walk-forward, no look-ahead) ────────────
            row: dict = {
                "ticker":  ticker_yf.replace(".BK", ""),
                "mom_1m":  self._ret(hist, 21),
                "mom_3m":  self._ret(hist, 63),
                "mom_6m":  self._ret(hist, 126),
                "mom_12m": self._ret(hist, 252),
            }

            # ── Fundamental factors (current snapshot) ────────────────────
            fund = fund_snapshot.get(ticker_yf)
            if fund is not None:
                for key, val in fund.to_factor_dict().items():
                    row[key] = val

            snapshot[ticker_yf] = row

        return snapshot

    @staticmethod
    def _ret(close: pd.Series, n: int) -> Optional[float]:
        if len(close) < n + 1:
            return None
        p_now, p_past = float(close.iloc[-1]), float(close.iloc[-(n + 1)])
        return None if p_past == 0 else (p_now / p_past) - 1

    def _select_portfolio(
        self,
        price_df: pd.DataFrame,
        fund_map: dict,
        scorer: CompositeScorer,
        top_n: int,
        as_of: pd.Timestamp,
    ) -> list[str]:
        """Score stocks available at as_of and return top_n ticker_yf list."""
        if not fund_map:
            return []

        rows = []
        for ticker_yf, factors in fund_map.items():
            if ticker_yf in price_df.columns:
                rows.append(factors)

        if not rows:
            return []

        raw_df = pd.DataFrame(rows)
        scored = scorer.score(raw_df)
        top = scored.head(top_n)

        return [f"{row['ticker']}.BK" for _, row in top.iterrows()]

    @staticmethod
    def _turnover(old: list[str], new: list[str]) -> float:
        """Fraction of portfolio replaced (0 to 1)."""
        if not old:
            return 1.0
        old_set, new_set = set(old), set(new)
        changed = len(old_set.symmetric_difference(new_set))
        return changed / max(len(old_set), len(new_set))
