"""
Core Data Fetcher
-----------------
Single entry point for ALL market data needs.
Wraps yfinance with:
  - Disk caching (via cache.py)
  - Retry logic (Yahoo Finance can be flaky)
  - Structured return types (models.py)
  - Bulk fetch with progress reporting

Usage:
    from src.data import DataFetcher, SETIndex

    fetcher = DataFetcher()

    # Single stock price history
    price = fetcher.get_price("PTT.BK", period="1y")

    # Single stock fundamentals
    fund = fetcher.get_fundamental("KBANK.BK")

    # Bulk fundamentals for SET50
    results = fetcher.get_universe_fundamentals(SETIndex.SET50)

    # Factor DataFrame — ready for screener
    df = fetcher.build_factor_dataframe(SETIndex.SET50)
"""
from __future__ import annotations

import logging
import time
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from . import cache
from .models import FetchResult, FundamentalData, PriceData, SETIndex
from .universe import get_tickers_yf, get_universe

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Field mapping: yfinance info key → FundamentalData field
# ---------------------------------------------------------------------------
_YF_FIELD_MAP = {
    "trailingPE":       "pe_ratio",
    "priceToBook":      "pb_ratio",
    "priceToSalesTrailingTwelveMonths": "ps_ratio",
    "enterpriseToEbitda": "ev_ebitda",
    "returnOnEquity":   "roe",
    "returnOnAssets":   "roa",
    "grossMargins":     "gross_margin",
    "profitMargins":    "net_margin",
    "debtToEquity":     "debt_to_equity",
    "currentRatio":     "current_ratio",
    "marketCap":        "market_cap",
    "dividendYield":    "dividend_yield",
    "payoutRatio":      "payout_ratio",
    "trailingEps":      "eps_ttm",
    "revenueGrowth":    "revenue_growth",
    "earningsGrowth":   "earnings_growth",
}


# ---------------------------------------------------------------------------
# DataFetcher
# ---------------------------------------------------------------------------

class DataFetcher:
    """
    Stateless data fetcher. Instantiate once and reuse.
    All methods are safe to call repeatedly — cache prevents redundant fetches.
    """

    def __init__(
        self,
        use_cache: bool = True,
        retry_attempts: int = 3,
        retry_delay_sec: float = 2.0,
    ) -> None:
        self.use_cache = use_cache
        self.retry_attempts = retry_attempts
        self.retry_delay_sec = retry_delay_sec

    # ------------------------------------------------------------------
    # Price Data
    # ------------------------------------------------------------------

    def get_price(
        self,
        ticker_yf: str,
        period: str = "5y",
        interval: str = "1d",
    ) -> Optional[PriceData]:
        """
        Fetch OHLCV history for one ticker.

        Args:
            ticker_yf: Yahoo Finance ticker e.g. 'PTT.BK'
            period:    yfinance period string — '1mo','3mo','1y','2y','5y','max'
            interval:  '1d' (daily) | '1wk' (weekly) | '1mo' (monthly)

        Returns:
            PriceData or None on failure.
        """
        # Cache hit (only for default daily period — don't cache intraday)
        if self.use_cache and interval == "1d" and period == "5y":
            cached_df = cache.get_price(ticker_yf)
            if cached_df is not None:
                return PriceData(ticker=ticker_yf, df=cached_df)

        df = self._fetch_price_with_retry(ticker_yf, period, interval)
        if df is None or df.empty:
            logger.warning("No price data returned for %s", ticker_yf)
            return None

        # Normalise columns
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="last")].sort_index()

        if self.use_cache and interval == "1d" and period == "5y":
            cache.set_price(ticker_yf, df)

        return PriceData(ticker=ticker_yf, df=df)

    def _fetch_price_with_retry(
        self, ticker_yf: str, period: str, interval: str
    ) -> Optional[pd.DataFrame]:
        for attempt in range(1, self.retry_attempts + 1):
            try:
                tk = yf.Ticker(ticker_yf)
                df = tk.history(period=period, interval=interval, auto_adjust=True)
                if not df.empty:
                    return df
                logger.debug("Empty price response for %s (attempt %d)", ticker_yf, attempt)
            except Exception as exc:
                logger.warning("Price fetch error %s attempt %d: %s", ticker_yf, attempt, exc)
            if attempt < self.retry_attempts:
                time.sleep(self.retry_delay_sec)
        return None

    # ------------------------------------------------------------------
    # Fundamental Data
    # ------------------------------------------------------------------

    def get_fundamental(self, ticker_yf: str) -> Optional[FundamentalData]:
        """
        Fetch fundamental metrics for one ticker.
        Returns FundamentalData or None on failure.
        """
        if self.use_cache:
            cached = cache.get_fundamental(ticker_yf)
            if cached is not None:
                return FundamentalData(**cached)

        raw_info = self._fetch_info_with_retry(ticker_yf)
        if raw_info is None:
            return None

        fields: dict = {"ticker": ticker_yf, "fetched_at": datetime.utcnow()}
        for yf_key, model_key in _YF_FIELD_MAP.items():
            val = raw_info.get(yf_key)
            # Coerce NaN/Inf to None so Pydantic stays happy
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                val = None
            fields[model_key] = val

        fund = FundamentalData(**fields)

        if self.use_cache:
            cache.set_fundamental(ticker_yf, fund.model_dump(mode="json"))

        return fund

    def _fetch_info_with_retry(self, ticker_yf: str) -> Optional[dict]:
        for attempt in range(1, self.retry_attempts + 1):
            try:
                info = yf.Ticker(ticker_yf).info
                if info and len(info) > 5:   # non-trivial response
                    return info
            except Exception as exc:
                logger.warning("Info fetch error %s attempt %d: %s", ticker_yf, attempt, exc)
            if attempt < self.retry_attempts:
                time.sleep(self.retry_delay_sec)
        return None

    # ------------------------------------------------------------------
    # Bulk / Universe Fetch
    # ------------------------------------------------------------------

    def get_universe_prices(
        self,
        index: SETIndex = SETIndex.SET50,
        period: str = "5y",
        verbose: bool = True,
    ) -> dict[str, PriceData]:
        """
        Fetch price history for all stocks in an index.
        Returns dict keyed by ticker_yf.
        """
        tickers = get_tickers_yf(index)
        results: dict[str, PriceData] = {}

        if verbose:
            print(f"Fetching prices for {len(tickers)} stocks ({index.value})...")

        for i, ticker_yf in enumerate(tickers, 1):
            price = self.get_price(ticker_yf, period=period)
            if price is not None:
                results[ticker_yf] = price
            if verbose:
                status = "OK" if price else "SKIP"
                print(f"  [{i:>2}/{len(tickers)}] {ticker_yf:<12} {status}")

        if verbose:
            print(f"Done. {len(results)}/{len(tickers)} fetched successfully.")

        return results

    def get_universe_fundamentals(
        self,
        index: SETIndex = SETIndex.SET50,
        verbose: bool = True,
    ) -> dict[str, FundamentalData]:
        """
        Fetch fundamentals for all stocks in an index.
        Returns dict keyed by ticker_yf.
        """
        tickers = get_tickers_yf(index)
        results: dict[str, FundamentalData] = {}

        if verbose:
            print(f"Fetching fundamentals for {len(tickers)} stocks ({index.value})...")

        for i, ticker_yf in enumerate(tickers, 1):
            fund = self.get_fundamental(ticker_yf)
            if fund is not None:
                results[ticker_yf] = fund
            if verbose:
                status = "OK" if fund else "SKIP"
                print(f"  [{i:>2}/{len(tickers)}] {ticker_yf:<12} {status}")

        if verbose:
            print(f"Done. {len(results)}/{len(tickers)} fetched successfully.")

        return results

    # ------------------------------------------------------------------
    # Factor DataFrame — ready for Screener / Quant Model
    # ------------------------------------------------------------------

    def build_factor_dataframe(
        self,
        index: SETIndex = SETIndex.SET50,
        include_momentum: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Build a wide DataFrame: one row per stock, columns = all factors.
        Adds momentum columns if include_momentum=True.

        Columns:
            ticker, name, sector | pe_ratio, pb_ratio, roe, ... |
            mom_1m, mom_3m, mom_6m, mom_12m (% returns)

        Returns:
            pd.DataFrame sorted by market_cap descending.
        """
        stocks = get_universe(index)
        fund_map = self.get_universe_fundamentals(index, verbose=verbose)

        rows = []
        for stock in stocks:
            fund = fund_map.get(stock.ticker_yf)
            if fund is None:
                continue

            row = {
                "ticker":  stock.ticker,
                "name":    stock.name,
                "sector":  stock.sector or "Unknown",
                **fund.to_factor_dict(),
            }

            # Momentum factors
            if include_momentum:
                price = self.get_price(stock.ticker_yf, period="1y")
                if price is not None and len(price.df) >= 20:
                    close = price.df["Close"]
                    row["mom_1m"]  = self._momentum(close, 21)
                    row["mom_3m"]  = self._momentum(close, 63)
                    row["mom_6m"]  = self._momentum(close, 126)
                    row["mom_12m"] = self._momentum(close, 252)
                else:
                    row.update({"mom_1m": None, "mom_3m": None, "mom_6m": None, "mom_12m": None})

            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty and "market_cap" in df.columns:
            df = df.sort_values("market_cap", ascending=False).reset_index(drop=True)

        return df

    @staticmethod
    def _momentum(close: pd.Series, lookback: int) -> Optional[float]:
        """Simple price momentum: (P_now / P_n_days_ago) - 1."""
        if len(close) < lookback + 1:
            return None
        p_now  = close.iloc[-1]
        p_past = close.iloc[-(lookback + 1)]
        if p_past == 0:
            return None
        return round((p_now / p_past) - 1, 4)

    # ------------------------------------------------------------------
    # SET Index Benchmark
    # ------------------------------------------------------------------

    def get_set_index(self, period: str = "5y") -> Optional[PriceData]:
        """Fetch SET Composite Index (^SET.BK) as benchmark."""
        return self.get_price("^SET.BK", period=period)
