"""
Thai SET Market Data Layer
--------------------------
Clean public API — import only from here, not from submodules directly.

Quick start:
    from src.data import DataFetcher, SETIndex, universe_summary

    fetcher = DataFetcher()
    df = fetcher.build_factor_dataframe(SETIndex.SET50)
    print(df[["ticker", "sector", "pe_ratio", "roe", "mom_3m"]].head(10))
"""
from .cache import cache_stats, clear_cache
from .fetcher import DataFetcher
from .models import (
    AssetClass,
    FetchResult,
    FundamentalData,
    PriceData,
    SETIndex,
    StockInfo,
)
from .universe import (
    find_stock,
    get_tickers,
    get_tickers_yf,
    get_universe,
    universe_summary,
)

__all__ = [
    # Fetcher
    "DataFetcher",
    # Models
    "PriceData",
    "FundamentalData",
    "StockInfo",
    "FetchResult",
    "SETIndex",
    "AssetClass",
    # Universe
    "get_universe",
    "get_tickers",
    "get_tickers_yf",
    "find_stock",
    "universe_summary",
    # Cache
    "cache_stats",
    "clear_cache",
]
