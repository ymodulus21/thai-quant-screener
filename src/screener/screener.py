"""
Screener — Main Entry Point
-----------------------------
Combines DataFetcher + CompositeScorer into one clean interface.

Usage:
    from src.screener import Screener, SETIndex

    sc = Screener()
    result = sc.run(index=SETIndex.SET50, style="blend", top_n=10)
    print(result[["rank","ticker","sector","composite_score","pe_ratio","roe","mom_3m"]])
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from src.data import DataFetcher, SETIndex
from .scorer import STYLE_PRESETS, CompositeScorer

# Columns shown in default output
DEFAULT_DISPLAY_COLS = [
    "rank", "ticker", "name", "sector",
    "composite_score",
    "pe_ratio", "pb_ratio", "roe", "debt_to_equity",
    "dividend_yield", "mom_1m", "mom_3m", "mom_6m",
    "market_cap",
]


class Screener:
    """
    High-level screener: one call from index selection to ranked output.

    Args:
        fetcher:   DataFetcher instance (created if not provided)
        use_cache: Whether to use disk cache for fetched data
    """

    def __init__(
        self,
        fetcher: Optional[DataFetcher] = None,
        use_cache: bool = True,
    ) -> None:
        self.fetcher = fetcher or DataFetcher(use_cache=use_cache)

    def run(
        self,
        index:   SETIndex = SETIndex.SET50,
        style:   str      = "blend",
        top_n:   int      = 10,
        sector:  Optional[str] = None,
        min_market_cap: Optional[float] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Full screener pipeline: fetch → score → filter → rank.

        Args:
            index:          Universe to screen (SET50, SET100, etc.)
            style:          Scoring style preset (value/quality/momentum/income/growth/blend)
            top_n:          Return top N stocks (0 = return all)
            sector:         Optional sector filter e.g. "Financials"
            min_market_cap: Minimum market cap in THB (e.g. 10e9 = 10B)
            verbose:        Print progress

        Returns:
            pd.DataFrame ranked by composite_score descending.
        """
        if style not in STYLE_PRESETS:
            raise ValueError(f"Style must be one of: {list(STYLE_PRESETS)}")

        # 1. Fetch factor data
        raw_df = self.fetcher.build_factor_dataframe(index=index, verbose=verbose)

        if raw_df.empty:
            return pd.DataFrame()

        # 2. Optional filters (pre-scoring)
        if sector:
            raw_df = raw_df[raw_df["sector"].str.lower() == sector.lower()]
        if min_market_cap:
            raw_df = raw_df[raw_df["market_cap"] >= min_market_cap]

        if raw_df.empty:
            return pd.DataFrame()

        # 3. Score & rank
        scorer = CompositeScorer(style=style)
        scored = scorer.score(raw_df)

        # 4. Slice top N
        result = scored.head(top_n) if top_n > 0 else scored

        # 5. Format output
        display_cols = [c for c in DEFAULT_DISPLAY_COLS if c in result.columns]
        result = result[display_cols + [c for c in result.columns if c not in display_cols]]

        # Human-readable market cap (THB billions)
        if "market_cap" in result.columns:
            result = result.copy()
            result["mkt_cap_bn"] = (result["market_cap"] / 1e9).round(1)

        return result.reset_index(drop=True)

    def compare_styles(
        self,
        index: SETIndex = SETIndex.SET50,
        top_n: int = 5,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Run all style presets and show which stocks appear most often in top N.
        Useful for finding 'consensus' stocks.

        Returns:
            DataFrame: ticker, name, count (how many styles rank it top N), styles_list
        """
        raw_df = self.fetcher.build_factor_dataframe(index=index, verbose=verbose)
        counts: dict[str, dict] = {}

        for style in STYLE_PRESETS:
            scorer = CompositeScorer(style=style)
            top = scorer.top_n(raw_df, n=top_n)
            for _, row in top.iterrows():
                t = row["ticker"]
                if t not in counts:
                    counts[t] = {"ticker": t, "name": row.get("name", ""), "count": 0, "styles": []}
                counts[t]["count"] += 1
                counts[t]["styles"].append(style)

        result = pd.DataFrame(counts.values())
        if not result.empty:
            result = result.sort_values("count", ascending=False).reset_index(drop=True)
            result["styles"] = result["styles"].apply(lambda x: ", ".join(x))

        return result
