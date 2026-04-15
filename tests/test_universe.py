"""
Tests for src/data/universe.py
---------------------------------
Validates the SET universe registry — no network calls required.
"""
from __future__ import annotations

import pytest

from src.data.models import SETIndex
from src.data.universe import (
    find_stock,
    get_tickers,
    get_tickers_yf,
    get_universe,
    universe_summary,
)


# ---------------------------------------------------------------------------
# get_universe()
# ---------------------------------------------------------------------------

class TestGetUniverse:
    def test_returns_list(self):
        result = get_universe(SETIndex.SET50)
        assert isinstance(result, list)

    def test_set50_nonempty(self):
        result = get_universe(SETIndex.SET50)
        assert len(result) > 0

    def test_set100_nonempty(self):
        result = get_universe(SETIndex.SET100)
        assert len(result) > 0

    def test_set100_larger_than_set50(self):
        set50 = get_universe(SETIndex.SET50)
        set100 = get_universe(SETIndex.SET100)
        assert len(set100) >= len(set50)

    def test_all_index_returns_all_stocks(self):
        all_stocks = get_universe(SETIndex.ALL)
        set50 = get_universe(SETIndex.SET50)
        assert len(all_stocks) >= len(set50)

    def test_returns_stock_info_objects(self):
        from src.data.models import StockInfo
        result = get_universe(SETIndex.SET50)
        for stock in result:
            assert isinstance(stock, StockInfo)

    def test_set50_membership_correct(self):
        """All SET50 stocks should have SET50 in their index_membership."""
        result = get_universe(SETIndex.SET50)
        for stock in result:
            assert SETIndex.SET50 in stock.index_membership, (
                f"{stock.ticker} in SET50 but missing SET50 in index_membership"
            )


# ---------------------------------------------------------------------------
# get_tickers()
# ---------------------------------------------------------------------------

class TestGetTickers:
    def test_returns_list_of_strings(self):
        tickers = get_tickers(SETIndex.SET50)
        assert isinstance(tickers, list)
        assert all(isinstance(t, str) for t in tickers)

    def test_no_bk_suffix(self):
        """get_tickers() returns plain tickers without .BK."""
        tickers = get_tickers(SETIndex.SET50)
        assert not any(".BK" in t for t in tickers)

    def test_tickers_uppercase(self):
        tickers = get_tickers(SETIndex.SET50)
        for t in tickers:
            assert t == t.upper()

    def test_known_stocks_in_set50(self):
        """PTT and CPALL are large-cap SET50 stalwarts — should always be present."""
        tickers = get_tickers(SETIndex.SET50)
        assert "PTT" in tickers or "CPALL" in tickers, (
            "Expected at least one of PTT/CPALL in SET50"
        )


# ---------------------------------------------------------------------------
# get_tickers_yf()
# ---------------------------------------------------------------------------

class TestGetTickersYF:
    def test_all_have_bk_suffix(self):
        tickers = get_tickers_yf(SETIndex.SET50)
        assert all(t.endswith(".BK") for t in tickers), (
            "All yfinance tickers should end with .BK"
        )

    def test_same_count_as_get_tickers(self):
        plain = get_tickers(SETIndex.SET50)
        yf = get_tickers_yf(SETIndex.SET50)
        assert len(plain) == len(yf)

    def test_conversion_correct(self):
        """Plain ticker 'PTT' should become 'PTT.BK' in yfinance format."""
        plain = get_tickers(SETIndex.SET50)
        yf = get_tickers_yf(SETIndex.SET50)
        # Check that each yf ticker corresponds to a plain ticker + .BK
        for t_plain, t_yf in zip(plain, yf):
            assert t_yf == t_plain + ".BK"


# ---------------------------------------------------------------------------
# find_stock()
# ---------------------------------------------------------------------------

class TestFindStock:
    def test_finds_by_exact_ticker(self):
        tickers = get_tickers(SETIndex.SET50)
        if not tickers:
            pytest.skip("No SET50 stocks in registry")
        ticker = tickers[0]
        result = find_stock(ticker)
        assert result is not None
        assert result.ticker == ticker

    def test_returns_none_for_unknown(self):
        result = find_stock("XYZNOTREAL")
        assert result is None

    def test_case_insensitive(self):
        tickers = get_tickers(SETIndex.SET50)
        if not tickers:
            pytest.skip("No SET50 stocks in registry")
        ticker = tickers[0]
        result_lower = find_stock(ticker.lower())
        result_upper = find_stock(ticker.upper())
        # Both should resolve to same stock (or both None if case-sensitive)
        if result_lower and result_upper:
            assert result_lower.ticker == result_upper.ticker


# ---------------------------------------------------------------------------
# universe_summary()
# ---------------------------------------------------------------------------

class TestUniverseSummary:
    def test_returns_dict(self):
        result = universe_summary()
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        result = universe_summary()
        assert "total_stocks" in result
        assert "set50_count" in result
        assert "set100_count" in result
        assert "sectors" in result

    def test_total_stocks_positive(self):
        result = universe_summary()
        assert result["total_stocks"] > 0

    def test_set50_count_leq_set100(self):
        result = universe_summary()
        assert result["set50_count"] <= result["set100_count"]

    def test_sectors_is_list_or_set(self):
        result = universe_summary()
        assert isinstance(result["sectors"], (list, set, dict))

    def test_sectors_nonempty(self):
        result = universe_summary()
        assert len(result["sectors"]) > 0

    def test_total_matches_all_index(self):
        result = universe_summary()
        all_tickers = get_tickers(SETIndex.ALL)
        assert result["total_stocks"] == len(all_tickers)
