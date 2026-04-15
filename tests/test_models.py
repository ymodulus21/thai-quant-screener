"""
Tests for src/data/models.py
------------------------------
Covers Pydantic models: StockInfo, FundamentalData, PriceData.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.data.models import (
    AssetClass,
    FetchResult,
    FundamentalData,
    PriceData,
    SETIndex,
    StockInfo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_df(n: int = 100, start: str = "2022-01-01") -> pd.DataFrame:
    """Synthetic OHLCV DataFrame."""
    idx = pd.date_range(start, periods=n, freq="B")
    close = 100 * (1 + np.random.normal(0.001, 0.01, n)).cumprod()
    return pd.DataFrame({
        "Open":   close * 0.99,
        "High":   close * 1.01,
        "Low":    close * 0.98,
        "Close":  close,
        "Volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=idx)


# ---------------------------------------------------------------------------
# SETIndex Enum
# ---------------------------------------------------------------------------

class TestSETIndex:
    def test_values_are_strings(self):
        assert SETIndex.SET50.value == "SET50"
        assert SETIndex.SET100.value == "SET100"

    def test_all_members_present(self):
        names = {e.name for e in SETIndex}
        assert {"SET50", "SET100", "SETHD", "SETWB", "ALL"}.issubset(names)


# ---------------------------------------------------------------------------
# StockInfo
# ---------------------------------------------------------------------------

class TestStockInfo:
    def test_basic_creation(self):
        s = StockInfo(ticker="ptt", ticker_yf="ptt.bk", name="PTT PCL")
        assert s.ticker == "PTT"          # uppercased
        assert s.ticker_yf == "PTT.BK"   # uppercased

    def test_default_asset_class_is_equity(self):
        s = StockInfo(ticker="PTT", ticker_yf="PTT.BK", name="PTT PCL")
        assert s.asset_class == AssetClass.EQUITY

    def test_sector_optional(self):
        s = StockInfo(ticker="PTT", ticker_yf="PTT.BK", name="PTT PCL")
        assert s.sector is None

    def test_index_membership_default_empty(self):
        s = StockInfo(ticker="PTT", ticker_yf="PTT.BK", name="PTT PCL")
        assert s.index_membership == []

    def test_index_membership_assigned(self):
        s = StockInfo(
            ticker="PTT", ticker_yf="PTT.BK", name="PTT PCL",
            index_membership=[SETIndex.SET50, SETIndex.SET100]
        )
        assert SETIndex.SET50 in s.index_membership

    def test_ticker_whitespace_stripped(self):
        s = StockInfo(ticker="  PTT  ", ticker_yf="PTT.BK", name="PTT PCL")
        assert s.ticker == "PTT"


# ---------------------------------------------------------------------------
# FundamentalData
# ---------------------------------------------------------------------------

class TestFundamentalData:
    def test_basic_creation(self):
        fd = FundamentalData(ticker="PTT")
        assert fd.ticker == "PTT"

    def test_all_fields_optional(self):
        fd = FundamentalData(ticker="PTT")
        assert fd.pe_ratio is None
        assert fd.roe is None
        assert fd.dividend_yield is None

    def test_is_value_true(self):
        fd = FundamentalData(ticker="X", pe_ratio=10.0, pb_ratio=1.0)
        assert fd.is_value is True

    def test_is_value_false_high_pe(self):
        fd = FundamentalData(ticker="X", pe_ratio=25.0, pb_ratio=1.0)
        assert fd.is_value is False

    def test_is_value_false_missing_data(self):
        fd = FundamentalData(ticker="X")
        assert fd.is_value is False

    def test_is_quality_true(self):
        fd = FundamentalData(ticker="X", roe=0.20, debt_to_equity=0.5)
        assert fd.is_quality is True

    def test_is_quality_false_low_roe(self):
        fd = FundamentalData(ticker="X", roe=0.10, debt_to_equity=0.5)
        assert fd.is_quality is False

    def test_is_quality_false_high_debt(self):
        fd = FundamentalData(ticker="X", roe=0.20, debt_to_equity=1.5)
        assert fd.is_quality is False

    def test_to_factor_dict_keys(self):
        fd = FundamentalData(ticker="PTT", pe_ratio=12.0, roe=0.18)
        d = fd.to_factor_dict()
        assert "pe_ratio" in d
        assert "roe" in d
        assert "dividend_yield" in d
        # market_cap is in the dict (used for size, not factor)
        assert "market_cap" in d

    def test_to_factor_dict_values(self):
        fd = FundamentalData(ticker="PTT", pe_ratio=12.0, roe=0.18)
        d = fd.to_factor_dict()
        assert d["pe_ratio"] == 12.0
        assert d["roe"] == 0.18

    def test_to_factor_dict_none_for_missing(self):
        fd = FundamentalData(ticker="PTT")
        d = fd.to_factor_dict()
        assert d["pe_ratio"] is None


# ---------------------------------------------------------------------------
# PriceData
# ---------------------------------------------------------------------------

class TestPriceData:
    def test_basic_creation(self):
        df = _make_price_df()
        pd_obj = PriceData(ticker="PTT", df=df)
        assert pd_obj.ticker == "PTT"

    def test_missing_column_raises(self):
        df = _make_price_df().drop(columns=["Volume"])
        with pytest.raises(ValueError, match="missing columns"):
            PriceData(ticker="PTT", df=df)

    def test_latest_close_is_last_row(self):
        df = _make_price_df(n=50)
        pd_obj = PriceData(ticker="PTT", df=df)
        assert pd_obj.latest_close == pytest.approx(float(df["Close"].iloc[-1]))

    def test_start_date_is_first_index(self):
        df = _make_price_df(n=50, start="2023-01-03")
        pd_obj = PriceData(ticker="PTT", df=df)
        assert pd_obj.start_date == df.index[0].date()

    def test_end_date_is_last_index(self):
        df = _make_price_df(n=50)
        pd_obj = PriceData(ticker="PTT", df=df)
        assert pd_obj.end_date == df.index[-1].date()

    def test_returns_length_is_n_minus_one(self):
        df = _make_price_df(n=50)
        pd_obj = PriceData(ticker="PTT", df=df)
        returns = pd_obj.returns()
        # pct_change drops first NaN
        assert len(returns) == 49

    def test_returns_no_nan(self):
        df = _make_price_df(n=100)
        pd_obj = PriceData(ticker="PTT", df=df)
        returns = pd_obj.returns()
        assert not returns.isna().any()

    def test_returns_is_series(self):
        df = _make_price_df()
        pd_obj = PriceData(ticker="PTT", df=df)
        assert isinstance(pd_obj.returns(), pd.Series)


# ---------------------------------------------------------------------------
# FetchResult
# ---------------------------------------------------------------------------

class TestFetchResult:
    def test_success_result(self):
        fr = FetchResult(ticker="PTT", success=True)
        assert fr.success is True
        assert fr.error is None

    def test_failure_result(self):
        fr = FetchResult(ticker="PTT", success=False, error="Timeout")
        assert fr.success is False
        assert fr.error == "Timeout"
