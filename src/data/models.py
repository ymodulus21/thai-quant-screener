"""
Data models for Thai SET market data layer.
Shared by both Model 4 (Report Summarizer) and Model 1 (Quant Screener).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SETIndex(str, Enum):
    SET50  = "SET50"
    SET100 = "SET100"
    SETHD  = "SETHD"   # High Dividend
    SETWB  = "SETWB"   # Well-Being
    ALL    = "ALL"      # All available tickers in universe


class AssetClass(str, Enum):
    EQUITY     = "EQUITY"
    ETF        = "ETF"
    REIT       = "REIT"
    INFRA      = "INFRA"


# ---------------------------------------------------------------------------
# Company / Ticker Info
# ---------------------------------------------------------------------------

class StockInfo(BaseModel):
    """Static metadata for a single SET-listed stock."""
    ticker: str                          # e.g. "PTT"
    ticker_yf: str                       # e.g. "PTT.BK"
    name: str
    sector: Optional[str]   = None
    industry: Optional[str] = None
    asset_class: AssetClass = AssetClass.EQUITY
    index_membership: list[SETIndex] = Field(default_factory=list)
    market_cap: Optional[float] = None   # THB, from latest fetch
    currency: str = "THB"

    @field_validator("ticker", "ticker_yf", mode="before")
    @classmethod
    def uppercase(cls, v: str) -> str:
        return v.upper().strip()


# ---------------------------------------------------------------------------
# Price Data
# ---------------------------------------------------------------------------

@dataclass
class PriceData:
    """OHLCV time series for a single ticker."""
    ticker: str
    df: pd.DataFrame      # columns: Open, High, Low, Close, Volume — index: DatetimeIndex
    fetched_at: datetime  = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"PriceData for {self.ticker} missing columns: {missing}")

    @property
    def latest_close(self) -> float:
        return float(self.df["Close"].iloc[-1])

    @property
    def start_date(self) -> date:
        return self.df.index[0].date()

    @property
    def end_date(self) -> date:
        return self.df.index[-1].date()

    def returns(self, freq: str = "D") -> pd.Series:
        """Daily ('D') or monthly ('ME') log returns."""
        close = self.df["Close"].resample(freq).last().dropna()
        return close.pct_change().dropna()


# ---------------------------------------------------------------------------
# Fundamental Data
# ---------------------------------------------------------------------------

class FundamentalData(BaseModel):
    """Point-in-time snapshot of key fundamental metrics."""
    ticker: str
    fetched_at: datetime = Field(default_factory=datetime.utcnow)

    # Valuation
    pe_ratio:      Optional[float] = None   # Trailing P/E
    pb_ratio:      Optional[float] = None   # Price-to-Book
    ps_ratio:      Optional[float] = None   # Price-to-Sales
    ev_ebitda:     Optional[float] = None

    # Quality
    roe:           Optional[float] = None   # Return on Equity (decimal)
    roa:           Optional[float] = None   # Return on Assets (decimal)
    gross_margin:  Optional[float] = None
    net_margin:    Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None

    # Size & Income
    market_cap:    Optional[float] = None   # THB
    dividend_yield: Optional[float] = None  # decimal, e.g. 0.05 = 5%
    payout_ratio:  Optional[float] = None
    eps_ttm:       Optional[float] = None

    # Growth (YoY %)
    revenue_growth:  Optional[float] = None
    earnings_growth: Optional[float] = None

    @property
    def is_value(self) -> bool:
        """Simple value screen: low P/E and low P/B."""
        return (
            self.pe_ratio is not None and self.pe_ratio < 15 and
            self.pb_ratio is not None and self.pb_ratio < 1.5
        )

    @property
    def is_quality(self) -> bool:
        """Simple quality screen: high ROE, low debt."""
        return (
            self.roe is not None and self.roe > 0.15 and
            self.debt_to_equity is not None and self.debt_to_equity < 1.0
        )

    def to_factor_dict(self) -> dict[str, Optional[float]]:
        """Flat dict of numeric factors for factor model use."""
        return {
            "pe_ratio":       self.pe_ratio,
            "pb_ratio":       self.pb_ratio,
            "ps_ratio":       self.ps_ratio,
            "ev_ebitda":      self.ev_ebitda,
            "roe":            self.roe,
            "roa":            self.roa,
            "gross_margin":   self.gross_margin,
            "net_margin":     self.net_margin,
            "debt_to_equity": self.debt_to_equity,
            "current_ratio":  self.current_ratio,
            "dividend_yield": self.dividend_yield,
            "market_cap":     self.market_cap,
            "revenue_growth": self.revenue_growth,
            "earnings_growth": self.earnings_growth,
        }


# ---------------------------------------------------------------------------
# Fetch Result — wraps success or failure cleanly
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    ticker: str
    success: bool
    error: Optional[str] = None
    data: Optional[PriceData | FundamentalData] = None
