"""
Disk Cache Layer
----------------
Two-tier cache:
  - Price data  → Parquet files (efficient columnar, pandas-native)
  - Fundamental → JSON files (human-readable, quick to inspect)

TTL defaults:
  - Price (daily OHLCV) : 1 day   — stale after market close
  - Fundamentals        : 7 days  — quarterly data, rarely changes

Cache location: project root / .cache/data/
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# On Streamlit Cloud, use /tmp for writable cache; locally use project .cache/
import os as _os
_CACHE_ROOT = (
    Path("/tmp/thai_quant_cache/data")
    if _os.getenv("STREAMLIT_SHARING_MODE") or _os.getenv("HOME") == "/home/appuser"
    else Path(__file__).resolve().parents[2] / ".cache" / "data"
)
_PRICE_TTL_HOURS = 24
_FUNDAMENTAL_TTL_HOURS = 24 * 7   # 7 days


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _price_path(ticker_yf: str) -> Path:
    safe = ticker_yf.replace(".", "_").upper()
    return _CACHE_ROOT / "price" / f"{safe}.parquet"


def _fundamental_path(ticker_yf: str) -> Path:
    safe = ticker_yf.replace(".", "_").upper()
    return _CACHE_ROOT / "fundamental" / f"{safe}.json"


def _is_fresh(path: Path, ttl_hours: int) -> bool:
    if not path.exists():
        return False
    age = datetime.utcnow() - datetime.utcfromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=ttl_hours)


# ---------------------------------------------------------------------------
# Price Cache
# ---------------------------------------------------------------------------

def get_price(ticker_yf: str) -> Optional[pd.DataFrame]:
    path = _price_path(ticker_yf)
    if not _is_fresh(path, _PRICE_TTL_HOURS):
        return None
    try:
        df = pd.read_parquet(path)
        logger.debug("Cache HIT  price %s", ticker_yf)
        return df
    except Exception as exc:
        logger.warning("Cache READ error price %s: %s", ticker_yf, exc)
        return None


def set_price(ticker_yf: str, df: pd.DataFrame) -> None:
    path = _price_path(ticker_yf)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, compression="snappy")
        logger.debug("Cache WRITE price %s (%d rows)", ticker_yf, len(df))
    except Exception as exc:
        logger.warning("Cache WRITE error price %s: %s", ticker_yf, exc)


# ---------------------------------------------------------------------------
# Fundamental Cache
# ---------------------------------------------------------------------------

def get_fundamental(ticker_yf: str) -> Optional[dict[str, Any]]:
    path = _fundamental_path(ticker_yf)
    if not _is_fresh(path, _FUNDAMENTAL_TTL_HOURS):
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.debug("Cache HIT  fundamental %s", ticker_yf)
        return data
    except Exception as exc:
        logger.warning("Cache READ error fundamental %s: %s", ticker_yf, exc)
        return None


def set_fundamental(ticker_yf: str, data: dict[str, Any]) -> None:
    path = _fundamental_path(ticker_yf)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
        logger.debug("Cache WRITE fundamental %s", ticker_yf)
    except Exception as exc:
        logger.warning("Cache WRITE error fundamental %s: %s", ticker_yf, exc)


# ---------------------------------------------------------------------------
# Cache Management
# ---------------------------------------------------------------------------

def cache_stats() -> dict:
    """Return summary of cache size and freshness."""
    price_dir = _CACHE_ROOT / "price"
    fund_dir  = _CACHE_ROOT / "fundamental"

    price_files = list(price_dir.glob("*.parquet")) if price_dir.exists() else []
    fund_files  = list(fund_dir.glob("*.json"))     if fund_dir.exists() else []

    fresh_price = sum(1 for f in price_files if _is_fresh(f, _PRICE_TTL_HOURS))
    fresh_fund  = sum(1 for f in fund_files  if _is_fresh(f, _FUNDAMENTAL_TTL_HOURS))

    return {
        "cache_root": str(_CACHE_ROOT),
        "price_files":      len(price_files),
        "price_fresh":      fresh_price,
        "price_stale":      len(price_files) - fresh_price,
        "fundamental_files": len(fund_files),
        "fundamental_fresh": fresh_fund,
        "fundamental_stale": len(fund_files) - fresh_fund,
    }


def clear_cache(ticker_yf: Optional[str] = None) -> int:
    """
    Delete cache files.
    If ticker_yf given → delete only that ticker's files.
    If None → clear everything (use with care).
    Returns number of files deleted.
    """
    deleted = 0
    if ticker_yf:
        for path in [_price_path(ticker_yf), _fundamental_path(ticker_yf)]:
            if path.exists():
                path.unlink()
                deleted += 1
    else:
        for path in _CACHE_ROOT.rglob("*"):
            if path.is_file():
                path.unlink()
                deleted += 1
    return deleted
