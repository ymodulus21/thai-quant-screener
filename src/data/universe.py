"""
SET Stock Universe Manager
--------------------------
Maintains curated ticker lists for SET50, SET100, SETHD indices.
Each ticker stored without .BK suffix internally; .BK appended for yfinance.

Source: SET official index constituents (updated Q1 2025).
To refresh: replace CONSTITUENTS dict and re-run.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from .models import AssetClass, SETIndex, StockInfo

# ---------------------------------------------------------------------------
# Constituent Registry
# Format: "TICKER": {"name": ..., "sector": ..., "industry": ...,
#                    "indices": [...SETIndex], "asset_class": ...}
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, dict] = {
    # --- Energy & Utilities ---
    "PTT":    {"name": "PTT Public Company",            "sector": "Energy",           "industry": "Oil & Gas",              "indices": [SETIndex.SET50, SETIndex.SET100]},
    "PTTEP":  {"name": "PTT Exploration & Production",  "sector": "Energy",           "industry": "Oil & Gas Exploration",  "indices": [SETIndex.SET50, SETIndex.SET100]},
    "PTTGC":  {"name": "PTT Global Chemical",           "sector": "Petrochemicals",   "industry": "Chemicals",              "indices": [SETIndex.SET50, SETIndex.SET100]},
    "TOP":    {"name": "Thai Oil",                       "sector": "Energy",           "industry": "Oil Refining",           "indices": [SETIndex.SET100]},
    "EGCO":   {"name": "Electricity Generating",         "sector": "Utilities",        "industry": "Power Generation",       "indices": [SETIndex.SET100]},
    "GPSC":   {"name": "Global Power Synergy",           "sector": "Utilities",        "industry": "Power Generation",       "indices": [SETIndex.SET50, SETIndex.SET100]},
    "GULF":   {"name": "Gulf Energy Development",        "sector": "Utilities",        "industry": "Power Generation",       "indices": [SETIndex.SET50, SETIndex.SET100]},
    "BGRIM":  {"name": "B.Grimm Power",                  "sector": "Utilities",        "industry": "Power Generation",       "indices": [SETIndex.SET100]},
    "EA":     {"name": "Energy Absolute",                "sector": "Utilities",        "industry": "Renewable Energy",       "indices": [SETIndex.SET100]},
    "RATCH":  {"name": "Ratch Group",                    "sector": "Utilities",        "industry": "Power Generation",       "indices": [SETIndex.SET100]},
    "BANPU":  {"name": "Banpu",                          "sector": "Energy",           "industry": "Coal & Diversified",     "indices": [SETIndex.SET50, SETIndex.SET100]},

    # --- Financials ---
    "KBANK":  {"name": "Kasikornbank",                   "sector": "Financials",       "industry": "Banking",                "indices": [SETIndex.SET50, SETIndex.SET100]},
    "SCB":    {"name": "SCB X (Siam Commercial Bank)",   "sector": "Financials",       "industry": "Banking",                "indices": [SETIndex.SET50, SETIndex.SET100]},
    "BBL":    {"name": "Bangkok Bank",                   "sector": "Financials",       "industry": "Banking",                "indices": [SETIndex.SET50, SETIndex.SET100]},
    "KTB":    {"name": "Krungthai Bank",                 "sector": "Financials",       "industry": "Banking",                "indices": [SETIndex.SET50, SETIndex.SET100]},
    "TTB":    {"name": "TMBThanachart Bank",             "sector": "Financials",       "industry": "Banking",                "indices": [SETIndex.SET100]},
    "BAY":    {"name": "Bank of Ayudhya (Krungsri)",     "sector": "Financials",       "industry": "Banking",                "indices": [SETIndex.SET100]},
    "TISCO":  {"name": "TISCO Financial Group",          "sector": "Financials",       "industry": "Banking",                "indices": [SETIndex.SET100, SETIndex.SETHD]},
    "KTC":    {"name": "Krungthai Card",                 "sector": "Financials",       "industry": "Consumer Finance",       "indices": [SETIndex.SET50, SETIndex.SET100]},
    "MTC":    {"name": "Muangthai Capital",              "sector": "Financials",       "industry": "Consumer Finance",       "indices": [SETIndex.SET50, SETIndex.SET100]},
    "TIDLOR": {"name": "Ngern Tid Lor",                  "sector": "Financials",       "industry": "Consumer Finance",       "indices": [SETIndex.SET100]},
    "SAWAD":  {"name": "SAWAD Corporation",              "sector": "Financials",       "industry": "Consumer Finance",       "indices": [SETIndex.SET100]},

    # --- Telecoms & Technology ---
    "ADVANC": {"name": "Advanced Info Service (AIS)",    "sector": "Technology",       "industry": "Telecom",                "indices": [SETIndex.SET50, SETIndex.SET100, SETIndex.SETHD]},
    # INTUCH delisted from Yahoo Finance — removed to prevent fetch errors
    "TRUE":   {"name": "True Corporation",               "sector": "Technology",       "industry": "Telecom",                "indices": [SETIndex.SET50, SETIndex.SET100]},
    "DTAC":   {"name": "Total Access Communication",     "sector": "Technology",       "industry": "Telecom",                "indices": [SETIndex.SET100]},

    # --- Consumer ---
    "CPALL":  {"name": "CP All (7-Eleven Thailand)",     "sector": "Consumer",         "industry": "Retail - Convenience",   "indices": [SETIndex.SET50, SETIndex.SET100]},
    "CPF":    {"name": "Charoen Pokphand Foods",         "sector": "Consumer",         "industry": "Food Production",        "indices": [SETIndex.SET50, SETIndex.SET100]},
    "CRC":    {"name": "Central Retail Corporation",     "sector": "Consumer",         "industry": "Retail - Department",    "indices": [SETIndex.SET50, SETIndex.SET100]},
    "MINT":   {"name": "Minor International",            "sector": "Consumer",         "industry": "Hotels & Restaurants",   "indices": [SETIndex.SET50, SETIndex.SET100]},
    "CENTEL": {"name": "Central Plaza Hotel",            "sector": "Consumer",         "industry": "Hotels & Restaurants",   "indices": [SETIndex.SET100]},
    "HMPRO":  {"name": "Home Product Center",            "sector": "Consumer",         "industry": "Home Improvement Retail","indices": [SETIndex.SET50, SETIndex.SET100]},
    "COM7":   {"name": "Com7",                           "sector": "Consumer",         "industry": "Electronics Retail",     "indices": [SETIndex.SET100]},
    "TU":     {"name": "Thai Union Group",               "sector": "Consumer",         "industry": "Food Production",        "indices": [SETIndex.SET100]},
    "OSP":    {"name": "Osotspa",                        "sector": "Consumer",         "industry": "Beverages",              "indices": [SETIndex.SET100]},
    "CBG":    {"name": "Carabao Group",                  "sector": "Consumer",         "industry": "Beverages",              "indices": [SETIndex.SET100]},

    # --- Industrials & Infrastructure ---
    "SCC":    {"name": "SCG (Siam Cement Group)",        "sector": "Industrials",      "industry": "Construction Materials", "indices": [SETIndex.SET50, SETIndex.SET100]},
    "SCGP":   {"name": "SCG Packaging",                  "sector": "Industrials",      "industry": "Paper & Packaging",      "indices": [SETIndex.SET50, SETIndex.SET100]},
    "IVL":    {"name": "Indorama Ventures",              "sector": "Petrochemicals",   "industry": "Chemicals",              "indices": [SETIndex.SET50, SETIndex.SET100]},
    "DELTA":  {"name": "Delta Electronics Thailand",     "sector": "Technology",       "industry": "Electronic Components",  "indices": [SETIndex.SET50, SETIndex.SET100]},
    "HANA":   {"name": "Hana Microelectronics",          "sector": "Technology",       "industry": "Semiconductor",          "indices": [SETIndex.SET100]},
    "BEM":    {"name": "Bangkok Expressway & Metro",     "sector": "Industrials",      "industry": "Transportation Infra",   "indices": [SETIndex.SET50, SETIndex.SET100]},
    "BTS":    {"name": "BTS Group Holdings",             "sector": "Industrials",      "industry": "Transportation",         "indices": [SETIndex.SET50, SETIndex.SET100]},
    "AOT":    {"name": "Airports of Thailand",           "sector": "Industrials",      "industry": "Airports",               "indices": [SETIndex.SET50, SETIndex.SET100]},

    # --- Property ---
    "LH":     {"name": "Land and Houses",                "sector": "Property",         "industry": "Residential Dev",        "indices": [SETIndex.SET50, SETIndex.SET100, SETIndex.SETHD]},
    "ORI":    {"name": "Origin Property",                "sector": "Property",         "industry": "Residential Dev",        "indices": [SETIndex.SET100]},
    "CPN":    {"name": "Central Pattana",                "sector": "Property",         "industry": "Commercial REIT/Dev",    "indices": [SETIndex.SET50, SETIndex.SET100]},
    "AWC":    {"name": "Asset World Corp",               "sector": "Property",         "industry": "Hospitality Property",   "indices": [SETIndex.SET50, SETIndex.SET100]},
    "WHA":    {"name": "WHA Corporation",                "sector": "Property",         "industry": "Industrial Estate",      "indices": [SETIndex.SET100]},

    # --- Healthcare ---
    "BDMS":   {"name": "Bangkok Dusit Medical Services", "sector": "Healthcare",       "industry": "Hospitals",              "indices": [SETIndex.SET50, SETIndex.SET100]},
    "BH":     {"name": "Bumrungrad International Hospital","sector": "Healthcare",     "industry": "Hospitals",              "indices": [SETIndex.SET50, SETIndex.SET100]},
    "BCH":    {"name": "Bangkok Chain Hospital",         "sector": "Healthcare",       "industry": "Hospitals",              "indices": [SETIndex.SET100]},

    # --- Oil & Retail ---
    "OR":     {"name": "PTT Oil and Retail Business",   "sector": "Energy",           "industry": "Oil Retail",             "indices": [SETIndex.SET50, SETIndex.SET100]},
    "GLOBAL": {"name": "Siam Global House",             "sector": "Consumer",         "industry": "Home Improvement Retail","indices": [SETIndex.SET100]},
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def get_all_stocks() -> list[StockInfo]:
    """Return StockInfo for every ticker in the registry."""
    result = []
    for ticker, meta in _REGISTRY.items():
        result.append(StockInfo(
            ticker=ticker,
            ticker_yf=f"{ticker}.BK",
            name=meta["name"],
            sector=meta.get("sector"),
            industry=meta.get("industry"),
            index_membership=meta.get("indices", []),
            asset_class=meta.get("asset_class", AssetClass.EQUITY),
        ))
    return result


@lru_cache(maxsize=None)
def get_universe(index: SETIndex = SETIndex.SET50) -> list[StockInfo]:
    """
    Return stocks belonging to a given index.
    SETIndex.ALL returns the entire registry.
    """
    all_stocks = get_all_stocks()
    if index == SETIndex.ALL:
        return all_stocks
    return [s for s in all_stocks if index in s.index_membership]


def get_tickers_yf(index: SETIndex = SETIndex.SET50) -> list[str]:
    """Return Yahoo Finance ticker strings (e.g. ['PTT.BK', 'KBANK.BK', ...])."""
    return [s.ticker_yf for s in get_universe(index)]


def get_tickers(index: SETIndex = SETIndex.SET50) -> list[str]:
    """Return clean ticker strings (e.g. ['PTT', 'KBANK', ...])."""
    return [s.ticker for s in get_universe(index)]


def find_stock(ticker: str) -> Optional[StockInfo]:
    """Lookup a single stock by ticker (case-insensitive). Returns None if not found."""
    ticker_clean = ticker.upper().replace(".BK", "").strip()
    return next(
        (s for s in get_all_stocks() if s.ticker == ticker_clean),
        None,
    )


def universe_summary() -> dict:
    """Quick stats on the universe — useful for debugging."""
    all_s = get_all_stocks()
    sectors = {}
    for s in all_s:
        sectors[s.sector or "Unknown"] = sectors.get(s.sector or "Unknown", 0) + 1

    return {
        "total_stocks": len(all_s),
        "set50_count":  len(get_universe(SETIndex.SET50)),
        "set100_count": len(get_universe(SETIndex.SET100)),
        "sethd_count":  len(get_universe(SETIndex.SETHD)),
        "sectors": sectors,
    }
