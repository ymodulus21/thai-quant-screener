# Thai Quant Factor Screener

> Systematic multi-factor equity screener and walk-forward backtester for the Stock Exchange of Thailand (SET), built with Python and Streamlit.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Pandas](https://img.shields.io/badge/Pandas-2.2%2B-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?style=flat)](https://docs.pydantic.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What It Does

Most Thai retail investors rely on intuition or simple P/E screening. This tool applies **systematic factor investing methodology** — the same framework used by institutional quant funds — to the SET market:

1. **Screen** 53 stocks across SET50/SET100 using 5 factor dimensions
2. **Score** each stock with cross-sectional z-score normalization (MSCI/Barra standard)
3. **Backtest** any style strategy with walk-forward methodology and a full 16-metric tearsheet
4. **Compare** results against the SET Composite Index benchmark

---

## Live Demo

**[thai-quant-screener.streamlit.app](https://thai-quant-screener.streamlit.app)** ← deploy link (coming soon)

---

## Factor Methodology

Each stock is scored across 5 factor groups, then combined into a composite score.

| Factor Group | Metrics | Academic Basis |
|---|---|---|
| **Value** | P/E, P/B, P/S, EV/EBITDA | Fama-French (1992, 2012) |
| **Quality** | ROE, ROA, Gross Margin, Net Margin, Debt/Equity | Novy-Marx (2013), Asness et al. (2019) |
| **Momentum** | 1M, 3M, 6M, 12M price return | Carhart (1997), Jegadeesh-Titman (1993) |
| **Income** | Dividend Yield | Gordon Growth Model |
| **Growth** | Revenue Growth, Earnings Growth | Earnings yield literature |

### Normalization Pipeline

```
Raw factor value
  → Winsorize at [5th, 95th] percentile   ← removes outlier distortion
  → Cross-sectional z-score               ← ranks relative to peers, not absolute
  → Direction adjustment (×±1)            ← higher z always = better
  → Group average                         ← combines factors within group
  → Weighted sum by style preset          ← composite score
```

### Style Presets

| Style | Value | Quality | Momentum | Income | Growth |
|---|---|---|---|---|---|
| **Value** | 70% | 20% | 5% | 5% | 0% |
| **Quality** | 20% | 60% | 10% | 5% | 5% |
| **Momentum** | 10% | 10% | 75% | 5% | 0% |
| **Income** | 20% | 20% | 5% | 50% | 5% |
| **Growth** | 10% | 30% | 20% | 0% | 40% |
| **Blend** | 25% | 30% | 20% | 10% | 15% |

---

## Backtest Engine

- **Walk-forward** monthly rebalancing (no look-ahead on price data)
- **Equal-weight** within selected portfolio
- **Transaction cost** 0.25% one-way (realistic for Thai online brokers)
- **Benchmark** SET Composite Index (`^SET.BK`)

### 16-Metric Tearsheet

**Standalone**

| Metric | Description |
|---|---|
| CAGR | Compound Annual Growth Rate |
| Volatility | Annualized standard deviation |
| Sharpe | Excess return / volatility (correct: uses daily excess returns) |
| Sortino | Excess return / downside deviation |
| Max Drawdown | Largest peak-to-trough decline |
| Max DD Duration | Longest consecutive days underwater |
| Calmar | CAGR / \|Max Drawdown\| |
| Win Rate | % of days with positive return |
| VaR (95%) | Worst daily loss 95% of the time |
| CVaR (95%) | Average loss beyond VaR — Expected Shortfall |

**Benchmark-Relative**

| Metric | Description |
|---|---|
| Beta | Systematic risk vs SET index |
| Alpha (Jensen's) | Excess return after adjusting for beta risk |
| Information Ratio | Active return / tracking error — skill measure |
| Up Capture | % of SET upside captured |
| Down Capture | % of SET downside suffered (lower = better) |

---

## Architecture

```
C:\COMPANY\
├── src/
│   ├── data/                   # Data layer (shared by all modules)
│   │   ├── models.py           # Pydantic types: PriceData, FundamentalData, StockInfo
│   │   ├── universe.py         # SET50/SET100 registry — 53 stocks, 9 sectors
│   │   ├── cache.py            # Parquet (price) + JSON (fundamentals) disk cache
│   │   └── fetcher.py          # yfinance wrapper with retry, bulk fetch, TTL cache
│   │
│   ├── screener/               # Factor scoring engine
│   │   ├── factors.py          # Winsorize → z-score → direction mapping
│   │   ├── scorer.py           # Style presets + composite scoring
│   │   └── screener.py         # High-level Screener class
│   │
│   ├── backtest/               # Backtesting engine
│   │   ├── metrics.py          # 16 performance metrics
│   │   └── engine.py           # Walk-forward monthly rebalancer
│   │
│   └── app/
│       └── main.py             # Streamlit UI (Screener + Backtest + About)
│
├── .streamlit/
│   └── config.toml             # Dark theme config
└── requirements.txt
```

**Data flow:**
```
yfinance API
  → DataFetcher (cache + retry)
  → PriceData / FundamentalData (Pydantic models)
  → build_factor_dataframe()
  → compute_factor_scores() [screener path]     → CompositeScorer → ranked table
  → BacktestEngine._build_snapshot_factors()    → walk-forward loop → tearsheet
```

---

## Universe

- **53 stocks** across SET50 and SET100
- **9 sectors:** Energy, Financials, Consumer, Technology, Industrials, Property, Utilities, Petrochemicals, Healthcare
- Data source: Yahoo Finance via `yfinance` (15–20 min delay for price, daily fundamentals)

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/thai-quant-screener.git
cd thai-quant-screener
pip install -r requirements.txt
streamlit run src/app/main.py
```

Open [http://localhost:8501](http://localhost:8501)

### Screener (3 lines)

```python
from src.data import DataFetcher, SETIndex
from src.screener import Screener

result = Screener().run(index=SETIndex.SET50, style="blend", top_n=10)
print(result[["rank", "ticker", "sector", "composite_score", "pe_ratio", "roe", "mom_3m"]])
```

### Backtest (5 lines)

```python
from src.backtest import BacktestEngine
from src.data import SETIndex

bt = BacktestEngine().run(index=SETIndex.SET50, style="value", top_n=10,
                          start="2021-01-01", end="2025-12-01")
print(bt.summary)
```

---

## Known Limitations

This project is transparent about its constraints — understanding them is part of using it correctly.

| Limitation | Impact | Status |
|---|---|---|
| **Survivorship Bias** | Universe = current members only. Delisted stocks excluded → CAGR overstated ~1-3% | Disclosed |
| **Fundamental Look-Ahead** | P/E, ROE values are current snapshots, not point-in-time filing dates | Disclosed |
| **Small Universe** | 32–53 stocks is small for robust factor statistics vs S&P 500 (500 stocks) | Structural |
| **No Sector Neutrality** | Value scores favor Financials (naturally low P/E) vs Healthcare (naturally high P/E) | Roadmap |

---

## Roadmap

- [ ] Sector-neutral z-scoring
- [ ] Point-in-time fundamental data (quarterly filing dates)
- [ ] Historical index membership for survivorship-bias correction
- [ ] Risk model (covariance matrix, tracking error constraint)
- [ ] Freemium tier with user authentication

---

## Author

**Kittipong Mahaheng (Bass)**
First-year Finance & Banking student | Python quant developer | CFA L1 candidate

[![GitHub](https://img.shields.io/badge/GitHub-ymodulus21-181717?style=flat&logo=github)](https://github.com/ymodulus21)

---

## Disclaimer

This tool is for **educational and research purposes only**. It does not constitute financial advice. Past backtest performance does not guarantee future results. Always conduct your own due diligence before making investment decisions.

---

*"A backtest that flatters you is more dangerous than one that humbles you."*
