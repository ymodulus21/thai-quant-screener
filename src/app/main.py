"""
Thai Quant Screener — Streamlit App
-------------------------------------
Run with:  streamlit run src/app/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from src.backtest import BacktestEngine, performance_summary
from src.data import DataFetcher, SETIndex, cache_stats, universe_summary
from src.screener import STYLE_PRESETS, Screener

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Thai Quant Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session-level singletons (cached across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_screener() -> Screener:
    return Screener(fetcher=DataFetcher(use_cache=True))

@st.cache_resource(show_spinner=False)
def get_engine() -> BacktestEngine:
    return BacktestEngine(fetcher=get_screener().fetcher)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 Thai Quant Screener")
    st.caption("SET Factor Investing Tool")

    st.divider()

    INDEX_OPTIONS = {
        "SET50  (33 stocks)":  SETIndex.SET50,
        "SET100 (54 stocks)":  SETIndex.SET100,
    }
    selected_index_label = st.selectbox("Universe", list(INDEX_OPTIONS))
    selected_index = INDEX_OPTIONS[selected_index_label]

    style = st.selectbox(
        "Investment Style",
        list(STYLE_PRESETS.keys()),
        index=list(STYLE_PRESETS.keys()).index("blend"),
        format_func=lambda x: x.title(),
    )

    top_n = st.slider("Top N Stocks", min_value=5, max_value=20, value=10, step=1)

    st.divider()

    # Style descriptions
    style_desc = {
        "value":    "Low P/E, low P/B — buy cheap stocks",
        "quality":  "High ROE, low debt — buy strong businesses",
        "momentum": "Recent price winners — trend following",
        "income":   "High dividend yield — cash flow focus",
        "growth":   "Revenue & earnings expansion",
        "blend":    "Balanced across all factors",
    }
    st.info(f"**{style.title()}:** {style_desc[style]}")

    st.divider()
    stats = cache_stats()
    st.caption(f"Cache: {stats['price_fresh']} price / {stats['fundamental_fresh']} fundamental files fresh")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_screen, tab_backtest, tab_about = st.tabs(["🔍 Screener", "📈 Backtest", "ℹ️ About"])


# ── Tab 1: Screener ─────────────────────────────────────────────────────────
with tab_screen:
    st.header("Factor Screener")

    run_btn = st.button("Run Screener", type="primary", use_container_width=True)

    if run_btn or "screener_result" in st.session_state:
        if run_btn:
            with st.spinner("Fetching data & computing scores..."):
                screener = get_screener()
                result = screener.run(
                    index=selected_index,
                    style=style,
                    top_n=top_n,
                    verbose=False,
                )
            st.session_state["screener_result"] = result
        else:
            result = st.session_state["screener_result"]

        if result.empty:
            st.warning("No results. Try a broader universe or different filters.")
        else:
            # ── KPI row ──────────────────────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Stocks Screened", len(result))
            with c2:
                top1 = result.iloc[0]
                st.metric("Top Pick", top1["ticker"], delta=f"Score: {top1['composite_score']:.2f}")
            with c3:
                avg_pe = result["pe_ratio"].dropna().mean()
                st.metric("Avg P/E (Top N)", f"{avg_pe:.1f}x" if pd.notna(avg_pe) else "N/A")
            with c4:
                avg_roe = result["roe"].dropna().mean()
                st.metric("Avg ROE (Top N)", f"{avg_roe:.1%}" if pd.notna(avg_roe) else "N/A")

            st.divider()

            # ── Ranked table ─────────────────────────────────────────────
            display_cols = ["rank", "ticker", "name", "sector", "composite_score",
                            "pe_ratio", "pb_ratio", "roe", "dividend_yield",
                            "mom_3m", "mom_6m", "mkt_cap_bn"]
            display_cols = [c for c in display_cols if c in result.columns]

            format_map = {
                "composite_score": "{:.3f}",
                "pe_ratio":        "{:.1f}",
                "pb_ratio":        "{:.2f}",
                "roe":             "{:.1%}",
                "dividend_yield":  "{:.1%}",
                "mom_3m":          "{:.1%}",
                "mom_6m":          "{:.1%}",
                "mkt_cap_bn":      "{:.0f}B",
            }
            fmt = {k: v for k, v in format_map.items() if k in display_cols}

            st.dataframe(
                result[display_cols].style.format(fmt, na_rep="—"),
                use_container_width=True,
                height=420,
            )

            # ── Sector breakdown ─────────────────────────────────────────
            if "sector" in result.columns:
                sector_counts = result["sector"].value_counts()
                st.subheader("Sector Breakdown")
                st.bar_chart(sector_counts)


# ── Tab 2: Backtest ──────────────────────────────────────────────────────────
with tab_backtest:
    st.header("Strategy Backtest")
    st.caption("Monthly rebalance | Equal-weight | 0.25% transaction cost")

    col_l, col_r = st.columns([1, 2])
    with col_l:
        bt_start = st.date_input("Start Date", value=pd.Timestamp("2021-01-01"))
        bt_end   = st.date_input("End Date",   value=pd.Timestamp("2025-12-31"))
        bt_btn   = st.button("Run Backtest", type="primary", use_container_width=True)

    if bt_btn:
        with st.spinner("Running walk-forward backtest... (may take ~30s)"):
            try:
                engine = get_engine()
                result = engine.run(
                    index=selected_index,
                    style=style,
                    top_n=top_n,
                    start=str(bt_start),
                    end=str(bt_end),
                    verbose=False,
                )
                st.session_state["bt_result"] = result
            except Exception as e:
                st.error(f"Backtest error: {e}")

    if "bt_result" in st.session_state:
        bt = st.session_state["bt_result"]

        # ── Summary metrics ──────────────────────────────────────────────
        st.subheader("Performance Summary")
        st.dataframe(bt.summary, use_container_width=True)

        # ── Equity curves ────────────────────────────────────────────────
        st.subheader("Equity Curve vs SET Index")
        curve_df = pd.DataFrame({
            f"{style.title()} Top{top_n}": bt.equity_curve,
            "SET Index":                   bt.benchmark_equity_curve,
        })
        st.line_chart(curve_df)

        # ── Drawdown ─────────────────────────────────────────────────────
        st.subheader("Drawdown")
        dd_df = bt.drawdown.rename("Drawdown")
        st.area_chart(dd_df)

        # ── Portfolio history ─────────────────────────────────────────────
        with st.expander("Monthly Portfolio Holdings"):
            hist_rows = [
                {"date": d, "portfolio": ", ".join(t.replace(".BK","") for t in tickers)}
                for d, tickers in bt.portfolio_history.items()
            ]
            st.dataframe(pd.DataFrame(hist_rows), use_container_width=True)


# ── Tab 3: About ─────────────────────────────────────────────────────────────
with tab_about:
    univ = universe_summary()
    st.header("About This Tool")
    st.markdown(f"""
    **Thai Quant Factor Screener** applies systematic factor investing methodology
    to the Stock Exchange of Thailand (SET).

    ### How Scoring Works
    Each stock is scored across 5 factor dimensions:

    | Factor Group | Metrics Used | CFA Reference |
    |---|---|---|
    | **Value** | P/E, P/B, P/S, EV/EBITDA | Equity Valuation |
    | **Quality** | ROE, ROA, Gross Margin, Debt/Equity | Financial Analysis |
    | **Momentum** | 1M, 3M, 6M, 12M price return | Technical/Behavioral |
    | **Income** | Dividend Yield | Fixed Income / Equity |
    | **Growth** | Revenue Growth, Earnings Growth | Equity Analysis |

    Each metric is **z-score normalized** within the universe — so the score tells you
    how a stock ranks *relative to peers*, not in absolute terms.

    ### Universe
    - **{univ['total_stocks']} total stocks** across {len(univ['sectors'])} sectors
    - SET50: {univ['set50_count']} stocks | SET100: {univ['set100_count']} stocks
    - Data: Yahoo Finance via yfinance (15–20 min delay)

    ### Backtest Methodology
    - Monthly rebalance on last trading day of each month
    - Equal-weight within selected portfolio
    - 0.25% one-way transaction cost on turnover
    - Benchmark: SET Composite Index (^SET.BK)

    ### Risk Warning
    Past performance does not guarantee future results.
    This tool is for **educational and research purposes only**.
    """)
