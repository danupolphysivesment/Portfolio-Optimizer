"""Quant Portfolio Lab — Streamlit application.

Two workbenches built on real historical ETF data:
  1. Optimizer  — choose assets, estimate moments from history, and build a
                  portfolio with one of ten construction methods.
  2. Backtester — stress-test any set of weights with periodic rebalancing and
                  transaction costs, or run a walk-forward (out-of-sample) test.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from portlab import backtest as bt
from portlab import data as dataio
from portlab import estimators as est
from portlab import optimizers as opt
from portlab import macro as macrolib
from portlab import metrics
from portlab import montecarlo as mc
from portlab import risk as risklib
from portlab import universe as uni

# --------------------------------------------------------------------------- #
# Page setup & styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Quant Portfolio Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Luxury "private wealth" palette ------------------------------------------- #
INK = "#16263A"        # deep navy ink (primary text / lines)
INK_SOFT = "#5E6B7A"   # muted text
IVORY = "#FAF8F3"      # warm canvas
GOLD = "#B6924C"       # champagne gold accent
GOLD_DEEP = "#8A6D2F"  # gold for small text on light

CLASS_COLORS = {
    uni.EQUITY: "#34557F",        # muted sapphire
    uni.FIXED_INCOME: "#2E7D6F",  # deep emerald-teal
    uni.ALTERNATIVES: "#B0883B",  # antique gold/bronze
    "Unknown": "#8A93A0",
}

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

      /* ---- Canvas & typography ---- */
      .stApp { background: #FAF8F3; }
      html, body, [class*="css"], .stMarkdown, .stMetric, input, textarea, button, select {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      }
      .block-container { padding-top: 2.0rem; max-width: 1280px; }
      h1, h2, h3, h4 {
        font-family: 'Playfair Display', Georgia, serif !important;
        color: #16263A; letter-spacing: 0.005em; font-weight: 600;
      }
      h1 { font-size: 2.15rem !important; line-height: 1.15; }
      /* Gold hairline under page titles */
      h1::after {
        content: ""; display: block; width: 56px; height: 2px; margin-top: 0.55rem;
        background: linear-gradient(90deg, #B6924C, rgba(182,146,76,0));
      }

      /* ---- Sidebar ---- */
      [data-testid="stSidebar"] {
        background: #F1ECE1;
        border-right: 1px solid #E6DEcd;
      }
      [data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

      /* ---- Metric cards ---- */
      [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #ECE4D3;
        border-left: 3px solid #B6924C;
        border-radius: 10px;
        padding: 14px 16px 12px 16px;
        box-shadow: 0 1px 2px rgba(22,38,58,0.04), 0 10px 24px rgba(22,38,58,0.05);
      }
      [data-testid="stMetricLabel"] p {
        text-transform: uppercase; letter-spacing: 0.09em;
        font-size: 0.68rem; font-weight: 600; color: #7A8696;
      }
      [data-testid="stMetricValue"] {
        font-size: 1.5rem; font-weight: 600; color: #16263A;
        font-variant-numeric: tabular-nums;
      }

      /* ---- Buttons ---- */
      .stButton > button, .stDownloadButton > button {
        border-radius: 8px; border: 1px solid #D8CDB5; font-weight: 600;
        letter-spacing: 0.01em; transition: all 0.15s ease;
      }
      .stButton > button[kind="primary"] {
        background: #1B2A41; border-color: #1B2A41; color: #F6EFDD;
      }
      .stButton > button[kind="primary"]:hover {
        background: #16263A; border-color: #B6924C;
        box-shadow: 0 4px 14px rgba(22,38,58,0.18);
      }

      /* ---- Dividers, expanders, tables ---- */
      hr { border-color: #E6DEcd !important; }
      [data-testid="stExpander"] {
        border: 1px solid #ECE4D3 !important; border-radius: 10px;
        background: #FFFFFF; box-shadow: 0 6px 18px rgba(22,38,58,0.04);
      }
      [data-testid="stExpander"] summary { font-weight: 600; color: #16263A; }
      [data-testid="stDataFrame"] { border: 1px solid #ECE4D3; border-radius: 10px; }

      /* ---- Brand wordmark (sidebar) ---- */
      .pl-brand { font-family: 'Playfair Display', serif; font-size: 1.32rem;
                  font-weight: 700; color: #16263A; line-height: 1.1; margin-bottom: 2px; }
      .pl-brand .pl-amp { color: #B6924C; }
      .pl-kicker { text-transform: uppercase; letter-spacing: 0.22em;
                   font-size: 0.62rem; font-weight: 600; color: #9A8552; margin-bottom: 0.2rem; }
      .pl-rule { height: 1px; background: linear-gradient(90deg, #B6924C, rgba(182,146,76,0));
                 margin: 0.7rem 0 0.2rem 0; }

      /* ---- Inputs: lock to luxury palette regardless of base config ---- */
      span[data-baseweb="tag"] {
        background-color: #1B2A41 !important; border-radius: 6px !important;
      }
      span[data-baseweb="tag"] span, span[data-baseweb="tag"] svg { color: #F4EFE2 !important; fill: #F4EFE2 !important; }
      [data-baseweb="slider"] [role="slider"] { background: #B6924C !important; }
      [data-testid="stWidgetLabel"] p, label p { color: #3A4654; font-weight: 500; }

      /* ---- Report-style page masthead ---- */
      .pl-pagehead { display:flex; justify-content:space-between; align-items:center;
                     margin-bottom:-0.35rem; }
      .pl-eyebrow { text-transform:uppercase; letter-spacing:0.24em; font-size:0.62rem;
                    font-weight:700; color:#9A8552; }
      .pl-asof { text-transform:uppercase; letter-spacing:0.14em; font-size:0.62rem;
                 font-weight:600; color:#9AA6B4; }

      /* ---- Misc helpers ---- */
      .pl-tag { display:inline-block; padding:3px 11px; border-radius:999px;
                font-size:0.7rem; font-weight:600; color:#fff; letter-spacing:0.02em; }
      .pl-sub { color:#5E6B7A; font-size:0.95rem; line-height:1.5; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Plotly luxury theme (applied to every chart) ------------------------- #
import plotly.io as pio

pio.templates["lux"] = go.layout.Template(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#3A4654", size=13),
        title=dict(font=dict(family="Playfair Display, Georgia, serif",
                             color="#16263A", size=17), x=0.01, xanchor="left"),
        colorway=["#1B2A41", "#B6924C", "#2E7D6F", "#34557F", "#9A6A4B", "#5E6B7A"],
        xaxis=dict(gridcolor="rgba(22,38,58,0.07)", zerolinecolor="rgba(22,38,58,0.12)",
                   linecolor="rgba(22,38,58,0.18)", tickcolor="rgba(22,38,58,0.18)"),
        yaxis=dict(gridcolor="rgba(22,38,58,0.07)", zerolinecolor="rgba(22,38,58,0.12)",
                   linecolor="rgba(22,38,58,0.18)", tickcolor="rgba(22,38,58,0.18)"),
        legend=dict(font=dict(color="#3A4654")),
        hoverlabel=dict(bgcolor="#16263A", font=dict(color="#F4EFE2", family="Inter")),
        colorscale=dict(sequential="Cividis"),
    )
)
pio.templates.default = "lux"


# --------------------------------------------------------------------------- #
# Cached data layer
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_prices_cached(tickers: tuple, start: str, end: str, source: str, convert_usd: bool = True):
    return dataio.load_prices(list(tickers), start, end, source=source, convert_usd=convert_usd)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_macro_cached(start: str, end: str, source: str):
    return macrolib.load_factor_changes(start, end, source)


def color_for(ticker: str) -> str:
    return CLASS_COLORS.get(uni.asset_class_of(ticker), CLASS_COLORS["Unknown"])


def fmt_pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def yahoo_url(ticker: str) -> str:
    """Yahoo Finance quote page for an ETF (holdings, performance, profile)."""
    from urllib.parse import quote

    return f"https://finance.yahoo.com/quote/{quote(ticker)}"


# --------------------------------------------------------------------------- #
# Sidebar — global settings
# --------------------------------------------------------------------------- #
def sidebar_settings() -> dict:
    st.sidebar.markdown(
        '<div class="pl-kicker">Private Wealth Desk</div>'
        '<div class="pl-brand">Quant Portfolio <span class="pl-amp">Lab</span></div>'
        '<div class="pl-rule"></div>'
        '<div class="pl-sub" style="font-size:0.8rem">Backtesting &amp; portfolio '
        "construction on historical market data</div>",
        unsafe_allow_html=True,
    )

    section = st.sidebar.radio(
        "Workbench",
        ["🧮 Optimizer", "🧪 Backtester", "🎲 Monte Carlo", "🌐 Macro Impulse",
         "📚 Asset Universe", "📖 Methods"],
        label_visibility="collapsed",
        key="cfg_section",
    )

    st.sidebar.divider()
    st.sidebar.markdown("### Data & estimation")
    source = st.sidebar.selectbox("Data source", dataio.SOURCES, index=0, key="cfg_source")
    convert_usd = True
    if source == dataio.SYNTHETIC:
        st.sidebar.caption("⚠️ Offline demo data — not real markets.")
    else:
        convert_usd = st.sidebar.toggle(
            "Convert prices to USD", value=True, key="cfg_usd",
            help="Normalize EUR/JPY/GBp listings to USD via FX so cross-currency "
                 "portfolios are comparable.",
        )

    c1, c2 = st.sidebar.columns(2)
    start = c1.date_input("Start", value=pd.Timestamp("2010-01-01")).strftime("%Y-%m-%d")
    end = c2.date_input("End", value=pd.Timestamp.today()).strftime("%Y-%m-%d")

    ret_method = st.sidebar.selectbox(
        "Expected-return estimator", ["Historical mean", "EWMA"], index=0
    )
    cov_method = st.sidebar.selectbox(
        "Covariance estimator",
        ["Ledoit-Wolf shrinkage", "Sample", "EWMA"],
        index=0,
    )
    span = 90
    if ret_method == "EWMA" or cov_method == "EWMA":
        span = st.sidebar.slider("EWMA span (days)", 20, 252, 90, 10)
    rf = st.sidebar.number_input(
        "Risk-free rate (annual)", 0.0, 0.10, 0.02, 0.005, format="%.3f"
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "Built for quantitative research & teaching. Estimates are historical "
        "and not investment advice."
    )
    return dict(
        section=section, source=source, start=start, end=end,
        ret_method=ret_method, cov_method=cov_method, span=span, rf=rf,
        convert_usd=convert_usd,
    )


# --------------------------------------------------------------------------- #
# Shared widgets
# --------------------------------------------------------------------------- #
def asset_picker(key: str, default=None) -> list:
    """Grouped multiselects (one per asset class) with an optional region filter.

    Options are rich display strings ("TICKER · note") mapped back to tickers,
    so the universe of 60+ ETFs stays scannable and searchable.
    """
    default = default or uni.default_selection()
    sel_regions = st.multiselect(
        "Filter equities/assets by region (optional)",
        uni.all_regions(), default=[], key=f"{key}_region",
        help="Leave empty to show every region.",
    )
    chosen = []
    cols = st.columns(3)
    for col, ac in zip(cols, uni.ASSET_CLASSES):
        with col:
            assets = uni.assets_in_class(ac)
            if sel_regions:
                assets = [a for a in assets if a.region in sel_regions]
            disp_to_ticker = {uni.display_name(a): a.ticker for a in assets}
            ticker_to_disp = {a.ticker: uni.display_name(a) for a in assets}
            options = list(disp_to_ticker.keys())
            default_disp = [ticker_to_disp[t] for t in default if t in ticker_to_disp]
            picked = st.multiselect(
                f"{ac}  ({len(options)})", options, default=default_disp, key=f"{key}_{ac}"
            )
            chosen.extend(disp_to_ticker[d] for d in picked)
    return chosen


def page_header(title: str, subtitle: str):
    """Consistent report-style masthead: brand eyebrow + 'as of' stamp + title."""
    today = pd.Timestamp.today().strftime("%d %b %Y").upper()
    st.markdown(
        '<div class="pl-pagehead">'
        '<div class="pl-eyebrow">Quant Portfolio Lab</div>'
        f'<div class="pl-asof">As of {today}</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.title(title)
    st.markdown(f'<div class="pl-sub">{subtitle}</div>', unsafe_allow_html=True)


def estimate_moments(prices: pd.DataFrame, cfg: dict):
    rets = est.to_returns(prices)
    ppy = est.infer_periods_per_year(rets.index)
    mu = est.mean_returns(rets, cfg["ret_method"], cfg["span"], ppy)
    cov = est.cov_matrix(rets, cfg["cov_method"], cfg["span"], ppy)
    return rets, mu, cov, ppy


_WEIGHT_METHODS = {
    "Equal Weight": opt.EQUAL_WEIGHT, "Max Sharpe": opt.MAX_SHARPE,
    "Min Volatility": opt.MIN_VOL, "Equal Risk Contribution": opt.ERC,
    "Inverse Volatility": opt.INVERSE_VOL,
}


def weights_selector(tickers, mu, cov, rf, key, container=None) -> pd.Series:
    """Compact portfolio-weighting picker (method or manual). Returns weights."""
    c = container or st
    choice = c.selectbox("Portfolio weights", list(_WEIGHT_METHODS) + ["Manual weights"],
                         index=1, key=f"{key}_method")
    if choice == "Manual weights":
        wdf = pd.DataFrame({"Asset": tickers, "Weight": [round(1 / len(tickers), 4)] * len(tickers)})
        edited = c.data_editor(
            wdf, hide_index=True, use_container_width=True, key=f"{key}_w",
            column_config={"Asset": st.column_config.TextColumn(disabled=True),
                           "Weight": st.column_config.NumberColumn(format="%.4f")},
        )
        w = pd.Series(np.array(edited["Weight"], float), index=edited["Asset"])
        return w / w.sum() if w.sum() > 0 else w
    return opt.optimize(_WEIGHT_METHODS[choice], mu=mu, cov=cov, rf=rf)


def metrics_row(stats: dict, keys: list):
    cols = st.columns(len(keys))
    pct_keys = {
        "Total Return", "CAGR", "Volatility (ann.)", "Max Drawdown",
        "VaR 95% (period)", "CVaR 95% (period)", "Best Period",
        "Worst Period", "Win Rate",
    }
    for col, k in zip(cols, keys):
        v = stats[k]
        col.metric(k, fmt_pct(v, 1) if k in pct_keys else f"{v:.2f}")


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def weights_bar(weights: pd.Series, title: str = "Portfolio weights") -> go.Figure:
    w = weights[weights.abs() > 1e-6].sort_values()
    fig = go.Figure(
        go.Bar(
            x=w.values, y=[uni.label(t) for t in w.index], orientation="h",
            marker_color=[color_for(t) for t in w.index],
            text=[fmt_pct(v, 1) for v in w.values], textposition="outside",
        )
    )
    fig.update_layout(
        title=title, xaxis_tickformat=".0%", height=max(260, 34 * len(w) + 90),
        margin=dict(l=10, r=30, t=50, b=10), xaxis_title="Weight",
    )
    return fig


def class_donut(weights: pd.Series) -> go.Figure:
    by_class = {}
    for t, w in weights.items():
        by_class[uni.asset_class_of(t)] = by_class.get(uni.asset_class_of(t), 0) + w
    labels = list(by_class.keys())
    fig = go.Figure(
        go.Pie(
            labels=labels, values=[by_class[k] for k in labels], hole=0.55,
            marker_colors=[CLASS_COLORS.get(k) for k in labels],
            textinfo="label+percent", sort=False,
        )
    )
    fig.update_layout(
        title="Allocation by asset class", height=320,
        margin=dict(l=10, r=10, t=50, b=10), showlegend=False,
    )
    return fig


def risk_contrib_bar(weights: pd.Series, cov: pd.DataFrame) -> go.Figure:
    rc = opt.risk_contributions(weights, cov)
    rc = rc[weights.abs() > 1e-6].sort_values()
    fig = go.Figure(
        go.Bar(
            x=rc.values, y=[uni.label(t) for t in rc.index], orientation="h",
            marker_color=[color_for(t) for t in rc.index],
            text=[fmt_pct(v, 1) for v in rc.values], textposition="outside",
        )
    )
    fig.update_layout(
        title="Risk contribution (share of portfolio variance)",
        xaxis_tickformat=".0%", height=max(260, 34 * len(rc) + 90),
        margin=dict(l=10, r=30, t=50, b=10),
    )
    return fig


def frontier_fig(mu, cov, current_w, rf) -> go.Figure:
    ef = opt.efficient_frontier(mu, cov, n_points=30)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ef["vol"], y=ef["ret"], mode="lines", name="Efficient frontier",
        line=dict(color="#1B2A41", width=3),
    ))
    # individual assets
    fig.add_trace(go.Scatter(
        x=np.sqrt(np.diag(cov.values)), y=mu.values, mode="markers+text",
        text=list(mu.index), textposition="top center", name="Assets",
        marker=dict(color=[color_for(t) for t in mu.index], size=9),
    ))
    # reference portfolios
    for name, w, sym, col in [
        ("Max Sharpe", opt.max_sharpe(mu, cov, rf), "star", "#B6924C"),
        ("Min Vol", opt.min_volatility(cov), "diamond", "#2E7D6F"),
        ("Selected", current_w, "circle", "#16263A"),
    ]:
        fig.add_trace(go.Scatter(
            x=[opt.portfolio_vol(w.values, cov.values)],
            y=[opt.portfolio_return(w.values, mu.values)],
            mode="markers", name=name,
            marker=dict(symbol=sym, size=16, color=col,
                        line=dict(width=1, color="white")),
        ))
    fig.update_layout(
        title="Efficient frontier (annualized)", height=460,
        xaxis_title="Volatility", yaxis_title="Expected return",
        xaxis_tickformat=".0%", yaxis_tickformat=".0%",
        margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2),
    )
    return fig


def equity_fig(curves: dict) -> go.Figure:
    fig = go.Figure()
    palette = ["#1B2A41", "#B6924C", "#2E7D6F", "#34557F"]
    for (name, curve), c in zip(curves.items(), palette):
        fig.add_trace(go.Scatter(
            x=curve.index, y=curve.values, mode="lines", name=name,
            line=dict(width=2.4 if name != "Benchmark" else 1.8, color=c,
                      dash="solid" if name != "Benchmark" else "dash"),
        ))
    fig.update_layout(
        title="Growth of initial capital", height=420,
        yaxis_title="Portfolio value", margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def drawdown_fig(returns: pd.Series) -> go.Figure:
    dd = metrics.drawdown_series(returns)
    fig = go.Figure(go.Scatter(
        x=dd.index, y=dd.values, fill="tozeroy", mode="lines",
        line=dict(color="#9E3B34", width=1.2), fillcolor="rgba(158,59,52,0.12)",
        name="Drawdown",
    ))
    fig.update_layout(
        title="Drawdown", height=260, yaxis_tickformat=".0%",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def weight_area_fig(weight_df: pd.DataFrame) -> go.Figure:
    # Offset object (not the "M" alias) so this works on pandas 1.5 → 3.x.
    wd = weight_df.dropna(how="all").resample(pd.offsets.MonthEnd()).last().dropna(how="all")
    fig = go.Figure()
    for t in wd.columns:
        fig.add_trace(go.Scatter(
            x=wd.index, y=wd[t], mode="lines", name=t, stackgroup="one",
            line=dict(width=0.5, color=color_for(t)),
        ))
    fig.update_layout(
        title="Weight drift over time (month-end)", height=320,
        yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


# --------------------------------------------------------------------------- #
# Section: Optimizer
# --------------------------------------------------------------------------- #
def method_controls(method: str, tickers: list, mu: pd.Series, rets: pd.DataFrame, ppy: float):
    """Render method-specific inputs; return a kwargs dict for the engine."""
    kw = {}
    with st.container():
        if method in (opt.MAX_SHARPE, opt.MIN_VOL, opt.TARGET_RETURN, opt.ROBUST,
                      opt.BLACK_LITTERMAN, opt.ENTROPY_POOLING):
            c1, c2 = st.columns(2)
            long_only = c1.toggle("Long only", value=True, key="lo")
            max_w = c2.slider("Max weight per asset", 0.1, 1.0, 1.0, 0.05, key="mw")
            lo = 0.0 if long_only else -1.0
            kw["weight_bounds"] = (lo, float(max_w))

        if method == opt.TARGET_RETURN:
            lo_r, hi_r = float(mu.min()), float(mu.max())
            default = float(np.clip(np.median(mu.values), lo_r, hi_r))
            kw["target"] = st.slider(
                "Target annual return", round(lo_r, 3), round(hi_r, 3),
                round(default, 3), 0.005, format="%.3f",
            )

        if method == opt.ROBUST:
            kw["kappa"] = st.slider(
                "Uncertainty aversion κ", 0.0, 4.0, 1.0, 0.25,
                help="Higher κ distrusts the estimated means more, shrinking "
                     "toward the minimum-variance portfolio.",
            )
            kw["n_obs"] = len(rets)

        if method == opt.RISK_BUDGET:
            st.caption("Set each asset's share of total portfolio risk (auto-normalized).")
            bdf = pd.DataFrame({"Asset": tickers, "Risk budget": [1.0] * len(tickers)})
            edited = st.data_editor(
                bdf, hide_index=True, use_container_width=True, key="rb",
                column_config={"Asset": st.column_config.TextColumn(disabled=True)},
            )
            kw["budgets"] = np.array(edited["Risk budget"], dtype=float)

        if method == opt.BLACK_LITTERMAN:
            c1, c2, c3 = st.columns(3)
            proxy = c1.selectbox("Market proxy", ["Equal weight", "Inverse volatility"])
            kw["risk_aversion"] = c2.slider("Risk aversion δ", 0.5, 6.0, 2.5, 0.5)
            kw["tau"] = c3.slider("Prior uncertainty τ", 0.01, 0.20, 0.05, 0.01)
            kw["proxy"] = proxy
            kw["base"] = st.radio(
                "Optimize posterior with", [opt.MAX_SHARPE, opt.MIN_VOL],
                horizontal=True, key="bl_base",
            )
            st.caption("Enter an annual expected-return **view** for any assets you "
                       "have an opinion on; leave blank to defer to equilibrium.")
            vdf = pd.DataFrame({"Asset": tickers, "View (annual return)": [np.nan] * len(tickers)})
            kw["views_df"] = st.data_editor(
                vdf, hide_index=True, use_container_width=True, key="bl_views",
                column_config={"Asset": st.column_config.TextColumn(disabled=True)},
            )

        if method == opt.ENTROPY_POOLING:
            kw["base"] = st.radio(
                "Optimize posterior with", [opt.MAX_SHARPE, opt.MIN_VOL],
                horizontal=True, key="ep_base",
            )
            st.caption("Impose an annual **mean view** on any assets; scenarios are "
                       "reweighted by minimum relative entropy to satisfy them.")
            vdf = pd.DataFrame({"Asset": tickers, "Mean view (annual)": [np.nan] * len(tickers)})
            kw["views_df"] = st.data_editor(
                vdf, hide_index=True, use_container_width=True, key="ep_views",
                column_config={"Asset": st.column_config.TextColumn(disabled=True)},
            )
    return kw


def run_optimizer(method, tickers, mu, cov, rets, ppy, rf, kw):
    """Returns (weights, mu_eff, cov_eff, note)."""
    note = ""
    mu_eff, cov_eff = mu, cov
    wb = kw.get("weight_bounds", (0.0, 1.0))

    if method == opt.BLACK_LITTERMAN:
        proxy_w = (opt.inverse_volatility(cov) if kw["proxy"] == "Inverse volatility"
                   else opt.equal_weight(tickers))
        views = []
        for _, r in kw["views_df"].iterrows():
            v = r["View (annual return)"]
            if pd.notna(v):
                views.append({"kind": "absolute", "asset": r["Asset"], "value": float(v)})
        P, Q = opt.build_view_matrices(tickers, views)
        mu_eff, cov_eff = opt.black_litterman_posterior(
            cov, proxy_w, kw["risk_aversion"], kw["tau"], P, Q)
        w = opt.optimize(kw["base"], mu=mu_eff, cov=cov_eff, rf=rf, weight_bounds=wb)
        note = f"{len(views)} view(s) blended with the equilibrium prior."
        return w, mu_eff, cov_eff, note

    if method == opt.ENTROPY_POOLING:
        mean_views = {}
        for _, r in kw["views_df"].iterrows():
            v = r["Mean view (annual)"]
            if pd.notna(v):
                mean_views[r["Asset"]] = float(v)
        mu_eff, cov_eff, q = opt.entropy_pooling_posterior(rets, mean_views, ppy)
        w = opt.optimize(kw["base"], mu=mu_eff, cov=cov_eff, rf=rf, weight_bounds=wb)
        ess = 1.0 / float(np.sum(q ** 2))
        note = (f"{len(mean_views)} mean view(s); effective sample size "
                f"{ess:.0f} of {len(rets)} scenarios.")
        return w, mu_eff, cov_eff, note

    w = opt.optimize(
        method, mu=mu, cov=cov, rf=rf,
        weight_bounds=wb, target=kw.get("target"),
        budgets=kw.get("budgets"), kappa=kw.get("kappa", 1.0),
        n_obs=kw.get("n_obs", len(rets)),
    )
    return w, mu_eff, cov_eff, note


def section_optimizer(cfg: dict):
    page_header(
        "🧮 Portfolio Optimizer",
        "Select assets, estimate return &amp; risk from history, and construct a "
        "portfolio with any of ten institutional methods.",
    )

    tickers = asset_picker("opt")
    if len(tickers) < 2:
        st.info("Select at least two assets to optimize.")
        return

    prices, warns = load_prices_cached(tuple(tickers), cfg["start"], cfg["end"], cfg["source"], cfg["convert_usd"])
    for wmsg in warns:
        st.warning(wmsg)
    if prices.empty or prices.shape[1] < 2:
        st.error("Could not load enough price data. Try a different range or source.")
        return
    tickers = list(prices.columns)
    rets, mu, cov, ppy = estimate_moments(prices, cfg)

    method = st.selectbox("Construction method", opt.ALL_METHODS, index=0, key="opt_method")
    kw = method_controls(method, tickers, mu, rets, ppy)

    try:
        weights, mu_eff, cov_eff, note = run_optimizer(
            method, tickers, mu, cov, rets, ppy, cfg["rf"], kw)
    except Exception as e:  # surface solver/feasibility problems cleanly
        st.error(f"Optimization failed: {e}")
        return
    if note:
        st.caption("ℹ️ " + note)

    # Headline portfolio stats (on the *estimated* moments).
    p_ret = opt.portfolio_return(weights.values, mu.reindex(weights.index).values)
    p_vol = opt.portfolio_vol(weights.values, cov.loc[weights.index, weights.index].values)
    p_sharpe = (p_ret - cfg["rf"]) / p_vol if p_vol > 0 else 0.0
    c = st.columns(4)
    c[0].metric("Expected return", fmt_pct(p_ret))
    c[1].metric("Volatility", fmt_pct(p_vol))
    c[2].metric("Sharpe (ex-ante)", f"{p_sharpe:.2f}")
    c[3].metric("Effective # assets", f"{1/np.sum(weights.values**2):.1f}")

    left, right = st.columns([1.15, 1])
    with left:
        st.plotly_chart(weights_bar(weights), use_container_width=True)
        st.plotly_chart(risk_contrib_bar(weights, cov), use_container_width=True)
    with right:
        st.plotly_chart(class_donut(weights), use_container_width=True)
        wt_table = pd.DataFrame({
            "Asset class": [uni.asset_class_of(t) for t in weights.index],
            "Weight": weights.values,
        }, index=weights.index)
        wt_table = wt_table[wt_table["Weight"].abs() > 1e-6].sort_values("Weight", ascending=False)
        st.dataframe(
            wt_table.style.format({"Weight": "{:.2%}"}),
            use_container_width=True, height=320,
        )
        st.download_button(
            "⬇️ Download weights (CSV)",
            weights.rename("weight").to_csv().encode(),
            file_name=f"weights_{method.replace(' ', '_').lower()}.csv",
        )

    st.plotly_chart(frontier_fig(mu, cov, weights, cfg["rf"]), use_container_width=True)

    # Hand-off to the backtester.
    if st.button("📥 Load these weights into the Backtester", type="primary"):
        st.session_state["seed_weights"] = weights.to_dict()
        st.session_state["seed_source"] = method
        st.success("Loaded. Open the **Backtester** workbench to test them.")

    with st.expander("🧪 Quick historical backtest of these weights", expanded=False):
        rb = st.selectbox("Rebalance", list(bt.REBALANCE_RULES.keys()), index=0, key="optbt_rb")
        cost = st.slider("Transaction cost (bps per side)", 0, 50, 5, 1, key="optbt_cost")
        res = bt.backtest_fixed_weights(prices, weights, rebalance=rb, cost_bps=cost)
        stats = metrics.summary_stats(res["returns"], cfg["rf"], ppy)
        metrics_row(stats, ["CAGR", "Volatility (ann.)", "Sharpe", "Max Drawdown", "Calmar"])
        st.plotly_chart(equity_fig({"Portfolio": res["equity"]}), use_container_width=True)


# --------------------------------------------------------------------------- #
# Section: Backtester
# --------------------------------------------------------------------------- #
def section_backtester(cfg: dict):
    page_header(
        "🧪 Backtester",
        "Stress-test any allocation with periodic rebalancing and trading costs, or "
        "run a walk-forward out-of-sample optimization.",
    )

    mode = st.radio(
        "Mode", ["Fixed weights", "Walk-forward optimization"], horizontal=True
    )
    seed = st.session_state.get("seed_weights")
    default_tickers = list(seed.keys()) if seed else uni.default_selection()
    tickers = asset_picker("bt", default=default_tickers)
    if len(tickers) < 2:
        st.info("Select at least two assets.")
        return

    prices, warns = load_prices_cached(tuple(tickers), cfg["start"], cfg["end"], cfg["source"], cfg["convert_usd"])
    for wmsg in warns:
        st.warning(wmsg)
    if prices.empty or prices.shape[1] < 2:
        st.error("Could not load enough price data.")
        return
    tickers = list(prices.columns)
    rets, mu, cov, ppy = estimate_moments(prices, cfg)

    cset = st.columns(4)
    rebal = cset[0].selectbox("Rebalance", list(bt.REBALANCE_RULES.keys()), index=0)
    cost = cset[1].slider("Cost (bps/side)", 0, 50, 5, 1)
    initial = cset[2].number_input("Initial capital", 1000, 10_000_000, 10_000, 1000)
    bench_choice = cset[3].selectbox(
        "Benchmark", ["60/40 (SPY/AGG)", "SPY only", "Equal weight", "None"]
    )

    if mode == "Fixed weights":
        if seed:
            st.caption(f"Pre-filled from optimizer ({st.session_state.get('seed_source','')}).")
        init_w = [float(seed.get(t, 0.0)) if seed else round(1 / len(tickers), 4) for t in tickers]
        wdf = pd.DataFrame({"Asset": tickers, "Weight": init_w})
        edited = st.data_editor(
            wdf, hide_index=True, use_container_width=True, key="bt_w",
            column_config={
                "Asset": st.column_config.TextColumn(disabled=True),
                "Weight": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        w = pd.Series(np.array(edited["Weight"], float), index=edited["Asset"])
        if w.sum() <= 0:
            st.error("Weights must sum to a positive number.")
            return
        w = w / w.sum()
        st.caption(f"Normalized weights sum to {w.sum():.2f}.")
        result = bt.backtest_fixed_weights(prices, w, rebalance=rebal, cost_bps=cost, initial=initial)
        weights_used = w
    else:
        oc = st.columns(3)
        wf_method = oc[0].selectbox(
            "Re-optimize each rebalance with",
            [opt.MAX_SHARPE, opt.MIN_VOL, opt.ERC, opt.RISK_BUDGET,
             opt.INVERSE_VOL, opt.ROBUST, opt.EQUAL_WEIGHT],
        )
        lookback_m = oc[1].slider("Look-back window (months)", 6, 60, 24, 3)
        wf_rebal = oc[2].selectbox(
            "Re-optimization frequency", ["Quarterly", "Monthly", "Semi-Annual", "Annual"], index=0
        )

        def weight_fn(window: pd.DataFrame) -> pd.Series:
            r = est.to_returns(window)
            ppy_w = est.infer_periods_per_year(r.index)
            mu_w = est.mean_returns(r, cfg["ret_method"], cfg["span"], ppy_w)
            cov_w = est.cov_matrix(r, cfg["cov_method"], cfg["span"], ppy_w)
            return opt.optimize(
                wf_method, mu=mu_w, cov=cov_w, rf=cfg["rf"], n_obs=len(r)
            )

        with st.spinner("Running walk-forward backtest…"):
            result = bt.backtest_walk_forward(
                prices, weight_fn, rebalance=wf_rebal,
                lookback_days=int(lookback_m * 21), cost_bps=cost, initial=initial,
            )
        weights_used = result["weights"].iloc[-1].dropna()
        st.caption(f"{result['n_rebalances']} out-of-sample re-optimizations, "
                   f"{lookback_m}-month look-back. No look-ahead bias.")

    port_rets = result["returns"]
    if port_rets.empty:
        st.error("Backtest produced no returns — widen the date range.")
        return

    # Benchmark
    bench_map = {
        "60/40 (SPY/AGG)": {"SPY": 0.6, "AGG": 0.4},
        "SPY only": {"SPY": 1.0},
        "Equal weight": {t: 1 / len(tickers) for t in tickers},
    }
    curves = {"Portfolio": result["equity"]}
    bench_rets = None
    if bench_choice != "None":
        bw = bench_map[bench_choice]
        need = [t for t in bw if t not in prices.columns]
        bprices = prices
        if need:
            bprices, _ = load_prices_cached(
                tuple(sorted(set(list(prices.columns) + list(bw.keys())))),
                cfg["start"], cfg["end"], cfg["source"], cfg["convert_usd"])
        bench_rets = bt.benchmark_returns(bprices, bw, rebalance=rebal if mode == "Fixed weights" else "Monthly")
        if bench_rets is not None:
            bench_rets = bench_rets.reindex(port_rets.index).dropna()
            curves["Benchmark"] = initial * (1 + bench_rets).cumprod()

    # Headline stats
    stats = metrics.summary_stats(port_rets, cfg["rf"], ppy)
    st.subheader("Performance")
    metrics_row(stats, ["Total Return", "CAGR", "Volatility (ann.)", "Sharpe", "Sortino"])
    metrics_row(stats, ["Max Drawdown", "Calmar", "VaR 95% (period)", "Win Rate", "Worst Period"])
    st.caption(f"Turnover (sum of |Δw|): {result['turnover']:.2f}× · "
               f"Total cost drag: {fmt_pct(result['total_cost_fraction'],2)} · "
               f"Rebalances: {result['n_rebalances']}")

    st.plotly_chart(equity_fig(curves), use_container_width=True)
    cc = st.columns([1, 1])
    with cc[0]:
        st.plotly_chart(drawdown_fig(port_rets), use_container_width=True)
    with cc[1]:
        st.plotly_chart(weight_area_fig(result["weights"]), use_container_width=True)

    # Portfolio vs benchmark comparison table
    if bench_rets is not None and len(bench_rets):
        comp = pd.DataFrame({
            "Portfolio": pd.Series(metrics.summary_stats(port_rets, cfg["rf"], ppy)),
            "Benchmark": pd.Series(metrics.summary_stats(bench_rets, cfg["rf"], ppy)),
        })
        st.dataframe(_fmt_stats_table(comp), use_container_width=True)

    st.download_button(
        "⬇️ Download daily returns (CSV)",
        port_rets.rename("return").to_csv().encode(),
        file_name="backtest_returns.csv",
    )


def _fmt_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    pct_rows = {
        "Total Return", "CAGR", "Volatility (ann.)", "Max Drawdown",
        "VaR 95% (period)", "CVaR 95% (period)", "Best Period", "Worst Period", "Win Rate",
    }
    # Build a fresh object-dtype frame rather than writing strings into the
    # original float columns — pandas 3.0 rejects that in-place dtype change.
    formatted = {}
    for r in df.index:
        fmt = (lambda x: fmt_pct(x, 2)) if r in pct_rows else (lambda x: f"{x:.2f}")
        formatted[r] = df.loc[r].map(fmt)
    return pd.DataFrame(formatted).T.reindex(df.index)


# --------------------------------------------------------------------------- #
# Section: Monte Carlo
# --------------------------------------------------------------------------- #
def mc_fan_fig(result: dict, ppy: float) -> go.Figure:
    """Fan chart: nested 5–95% and 25–75% bands around the median path."""
    vp = result["value_paths"]
    bands = mc.percentile_bands(vp)
    x = np.arange(vp.shape[1]) / ppy  # horizon in years
    navy = "27,42,65"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=bands[95], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=bands[5], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=f"rgba({navy},0.08)",
                             name="5–95% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=bands[75], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=bands[25], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=f"rgba({navy},0.18)",
                             name="25–75% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=bands[50], mode="lines", name="Median path",
                             line=dict(color="#B6924C", width=2.6)))
    fig.add_hline(y=result["initial"], line=dict(color="#9E3B34", width=1, dash="dot"),
                  annotation_text="Start", annotation_position="bottom left")
    fig.update_layout(
        title="Projected portfolio value — simulated paths", height=440,
        xaxis_title="Years ahead", yaxis_title="Portfolio value",
        margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2),
    )
    return fig


def mc_terminal_fig(result: dict) -> go.Figure:
    term = result["terminal"]
    init = result["initial"]
    fig = go.Figure(go.Histogram(
        x=term, nbinsx=60, marker_color="#34557F",
        marker_line_color="rgba(255,255,255,0.4)", marker_line_width=0.5,
        opacity=0.85, name="Terminal value",
    ))
    fig.add_vline(x=init, line=dict(color="#9E3B34", width=1.5, dash="dot"),
                  annotation_text="Break-even", annotation_position="top")
    fig.add_vline(x=float(np.median(term)), line=dict(color="#B6924C", width=2),
                  annotation_text="Median", annotation_position="top right")
    fig.update_layout(
        title="Distribution of terminal value", height=320,
        xaxis_title="Portfolio value at horizon", yaxis_title="Frequency",
        margin=dict(l=10, r=10, t=50, b=10), bargap=0.02,
    )
    return fig


def _fmt_risk_table(table: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for col in table.columns:
        out[col] = table[col].map(lambda x: "—" if pd.isna(x) else f"{x * 100:.2f}%")
    return pd.DataFrame(out, index=table.index)


def evt_tail_fig(curve: pd.DataFrame, fit: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve["loss"] * 100, y=curve["empirical"], mode="markers",
        name="Empirical", marker=dict(color="#34557F", size=6),
    ))
    fig.add_trace(go.Scatter(
        x=curve["loss"] * 100, y=curve["gpd_fit"], mode="lines",
        name="Fitted GPD tail", line=dict(color="#B6924C", width=2.4),
    ))
    fig.add_vline(x=fit["u"] * 100, line=dict(color="#9E3B34", width=1, dash="dot"),
                  annotation_text="Threshold u", annotation_position="top")
    fig.update_layout(
        title="Loss-tail fit — Extreme Value Theory (peaks over threshold)",
        height=340, xaxis_title="Daily loss (%)", yaxis_title="P(loss &gt; x)",
        yaxis_type="log", margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def render_risk_tab(pr: pd.Series, ppy: float):
    st.markdown(
        '<div class="pl-sub">One-period (daily) loss risk of the portfolio, estimated '
        "four ways. The normal model usually understates deep losses — compare it with "
        "the fat-tail estimators.</div>",
        unsafe_allow_html=True,
    )
    table, fit = risklib.var_comparison_table(pr, levels=(0.95, 0.99), threshold_q=0.95)
    st.dataframe(_fmt_risk_table(table), use_container_width=True)
    st.caption(
        "VaR = loss not exceeded with the stated confidence; CVaR / Expected Shortfall "
        "= average loss *when* VaR is breached. Cornish-Fisher adjusts the normal "
        "quantile for skew &amp; kurtosis; EVT fits the tail itself."
    )

    st.markdown("##### Extreme Value Theory — tail fit")
    if fit.get("ok"):
        xi = fit["xi"]
        ec = st.columns(4)
        ec[0].metric("Tail index ξ (shape)", f"{xi:.3f}")
        ec[1].metric("Scale β", f"{fit['beta']*100:.3f}%")
        ec[2].metric("Threshold u", fmt_pct(fit["u"], 2))
        ec[3].metric("Tail exceedances", f"{fit['n_u']} / {fit['n']}")
        st.caption(
            ("🔴 **ξ > 0 → heavy (fat) tails**: extreme losses are far more likely than "
             "a normal model implies — VaR/CVaR should use the EVT row."
             if xi > 0.02 else
             "🟢 **ξ ≤ 0 → bounded / thin tail** over this window: losses fall off "
             "quickly. With longer history spanning a crisis, ξ typically turns positive.")
        )
        curve = risklib.evt_tail_curve(pr, fit)
        if curve is not None:
            st.plotly_chart(evt_tail_fig(curve, fit), use_container_width=True)
    else:
        st.info("Not enough tail observations to fit the EVT model — widen the date range.")


def stress_bar(values: pd.Series, title: str, pct=True) -> go.Figure:
    colors = ["#2E7D6F" if v >= 0 else "#9E3B34" for v in values.values]
    fig = go.Figure(go.Bar(
        x=values.values, y=list(values.index), orientation="h", marker_color=colors,
        text=[f"{v*100:+.1f}%" if pct else f"{v:+.2f}" for v in values.values],
        textposition="outside",
    ))
    fig.update_layout(
        title=title, xaxis_tickformat=".0%", height=max(240, 40 * len(values) + 80),
        margin=dict(l=10, r=30, t=50, b=10),
    )
    return fig


def render_stress_tab(prices: pd.DataFrame, weights: pd.Series):
    st.markdown("##### Historical crisis replay")
    st.markdown(
        '<div class="pl-sub">How this exact allocation would have moved through real '
        "crisis windows, using the assets' actual returns.</div>",
        unsafe_allow_html=True,
    )
    hs, skipped = risklib.historical_stress(prices, weights)
    if len(hs):
        ser = pd.Series(hs["Portfolio"].values, index=hs["Scenario"].values).iloc[::-1]
        st.plotly_chart(stress_bar(ser, "Portfolio return over each crisis"),
                        use_container_width=True)
        disp = hs.copy()
        disp["Portfolio"] = disp["Portfolio"].map(lambda x: fmt_pct(x, 1))
        st.dataframe(disp, hide_index=True, use_container_width=True)
    else:
        st.info("None of the crisis windows fall fully inside your data range.")
    if skipped:
        st.caption(
            "Out of range (widen the **Start** date and use long-history assets like "
            f"SPY, GLD, AGG, TLT to replay these): {', '.join(skipped)}."
        )

    st.divider()
    st.markdown("##### Hypothetical factor shocks")
    st.markdown(
        '<div class="pl-sub">Instantaneous, forward-looking shocks applied by asset '
        "class — no history required.</div>",
        unsafe_allow_html=True,
    )
    impacts = {name: risklib.apply_factor_shock(weights, sh)[0]
               for name, sh in risklib.FACTOR_SCENARIOS.items()}
    iser = pd.Series(impacts).sort_values()
    st.plotly_chart(stress_bar(iser, "Portfolio P&L under preset shocks"),
                    use_container_width=True)

    pick = st.selectbox("Break down a scenario", list(risklib.FACTOR_SCENARIOS.keys()))
    total, contrib = risklib.apply_factor_shock(weights, risklib.FACTOR_SCENARIOS[pick])
    cc = st.columns([1, 2])
    cc[0].metric("Portfolio impact", fmt_pct(total, 1))
    contrib = contrib[contrib.abs() > 1e-9].sort_values()
    cc[1].plotly_chart(stress_bar(contrib, "Contribution by asset"),
                       use_container_width=True)

    with st.expander("🎛️ Build a custom shock"):
        sc = st.columns(3)
        eq = sc[0].slider("Equity shock", -0.6, 0.4, -0.20, 0.05, format="%.2f")
        fi = sc[1].slider("Fixed Income shock", -0.3, 0.3, 0.02, 0.01, format="%.2f")
        al = sc[2].slider("Alternatives shock", -0.5, 0.5, -0.05, 0.05, format="%.2f")
        custom = {uni.EQUITY: eq, uni.FIXED_INCOME: fi, uni.ALTERNATIVES: al}
        ctot, _ = risklib.apply_factor_shock(weights, custom)
        st.metric("Custom-shock portfolio impact", fmt_pct(ctot, 1))


def section_montecarlo(cfg: dict):
    page_header(
        "🎲 Monte Carlo Simulation",
        "Project the portfolio forward with thousands of random scenarios. Asset "
        "co-movement is preserved by <b>Cholesky decomposition</b> of the historical "
        "covariance, so the simulated correlations match the real ones.",
    )

    tickers = asset_picker("mc")
    if len(tickers) < 2:
        st.info("Select at least two assets to simulate.")
        return
    prices, warns = load_prices_cached(tuple(tickers), cfg["start"], cfg["end"],
                                       cfg["source"], cfg["convert_usd"])
    for wmsg in warns:
        st.warning(wmsg)
    if prices.empty or prices.shape[1] < 2:
        st.error("Could not load enough price data.")
        return
    tickers = list(prices.columns)
    rets, mu, cov, ppy = estimate_moments(prices, cfg)

    # ---- Portfolio weights to simulate -----------------------------------
    method_map = {
        "Equal Weight": opt.EQUAL_WEIGHT, "Max Sharpe": opt.MAX_SHARPE,
        "Min Volatility": opt.MIN_VOL, "Equal Risk Contribution": opt.ERC,
        "Inverse Volatility": opt.INVERSE_VOL,
    }
    wcol, scol = st.columns([1, 1])
    weight_choice = wcol.selectbox(
        "Portfolio to simulate", list(method_map.keys()) + ["Manual weights"], index=1
    )
    if weight_choice == "Manual weights":
        wdf = pd.DataFrame({"Asset": tickers, "Weight": [round(1 / len(tickers), 4)] * len(tickers)})
        edited = scol.data_editor(
            wdf, hide_index=True, use_container_width=True, key="mc_w",
            column_config={"Asset": st.column_config.TextColumn(disabled=True),
                           "Weight": st.column_config.NumberColumn(format="%.4f")},
        )
        weights = pd.Series(np.array(edited["Weight"], float), index=edited["Asset"])
        weights = weights / weights.sum() if weights.sum() > 0 else weights
    else:
        weights = opt.optimize(method_map[weight_choice], mu=mu, cov=cov, rf=cfg["rf"])
        scol.plotly_chart(weights_bar(weights, "Weights being simulated"),
                          use_container_width=True)

    # ---- Simulation controls ---------------------------------------------
    st.markdown("##### Simulation settings")
    c = st.columns(4)
    horizon_m = c[0].slider("Horizon (months)", 1, 60, 12, 1)
    n_sims = c[1].select_slider("Scenarios", [500, 1000, 2000, 5000, 10000], value=2000)
    initial = c[2].number_input("Initial capital", 1000, 100_000_000, 10_000, 1000)
    dist = c[3].selectbox("Shock distribution", ["Normal", "Student-t (fat tails)"])
    c2 = st.columns(4)
    df_t = 5
    if dist.startswith("Student"):
        df_t = c2[0].slider("t degrees of freedom", 3, 30, 5, 1,
                            help="Lower = fatter tails / more extreme moves.")
    haircut = c2[1].slider(
        "Drift haircut → risk-free", 0.0, 1.0, 0.0, 0.1,
        help="Historical average returns are noisy. Shrink the expected-return "
             "drift toward the risk-free rate (1.0 = pure risk-free drift).",
    )
    target = c2[2].number_input(
        "Goal return over horizon", -0.5, 5.0, 0.10, 0.05, format="%.2f",
        help="Total return over the whole horizon; used to report the probability "
             "of reaching it.")
    seed = c2[3].number_input("Random seed", 0, 10_000, 42, 1)

    if not st.button("▶ Run simulation", type="primary", key="mc_run"):
        st.caption("Set your portfolio and settings, then run the simulation.")
        return

    # Shrink drift toward the risk-free rate if requested.
    mu_sim = cfg["rf"] + (1.0 - haircut) * (mu - cfg["rf"])
    horizon_steps = max(1, int(round(horizon_m / 12.0 * ppy)))
    dist_key = "Student-t" if dist.startswith("Student") else "Normal"

    with st.spinner(f"Simulating {n_sims:,} scenarios over {horizon_m} months…"):
        result = mc.simulate_portfolio(
            mu_sim, cov, weights, horizon_steps, int(n_sims), ppy=ppy,
            initial=float(initial), dist=dist_key, df=int(df_t), seed=int(seed),
        )
    stats = mc.terminal_stats(result, target_total_return=float(target))
    dd = mc.path_max_drawdowns(result["value_paths"])
    pr_hist = risklib.portfolio_returns(rets, weights)  # historical return series

    proj_tab, risk_tab, stress_tab = st.tabs(
        ["📈 Projection", "📉 VaR · CVaR · EVT", "🔥 Stress tests"]
    )

    # ---- Tab 1: Monte Carlo projection -----------------------------------
    with proj_tab:
        m = st.columns(4)
        m[0].metric("Median outcome", f"${stats['Median terminal']:,.0f}",
                    fmt_pct(stats["Median return"]))
        m[1].metric("90% range",
                    f"${stats['p5 terminal']:,.0f} – ${stats['p95 terminal']:,.0f}")
        m[2].metric("Probability of loss", fmt_pct(stats["Prob. of loss"]))
        m[3].metric(f"Prob. ≥ {fmt_pct(target,0)}", fmt_pct(stats["Prob. ≥ target"]))
        m2 = st.columns(4)
        m2[0].metric("Terminal VaR 95%", fmt_pct(stats["VaR 95% (terminal)"]))
        m2[1].metric("Terminal CVaR 95%", fmt_pct(stats["CVaR 95% (terminal)"]))
        m2[2].metric("Median worst drawdown", fmt_pct(float(np.median(dd))))
        m2[3].metric("Worst-case (p5) drawdown", fmt_pct(float(np.percentile(dd, 5))))

        st.plotly_chart(mc_fan_fig(result, ppy), use_container_width=True)
        st.plotly_chart(mc_terminal_fig(result), use_container_width=True)

        with st.expander("🔗 Correlation preservation (Cholesky check)", expanded=False):
            st.markdown(
                "Shocks are generated as **L · z**, where **L** is the Cholesky factor "
                "of the covariance (Σ = L · Lᵀ) and z are independent standard normals. "
                "This rotates uncorrelated noise into the historical correlation "
                "structure."
            )
            st.caption(
                f"Largest gap between simulated and historical correlation: "
                f"**{result['corr_err']:.3f}** (≈0 means the structure is preserved)."
            )
            corr = result["corr_input"]
            hm = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                           zmin=-1, zmax=1, aspect="auto", title="Correlation matrix used")
            hm.update_layout(height=440, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(hm, use_container_width=True)

    # ---- Tab 2: VaR / CVaR / EVT -----------------------------------------
    with risk_tab:
        render_risk_tab(pr_hist, ppy)

    # ---- Tab 3: Stress tests ---------------------------------------------
    with stress_tab:
        render_stress_tab(prices, weights)

    st.caption(
        "⚠️ Projections and risk estimates assume returns and correlations stay like "
        "the chosen history. Expected returns are especially uncertain — use the drift "
        "haircut to stay conservative. Educational, not investment advice."
    )


# --------------------------------------------------------------------------- #
# Section: Macro Impulse Response
# --------------------------------------------------------------------------- #
def macro_irf_fig(cum: pd.DataFrame, freq_label: str) -> go.Figure:
    """Cumulative portfolio response (%) to a +1-SD shock in each factor."""
    fig = go.Figure()
    palette = ["#1B2A41", "#B6924C", "#2E7D6F", "#34557F", "#9E3B34", "#9A6A4B"]
    x = list(range(len(cum)))
    for col, color in zip(cum.columns, palette):
        fig.add_trace(go.Scatter(
            x=x, y=cum[col].values * 100, mode="lines", name=col,
            line=dict(width=2.4, color=color),
        ))
    fig.add_hline(y=0, line=dict(color="rgba(22,38,58,0.25)", width=1))
    fig.update_layout(
        title="Cumulative portfolio response to a +1σ macro shock", height=420,
        xaxis_title=f"{freq_label} ahead", yaxis_title="Cumulative response (%)",
        margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.22),
    )
    return fig


def macro_factor_band_fig(cum: pd.Series, se, factor: str, freq_label: str) -> go.Figure:
    x = list(range(len(cum)))
    y = cum.values * 100
    fig = go.Figure()
    if se is not None:
        hi, lo = (y + 1.96 * se.values * 100), (y - 1.96 * se.values * 100)
        fig.add_trace(go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(27,42,65,0.12)",
                                 name="95% band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=factor,
                             line=dict(color="#B6924C", width=2.6)))
    fig.add_hline(y=0, line=dict(color="rgba(22,38,58,0.25)", width=1))
    fig.update_layout(
        title=f"Response to a +1σ shock in {factor}", height=340,
        xaxis_title=f"{freq_label} ahead", yaxis_title="Cumulative response (%)",
        margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.25),
    )
    return fig


def macro_bar(series: pd.Series, title: str, pct=True) -> go.Figure:
    s = series.sort_values()
    colors = ["#2E7D6F" if v >= 0 else "#9E3B34" for v in s.values]
    fig = go.Figure(go.Bar(
        x=s.values, y=list(s.index), orientation="h", marker_color=colors,
        text=[f"{v*100:+.2f}%" if pct else f"{v*100:.1f}%" for v in s.values],
        textposition="outside",
    ))
    fig.update_layout(title=title, xaxis_tickformat=".0%",
                      height=max(240, 42 * len(s) + 80), margin=dict(l=10, r=30, t=50, b=10))
    return fig


def section_macro(cfg: dict):
    page_header(
        "🌐 Macro Impulse Response",
        "Model the portfolio jointly with macro factors in a <b>vector "
        "autoregression</b> and trace how it reacts, period by period, to a shock "
        "in each factor — <b>orthogonalized impulse responses</b>, identified by a "
        "Cholesky ordering of the residual covariance.",
    )

    tickers = asset_picker("macro")
    if len(tickers) < 2:
        st.info("Select at least two assets.")
        return
    prices, warns = load_prices_cached(tuple(tickers), cfg["start"], cfg["end"],
                                       cfg["source"], cfg["convert_usd"])
    for wmsg in warns:
        st.warning(wmsg)
    if prices.empty or prices.shape[1] < 2:
        st.error("Could not load enough price data.")
        return
    tickers = list(prices.columns)
    rets, mu, cov, ppy = estimate_moments(prices, cfg)

    wcol, ccol = st.columns([1, 1])
    weights = weights_selector(tickers, mu, cov, cfg["rf"], "macro", container=wcol)
    ccol.caption(
        "**Macro factors** (Yahoo proxies): S&P 500, Δ10-year yield, US-dollar index, "
        "WTI crude, Δ VIX, and gold. Returns for price factors, differences for "
        "yields/VIX."
    )

    st.markdown("##### Model settings")
    s = st.columns(4)
    freq = s[0].selectbox("Frequency", ["Weekly", "Monthly"], index=0)
    freq_label = "Weeks" if freq == "Weekly" else "Months"
    horizon = s[1].slider(f"Horizon ({freq_label.lower()})", 4, 26, 12, 1)
    lag_mode = s[2].selectbox("Lag order", ["Auto (AIC)", "Manual"])
    lag = None if lag_mode.startswith("Auto") else s[3].slider("Lags", 1, 8, 2, 1)

    if not st.button("▶ Run impulse-response analysis", type="primary", key="macro_run"):
        st.caption("Pick a portfolio and settings, then run the analysis.")
        return

    factor_daily, fwarn = load_macro_cached(cfg["start"], cfg["end"], cfg["source"])
    if factor_daily.empty:
        st.error("Could not load macro factor data.")
        return
    port_daily = risklib.portfolio_returns(rets, weights)
    frame = macrolib.build_frame(port_daily, factor_daily, freq)
    if len(frame) < 8 * frame.shape[1]:
        st.warning(
            f"Only {len(frame)} {freq.lower()} observations — the VAR may be "
            "unstable. Use a longer history, weekly frequency, or fewer assets."
        )
    try:
        with st.spinner("Fitting VAR and computing impulse responses…"):
            out = macrolib.run_var_irf(frame, horizon=horizon, maxlags=8, lag=lag)
            impact, r2 = macrolib.static_sensitivities(frame)
    except Exception as e:
        st.error(f"VAR estimation failed: {e}. Try a longer history or fewer assets.")
        return

    st.caption(
        f"VAR({out['lag']}) on {out['nobs']} {freq.lower()} observations · "
        f"{len(out['factor_cols'])} macro factors · Cholesky-ordered shocks."
    )

    cum = out["cum_irf"]
    final = cum.iloc[-1]
    st.plotly_chart(macro_irf_fig(cum, freq_label), use_container_width=True)

    cc = st.columns(2)
    cc[0].plotly_chart(
        macro_bar(final, f"Cumulative impact after {horizon} {freq_label.lower()}"),
        use_container_width=True)
    if out["fevd"] is not None:
        fevd = out["fevd"].rename(index={out["port_name"]: "Own / idiosyncratic"})
        cc[1].plotly_chart(
            macro_bar(fevd, f"Variance share explained at {horizon} {freq_label.lower()}",
                      pct=False),
            use_container_width=True)

    st.markdown("##### Drill into one factor")
    pick = st.selectbox("Macro factor", list(cum.columns))
    se_col = out["cum_se"][pick] if out["cum_se"] is not None else None
    st.plotly_chart(macro_factor_band_fig(cum[pick], se_col, pick, freq_label),
                    use_container_width=True)

    with st.expander("📐 Static factor sensitivities (contemporaneous)"):
        st.caption(f"Portfolio response to a +1-SD move in each factor · regression "
                   f"R² = {r2:.2f}.")
        tbl = pd.DataFrame({"1σ impact": impact.map(lambda x: fmt_pct(x, 2))})
        st.dataframe(tbl, use_container_width=True)

    with st.expander("ℹ️ How to read this"):
        st.markdown(
            "- A **vector autoregression (VAR)** lets every variable depend on recent "
            "lags of itself and all the others.\n"
            "- An **orthogonalized impulse response** traces the portfolio's reaction "
            "to a one-standard-deviation shock in a factor, holding the others fixed at "
            "impact. Shocks are disentangled with a **Cholesky** factorization of the "
            "residual covariance (factors ordered before the portfolio).\n"
            "- **FEVD** (variance decomposition) splits the portfolio's forecast "
            "uncertainty at the horizon among the factors and its own shock.\n"
            "- These are statistical, in-sample relationships on factor *proxies* — "
            "association, not guaranteed causation, and ordering matters."
        )


# --------------------------------------------------------------------------- #
# Section: Asset Universe
# --------------------------------------------------------------------------- #
def section_universe(cfg: dict):
    page_header(
        "📚 Asset Universe",
        "A broad ETF universe across three asset classes — global, regional, and "
        "US-sector equity, a full fixed-income ladder, and alternatives. Explore "
        "historical behavior and co-movement.",
    )
    st.caption(
        f"{len(uni.ASSETS)} ETFs total. Non-USD listings (EUR/JPY/GBp) are "
        "converted to USD at load time when the sidebar toggle is on. "
        "**Click any ticker** to open its Yahoo Finance page (holdings, performance, "
        "profile) before investing."
    )
    link_cfg = {
        "Ticker (Yahoo)": st.column_config.LinkColumn(
            "Ticker (Yahoo)", display_text=r"quote/(.+)$",
            help="Opens the fund's Yahoo Finance page in a new tab.",
        )
    }
    for ac in uni.ASSET_CLASSES:
        members = uni.assets_in_class(ac)
        st.markdown(
            f'<span class="pl-tag" style="background:{CLASS_COLORS[ac]}">{ac} '
            f"· {len(members)}</span>",
            unsafe_allow_html=True,
        )
        rows = [{
            "Ticker (Yahoo)": yahoo_url(a.ticker), "Name": a.name, "Region": a.region,
            "Ccy": a.currency, "Note": a.note, "Bloomberg": a.bb or "—",
        } for a in members]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config=link_cfg)

    st.divider()
    st.subheader("Explore history")
    tickers = asset_picker("uni")
    if len(tickers) < 1:
        return
    prices, warns = load_prices_cached(tuple(tickers), cfg["start"], cfg["end"], cfg["source"], cfg["convert_usd"])
    for wmsg in warns:
        st.warning(wmsg)
    if prices.empty:
        st.error("No data loaded.")
        return

    norm = prices / prices.iloc[0]
    fig = go.Figure()
    for t in norm.columns:
        fig.add_trace(go.Scatter(x=norm.index, y=norm[t], name=t,
                                 line=dict(color=color_for(t), width=1.6)))
    fig.update_layout(title="Growth of $1 (normalized)", height=420,
                      margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    if prices.shape[1] >= 2:
        rets, _, cov, _ = estimate_moments(prices, cfg)
        corr = est.correlation_from_cov(cov)
        hm = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                       zmin=-1, zmax=1, aspect="auto", title="Return correlation")
        hm.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(hm, use_container_width=True)


# --------------------------------------------------------------------------- #
# Section: Methods (reference)
# --------------------------------------------------------------------------- #
def section_methods(cfg: dict):
    page_header(
        "📖 Construction Methods",
        "Every method takes the same historical data and turns it into a set of "
        "weights — they just disagree on <em>how</em>. Start with the cheat-sheet, "
        "then open any method for the full story.",
    )

    # ---- Orientation cheat-sheet -----------------------------------------
    st.subheader("Quick comparison")
    cheat = pd.DataFrame([
        ["Equal Weight",            "Put the same dollar amount in everything",          "No"],
        ["Inverse Volatility",      "Less money in the bumpy assets",                    "No"],
        ["Equal Risk Contribution", "Every asset adds the same amount of risk",          "No"],
        ["Risk Budgeting",          "You decide each asset's share of the risk",         "No"],
        ["Min Volatility",          "The smoothest possible ride",                       "No"],
        ["Max Sharpe",              "The best return for the risk taken",                "Yes"],
        ["Target Return",           "Smoothest way to reach a return goal",              "Yes"],
        ["Black-Litterman",         "The market's view nudged by your opinions",         "Yes (views)"],
        ["Entropy Pooling",         "Reweight real history to match your beliefs",       "Yes (views)"],
        ["Robust",                  "Max Sharpe that assumes your forecasts are noisy",  "Yes"],
    ], columns=["Method", "In one line", "Needs a return forecast?"])
    st.dataframe(cheat, hide_index=True, use_container_width=True)
    st.caption(
        "Methods that don't need a return forecast lean only on risk (how assets move) "
        "and tend to be steadier. Ones that do need forecasts can earn more but are more "
        "sensitive to your assumptions."
    )

    st.subheader("Each method, explained")
    # icon, name, tagline, plain-English, good-for, watch-out, under-the-hood
    docs = [
        ("⚖️", "Equal Weight", "split evenly across everything",
         "Cut the pie into equal slices — the same dollar amount in every asset. No "
         "forecasting, no math, no opinions. It sounds naive, but because it never bets "
         "big on a single guess, it is famously hard to beat in the real world.",
         "A no-fuss starting point and a benchmark to judge everything else against.",
         "Treats a calm bond and a wild tech ETF the same, so risk can be lopsided.",
         "wᵢ = 1/N for every asset."),

        ("🪶", "Inverse Volatility", "less money in the jumpy assets",
         "Give each asset a slice sized by how calm it is: steady assets get more, "
         "volatile ones get less. A quick, intuitive way to stop one wild holding from "
         "dominating the ride.",
         "A simple risk-aware upgrade over equal weight; very fast.",
         "Looks at each asset in isolation — it ignores how assets move together.",
         "wᵢ ∝ 1/σᵢ (the inverse of each asset's volatility), then rescaled to sum to 1."),

        ("🧬", "Equal Risk Contribution", "everyone shares the risk equally",
         "Instead of equal money, give weights so each asset contributes the same amount "
         "of the portfolio's overall wobble. A small slice of a risky asset can carry as "
         "much risk as a big slice of a calm one — this balances that. Also called "
         "'risk parity'.",
         "Genuine diversification of risk rather than just dollars; smooth, stable mixes.",
         "Often piles into low-volatility bonds (real funds lever them back up).",
         "Solve min ½w′Σw − Σ(1/N)·ln(wᵢ), then renormalize — equalizes each asset's "
         "risk contribution wᵢ·(Σw)ᵢ."),

        ("🎚️", "Risk Budgeting", "you set the risk shares",
         "Same idea as Equal Risk Contribution, but you hold the dials. Decide that, say, "
         "stocks should drive 60% of the risk and bonds 40%, and it finds the weights "
         "that deliver exactly those risk shares.",
         "When you have a clear view on where you want your risk to come from.",
         "Your budgets are only as good as your judgment about them.",
         "Same log-barrier solver as ERC with custom budgets bᵢ instead of 1/N."),

        ("🛡️", "Min Volatility", "the smoothest possible ride",
         "Find the single mix that bounces around the least, by exploiting how assets "
         "offset each other (when one zigs, another zags). It completely ignores returns "
         "— its only goal is calm.",
         "Capital preservation and the lowest-stress equity-like exposure.",
         "Can crowd into a handful of low-volatility names; low risk often means low return.",
         "Minimize w′Σw subject to the weights summing to 1 — the left-most point of the "
         "efficient frontier."),

        ("🏆", "Max Sharpe", "best bang for the risk",
         "Find the mix with the best return per unit of risk — the most 'efficient' deal "
         "on the menu. This is the classic textbook optimal portfolio.",
         "When you genuinely trust your return estimates.",
         "Very sensitive to those estimates: a small forecast error can produce wild, "
         "concentrated bets. (That's exactly why Robust and Black-Litterman exist.)",
         "Maximize the Sharpe ratio (w′μ − r_f)/√(w′Σw) — the 'tangency' portfolio."),

        ("🎯", "Target Return", "hit a goal with the least risk",
         "You name a return you want — say 7% a year — and it builds the smoothest "
         "portfolio expected to reach it. Slide the target up and down and you trace out "
         "every efficient trade-off between risk and reward.",
         "When you have a specific return objective to hit.",
         "Aiming above what the assets can realistically deliver just maximizes risk.",
         "Minimize w′Σw subject to w′μ = your target return."),

        ("🧠", "Black-Litterman", "the market's wisdom + your opinions",
         "Start from what the whole market implicitly believes is fair (so you're not "
         "fighting it), then gently tilt toward your specific opinions — e.g. 'I think "
         "emerging markets will beat bonds by 5%' — weighted by how confident you are. "
         "With no opinions, you simply get the market portfolio back.",
         "Adding a few personal views without the crazy, lopsided bets raw Max Sharpe "
         "produces. Much more stable.",
         "You still need sensible views and confidence levels.",
         "Bayesian blend of equilibrium returns π = δΣw_mkt with views (P, Q): "
         "μ_BL = [(τΣ)⁻¹ + P′Ω⁻¹P]⁻¹[(τΣ)⁻¹π + P′Ω⁻¹Q], then optimize."),

        ("🌀", "Entropy Pooling", "reweight history to fit your beliefs",
         "Keep every real historical day, but quietly let some days count a little more "
         "and others a little less so the overall picture matches your view (e.g. 'gold "
         "averages 8% a year') — while changing the original history as little as "
         "possible. No bell-curve assumption needed.",
         "Expressing views when returns are skewed or have fat tails; full-distribution "
         "views, not just averages.",
         "Strong views far from history distort the scenarios a lot (the app shows the "
         "'effective sample size' so you can watch for this).",
         "Find probabilities q closest to the prior (minimum relative entropy / KL "
         "divergence) that satisfy your views E_q[rₖ] = vₖ, then re-estimate moments."),

        ("🧱", "Robust", "plan for your forecasts being wrong",
         "Max Sharpe's careful sibling. It assumes your return guesses are noisy and "
         "optimizes for the worst case within a reasonable margin of error. The more you "
         "distrust the estimates (a higher κ dial), the safer and more spread-out the "
         "portfolio becomes.",
         "The real world, where return estimates are always uncertain — gives steadier, "
         "less concentrated weights.",
         "Crank κ too high and it collapses toward minimum-variance.",
         "Penalize the worst-case mean: maximize (w′μ − κ·√(w′(Σ/T)w))/√(w′Σw) "
         "(Ceria–Stubbs)."),
    ]
    for icon, name, tag, plain, good, watch, math in docs:
        with st.expander(f"{icon}  {name} — {tag}", expanded=False):
            st.markdown(f"**In plain English.** {plain}")
            st.markdown(f"✅ **Great for:** {good}")
            st.markdown(f"⚠️ **Watch out:** {watch}")
            st.caption(f"🔬 Under the hood — {math}")

    st.divider()
    st.subheader("How the backtest works")
    st.markdown(
        "- **Fixed weights** — pick an allocation and hold it. On each rebalance date the "
        "app trades back to your targets (paying a small cost), and in between the weights "
        "drift naturally as prices move. This answers: *how would this exact mix have done?*\n"
        "- **Walk-forward** — the honest test. On each rebalance date the optimizer is "
        "re-run using **only the data available up to that day**, then held until the next "
        "one. Because it never peeks at the future, it shows how a strategy would have "
        "performed *live*, with no hindsight bias."
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    cfg = sidebar_settings()
    section = cfg["section"]
    if section.endswith("Optimizer"):
        section_optimizer(cfg)
    elif section.endswith("Backtester"):
        section_backtester(cfg)
    elif section.endswith("Monte Carlo"):
        section_montecarlo(cfg)
    elif section.endswith("Macro Impulse"):
        section_macro(cfg)
    elif section.endswith("Asset Universe"):
        section_universe(cfg)
    else:
        section_methods(cfg)


if __name__ == "__main__":
    main()
