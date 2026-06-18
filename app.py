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
from portlab import metrics
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

CLASS_COLORS = {
    uni.EQUITY: "#4f46e5",
    uni.FIXED_INCOME: "#0d9488",
    uni.ALTERNATIVES: "#d97706",
    "Unknown": "#6b7280",
}

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; max-width: 1300px;}
      h1, h2, h3 {letter-spacing: -0.01em;}
      div[data-testid="stMetricValue"] {font-size: 1.35rem;}
      .pl-tag {display:inline-block;padding:2px 9px;border-radius:999px;
               font-size:0.72rem;font-weight:600;color:#fff;}
      .pl-sub {color:#6b7280;font-size:0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached data layer
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_prices_cached(tickers: tuple, start: str, end: str, source: str, convert_usd: bool = True):
    return dataio.load_prices(list(tickers), start, end, source=source, convert_usd=convert_usd)


def color_for(ticker: str) -> str:
    return CLASS_COLORS.get(uni.asset_class_of(ticker), CLASS_COLORS["Unknown"])


def fmt_pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


# --------------------------------------------------------------------------- #
# Sidebar — global settings
# --------------------------------------------------------------------------- #
def sidebar_settings() -> dict:
    st.sidebar.markdown("## 📈 Quant Portfolio Lab")
    st.sidebar.caption("Backtesting & portfolio construction on historical ETF data")

    section = st.sidebar.radio(
        "Workbench",
        ["🧮 Optimizer", "🧪 Backtester", "📚 Asset Universe", "📖 Methods"],
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


def estimate_moments(prices: pd.DataFrame, cfg: dict):
    rets = est.to_returns(prices)
    ppy = est.infer_periods_per_year(rets.index)
    mu = est.mean_returns(rets, cfg["ret_method"], cfg["span"], ppy)
    cov = est.cov_matrix(rets, cfg["cov_method"], cfg["span"], ppy)
    return rets, mu, cov, ppy


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
        line=dict(color="#4f46e5", width=3),
    ))
    # individual assets
    fig.add_trace(go.Scatter(
        x=np.sqrt(np.diag(cov.values)), y=mu.values, mode="markers+text",
        text=list(mu.index), textposition="top center", name="Assets",
        marker=dict(color=[color_for(t) for t in mu.index], size=9),
    ))
    # reference portfolios
    for name, w, sym, col in [
        ("Max Sharpe", opt.max_sharpe(mu, cov, rf), "star", "#dc2626"),
        ("Min Vol", opt.min_volatility(cov), "diamond", "#0d9488"),
        ("Selected", current_w, "circle", "#111827"),
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
    palette = ["#4f46e5", "#9ca3af", "#0d9488", "#d97706"]
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
        line=dict(color="#dc2626", width=1), name="Drawdown",
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
    st.title("🧮 Portfolio Optimizer")
    st.markdown(
        '<span class="pl-sub">Pick assets, estimate return & risk from history, '
        "and construct a portfolio with any of ten methods.</span>",
        unsafe_allow_html=True,
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
    st.title("🧪 Backtester")
    st.markdown(
        '<span class="pl-sub">Test any allocation with periodic rebalancing and '
        "trading costs, or run a walk-forward out-of-sample optimization.</span>",
        unsafe_allow_html=True,
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
    out = df.copy()
    for r in out.index:
        if r in pct_rows:
            out.loc[r] = out.loc[r].map(lambda x: fmt_pct(x, 2))
        else:
            out.loc[r] = out.loc[r].map(lambda x: f"{x:.2f}")
    return out


# --------------------------------------------------------------------------- #
# Section: Asset Universe
# --------------------------------------------------------------------------- #
def section_universe(cfg: dict):
    st.title("📚 Asset Universe")
    st.markdown(
        '<span class="pl-sub">A broad ETF universe across three asset classes — '
        "global, regional, and US-sector equity, a full fixed-income ladder, and "
        "alternatives. Explore historical behavior and co-movement.</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"{len(uni.ASSETS)} ETFs total. Non-USD listings (EUR/JPY/GBp) are "
        "converted to USD at load time when the sidebar toggle is on."
    )
    for ac in uni.ASSET_CLASSES:
        members = uni.assets_in_class(ac)
        st.markdown(
            f'<span class="pl-tag" style="background:{CLASS_COLORS[ac]}">{ac} '
            f"· {len(members)}</span>",
            unsafe_allow_html=True,
        )
        rows = [{
            "Ticker (Yahoo)": a.ticker, "Name": a.name, "Region": a.region,
            "Ccy": a.currency, "Note": a.note, "Bloomberg": a.bb or "—",
        } for a in members]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

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
    st.title("📖 Construction Methods")
    st.caption("How each optimizer turns historical estimates into weights.")
    docs = [
        ("Equal Weight", "Allocate 1/N to every asset. No estimation, hard to beat out-of-sample; a robust baseline."),
        ("Inverse Volatility", "Weight ∝ 1/σᵢ. Down-weights volatile assets using only the diagonal of the covariance — ignores correlations."),
        ("Equal Risk Contribution", "Each asset contributes the same share of portfolio variance. Solved via the convex program min ½w′Σw − Σ(1/N)·ln(wᵢ), then renormalized — the 'risk parity' portfolio."),
        ("Risk Budgeting", "Generalizes ERC to arbitrary risk shares bᵢ: each asset's risk contribution equals its budget. Same log-barrier solver with custom budgets."),
        ("Min Volatility", "Minimize w′Σw subject to the budget and weight bounds — the left-most point of the efficient frontier. Depends only on covariances."),
        ("Max Sharpe", "Maximize (w′μ − r_f)/√(w′Σw): the tangency portfolio. Sensitive to estimation error in μ — see Robust and Black-Litterman."),
        ("Target Return", "Minimum-variance portfolio achieving a chosen expected return w′μ = target. Sweeping the target traces the efficient frontier."),
        ("Black-Litterman", "Start from CAPM-implied equilibrium returns π = δΣw_mkt, then blend in subjective views (P, Q) via Bayes: μ_BL = [(τΣ)⁻¹ + P′Ω⁻¹P]⁻¹[(τΣ)⁻¹π + P′Ω⁻¹Q]. Optimize the posterior. Tames the instability of raw mean-variance."),
        ("Entropy Pooling", "Meucci's approach: keep the historical scenarios but re-weight their probabilities to the distribution closest (minimum relative entropy / Kullback-Leibler) to the prior that satisfies your views, e.g. E_q[rₖ]=vₖ. Revised moments then feed mean-variance. Handles non-normal, full-distribution views."),
        ("Robust (worst-case mean)", "Account for estimation error in μ: the sample mean has error covariance ≈ Σ/T, so the worst-case return inside a confidence ellipsoid is w′μ − κ·√(w′(Σ/T)w). Maximizing the resulting worst-case Sharpe (Ceria–Stubbs) yields more stable, less concentrated weights as κ grows."),
    ]
    for name, body in docs:
        with st.expander(name, expanded=False):
            st.markdown(body)
    st.divider()
    st.markdown(
        "**Backtesting.** Fixed-weight mode holds a target allocation, rebalancing on "
        "your schedule and charging turnover-based costs; weights drift with the market "
        "between rebalances. Walk-forward mode re-estimates moments and re-optimizes on "
        "each rebalance date using only a trailing window — an honest out-of-sample test "
        "free of look-ahead bias."
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
    elif section.endswith("Asset Universe"):
        section_universe(cfg)
    else:
        section_methods(cfg)


if __name__ == "__main__":
    main()
