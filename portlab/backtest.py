"""Backtesting engine.

Two modes:

* ``backtest_fixed_weights`` — hold a static target allocation, rebalancing on a
  schedule (with optional transaction costs). Between rebalances the weights
  drift with the market.
* ``backtest_walk_forward`` — re-estimate moments and re-optimize on each
  rebalance date using only a trailing window of data (no look-ahead), then hold
  those weights until the next rebalance.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Map each UI label to a canonical period token. Using explicit tokens (rather
# than pandas offset aliases like "M"/"Q"/"A") keeps this working across pandas
# versions — those aliases were deprecated in 2.2 and removed in 3.0.
REBALANCE_RULES = {
    "Monthly": "Monthly",
    "Quarterly": "Quarterly",
    "Semi-Annual": "Semi-Annual",
    "Annual": "Annual",
    "Buy & Hold (never)": None,
}


def _rebalance_dates(index: pd.DatetimeIndex, rule: Optional[str]) -> set:
    """Last trading day within each calendar period — no resample/offset aliases."""
    if rule is None:
        return set()
    idx = pd.DatetimeIndex(index).sort_values()
    if rule == "Monthly":
        keys = list(zip(idx.year, idx.month))
    elif rule == "Quarterly":
        keys = list(zip(idx.year, idx.quarter))
    elif rule == "Semi-Annual":
        keys = list(zip(idx.year, (idx.month - 1) // 6))
    elif rule == "Annual":
        keys = list(idx.year)
    else:
        return set()
    # idx is ascending, so the last write per key is that period's final day.
    last_by_key = {}
    for key, date in zip(keys, idx):
        last_by_key[key] = date
    return set(last_by_key.values())


def backtest_fixed_weights(
    prices: pd.DataFrame,
    weights: pd.Series,
    rebalance: str = "Monthly",
    cost_bps: float = 0.0,
    initial: float = 10_000.0,
) -> Dict[str, object]:
    """Simulate a static-weight portfolio with periodic rebalancing.

    Returns a dict with the portfolio return series, equity curve, the drifting
    weight history, realized turnover, and total costs paid.
    """
    tickers = [t for t in weights.index if t in prices.columns]
    weights = weights[tickers]
    weights = weights / weights.sum()
    px = prices[tickers].dropna()
    asset_rets = px.pct_change().dropna()
    if asset_rets.empty:
        raise ValueError("Not enough price history to run a backtest.")

    rule = REBALANCE_RULES.get(rebalance, "Monthly")
    rebal_dates = _rebalance_dates(asset_rets.index, rule)

    target = weights.values
    w = target.copy()  # current drifting weights
    cost_rate = cost_bps / 1e4

    port_rets, weight_hist, dates = [], [], []
    total_cost = 0.0
    total_turnover = 0.0

    for date, row in asset_rets.iterrows():
        r = row.values
        # Rebalance at the open of a rebalance day before earning the day's return.
        if date in rebal_dates:
            turnover = np.abs(target - w).sum()
            cost = turnover * cost_rate
            total_cost += cost
            total_turnover += turnover
            w = target.copy()
            day_ret = w @ r - cost
        else:
            day_ret = w @ r
        weight_hist.append(w.copy())
        # Drift weights with realized returns for the next day.
        grown = w * (1.0 + r)
        w = grown / grown.sum()
        port_rets.append(day_ret)
        dates.append(date)

    port_rets = pd.Series(port_rets, index=pd.DatetimeIndex(dates), name="Portfolio")
    weight_df = pd.DataFrame(weight_hist, index=pd.DatetimeIndex(dates), columns=tickers)
    curve = initial * (1.0 + port_rets).cumprod()

    return {
        "returns": port_rets,
        "equity": curve,
        "weights": weight_df,
        "turnover": total_turnover,
        "total_cost_fraction": total_cost,
        "n_rebalances": len(rebal_dates),
    }


def backtest_walk_forward(
    prices: pd.DataFrame,
    weight_fn: Callable[[pd.DataFrame], pd.Series],
    rebalance: str = "Quarterly",
    lookback_days: int = 504,
    cost_bps: float = 0.0,
    initial: float = 10_000.0,
    min_history: int = 60,
) -> Dict[str, object]:
    """Out-of-sample backtest that re-optimizes on each rebalance date.

    ``weight_fn`` receives the trailing price window (up to and including the
    rebalance date) and returns a weight Series. Only past data is used, so the
    result is free of look-ahead bias.
    """
    px = prices.dropna()
    asset_rets = px.pct_change().dropna()
    rule = REBALANCE_RULES.get(rebalance, "Quarterly") or "Quarterly"
    rebal_dates = sorted(_rebalance_dates(asset_rets.index, rule))

    cols = list(px.columns)
    w = pd.Series(np.repeat(1.0 / len(cols), len(cols)), index=cols)
    have_weights = False
    cost_rate = cost_bps / 1e4

    port_rets, weight_hist, dates = [], [], []
    total_cost = 0.0
    total_turnover = 0.0
    applied: List[pd.Timestamp] = []

    for date, row in asset_rets.iterrows():
        r = row.reindex(cols).fillna(0.0).values
        if date in set(rebal_dates):
            window = px.loc[:date].tail(lookback_days)
            if len(window) >= min_history:
                new_w = weight_fn(window).reindex(cols).fillna(0.0)
                if new_w.sum() > 0:
                    new_w = new_w / new_w.sum()
                    turnover = np.abs(new_w.values - w.values).sum() if have_weights else new_w.abs().sum()
                    cost = turnover * cost_rate
                    total_cost += cost
                    total_turnover += turnover
                    w = new_w
                    have_weights = True
                    applied.append(date)
                    day_ret = float(w.values @ r) - cost
                    weight_hist.append(w.values.copy())
                    dates.append(date)
                    grown = w.values * (1.0 + r)
                    w = pd.Series(grown / grown.sum(), index=cols)
                    port_rets.append(day_ret)
                    continue
        # Non-rebalance day (or skipped): only invest once we have weights.
        if not have_weights:
            port_rets.append(0.0)
            weight_hist.append(np.repeat(np.nan, len(cols)))
            dates.append(date)
            continue
        day_ret = float(w.values @ r)
        weight_hist.append(w.values.copy())
        grown = w.values * (1.0 + r)
        w = pd.Series(grown / grown.sum(), index=cols)
        port_rets.append(day_ret)
        dates.append(date)

    port_rets = pd.Series(port_rets, index=pd.DatetimeIndex(dates), name="Portfolio")
    # Trim the leading flat segment before the first allocation.
    if applied:
        port_rets = port_rets.loc[applied[0]:]
    weight_df = pd.DataFrame(weight_hist, index=pd.DatetimeIndex(dates), columns=cols).loc[
        port_rets.index
    ]
    curve = initial * (1.0 + port_rets).cumprod()

    return {
        "returns": port_rets,
        "equity": curve,
        "weights": weight_df,
        "turnover": total_turnover,
        "total_cost_fraction": total_cost,
        "n_rebalances": len(applied),
        "rebalance_dates": applied,
    }


def benchmark_returns(
    prices: pd.DataFrame, weights: Dict[str, float], rebalance: str = "Monthly"
) -> Optional[pd.Series]:
    """Convenience benchmark (e.g. 60/40) over the available tickers."""
    avail = {t: wt for t, wt in weights.items() if t in prices.columns}
    if not avail:
        return None
    w = pd.Series(avail)
    res = backtest_fixed_weights(prices, w, rebalance=rebalance, cost_bps=0.0)
    return res["returns"]
