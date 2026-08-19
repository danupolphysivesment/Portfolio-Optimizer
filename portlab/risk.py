"""Tail-risk analytics: VaR, CVaR (Expected Shortfall), Extreme Value Theory,
and stress testing.

VaR/CVaR are reported as positive loss fractions per period (at the data's
frequency, i.e. 1-day for daily data). Four estimators are provided:

* Historical  — empirical quantile of realized returns.
* Gaussian    — parametric normal (mean/σ).
* Cornish-Fisher — normal quantile adjusted for skew & excess kurtosis.
* EVT (POT)   — Generalized Pareto fit to the loss tail (McNeil–Frey), the
                gold standard for rare, extreme losses.

Stress testing covers historical crisis replays (actual asset moves over real
crisis windows) and hypothetical instantaneous factor shocks by asset class.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from . import universe as uni


# --------------------------------------------------------------------------- #
# Portfolio return series
# --------------------------------------------------------------------------- #
def portfolio_returns(rets: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Constant-mix (rebalanced) historical portfolio return series."""
    w = weights.reindex(rets.columns).fillna(0.0)
    s = w.sum()
    if s > 0:
        w = w / s
    return (rets[w.index] * w.values).sum(axis=1)


# --------------------------------------------------------------------------- #
# VaR / CVaR estimators (return positive loss fractions)
# --------------------------------------------------------------------------- #
def var_historical(r: pd.Series, level: float) -> float:
    return float(-np.percentile(r, (1 - level) * 100))


def cvar_historical(r: pd.Series, level: float) -> float:
    var = var_historical(r, level)
    tail = r[r <= -var]
    return float(-tail.mean()) if len(tail) else var


def var_gaussian(r: pd.Series, level: float) -> float:
    z = stats.norm.ppf(1 - level)
    return float(-(r.mean() + z * r.std(ddof=1)))


def cvar_gaussian(r: pd.Series, level: float) -> float:
    z = stats.norm.ppf(1 - level)
    return float(-(r.mean() - r.std(ddof=1) * stats.norm.pdf(z) / (1 - level)))


def var_cornish_fisher(r: pd.Series, level: float) -> float:
    z = stats.norm.ppf(1 - level)
    s = float(stats.skew(r))
    k = float(stats.kurtosis(r, fisher=True))  # excess kurtosis
    z_cf = (z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24
            - (2 * z**3 - 5 * z) * s**2 / 36)
    return float(-(r.mean() + z_cf * r.std(ddof=1)))


def evt_pot_fit(r: pd.Series, threshold_q: float = 0.95) -> Dict[str, float]:
    """Fit a Generalized Pareto Distribution to the loss tail (peaks-over-
    threshold). Returns shape ξ, scale β, threshold u, and counts."""
    losses = -np.asarray(r, dtype=float)
    u = float(np.quantile(losses, threshold_q))
    excess = losses[losses > u] - u
    n, n_u = len(losses), len(excess)
    if n_u < 10:
        return {"ok": False, "n": n, "n_u": n_u, "u": u}
    xi, _, beta = stats.genpareto.fit(excess, floc=0.0)
    return {"ok": True, "xi": float(xi), "beta": float(beta), "u": u,
            "n": int(n), "n_u": int(n_u), "threshold_q": threshold_q}


def evt_var_cvar(fit: Dict[str, float], level: float) -> Tuple[float, float]:
    """VaR & CVaR at ``level`` from a fitted GPD tail (McNeil–Frey formulas)."""
    if not fit.get("ok"):
        return float("nan"), float("nan")
    xi, beta, u = fit["xi"], fit["beta"], fit["u"]
    n, n_u = fit["n"], fit["n_u"]
    ratio = (n / n_u) * (1 - level)
    if abs(xi) < 1e-6:  # exponential limit
        var = u - beta * np.log(ratio)
    else:
        var = u + (beta / xi) * (ratio ** (-xi) - 1.0)
    cvar = (var + beta - xi * u) / (1 - xi) if xi < 1 else float("nan")
    return float(var), float(cvar)


def evt_tail_curve(r: pd.Series, fit: Dict[str, float], n_points: int = 60):
    """Empirical vs fitted GPD exceedance probability for the loss tail."""
    losses = np.sort(-np.asarray(r, dtype=float))
    u = fit["u"]
    tail = losses[losses > u]
    if len(tail) < 2:
        return None
    xs = np.linspace(u, tail.max(), n_points)
    n, n_u = fit["n"], fit["n_u"]
    emp = np.array([(losses > x).mean() for x in xs])           # P(L > x)
    fitted = (n_u / n) * stats.genpareto.sf(xs - u, fit["xi"], scale=fit["beta"])
    return pd.DataFrame({"loss": xs, "empirical": emp, "gpd_fit": fitted})


def var_comparison_table(r: pd.Series, levels=(0.95, 0.99),
                         threshold_q: float = 0.95) -> pd.DataFrame:
    """Tidy VaR/CVaR table across all four methods and given levels."""
    fit = evt_pot_fit(r, threshold_q)
    rows = {}
    for lv in levels:
        ev_var, ev_cvar = evt_var_cvar(fit, lv)
        tag = f"{int(lv*100)}%"
        rows[f"VaR {tag}"] = {
            "Historical": var_historical(r, lv),
            "Gaussian": var_gaussian(r, lv),
            "Cornish-Fisher": var_cornish_fisher(r, lv),
            "EVT (POT)": ev_var,
        }
        rows[f"CVaR {tag}"] = {
            "Historical": cvar_historical(r, lv),
            "Gaussian": cvar_gaussian(r, lv),
            "Cornish-Fisher": np.nan,  # not standard for CF
            "EVT (POT)": ev_cvar,
        }
    table = pd.DataFrame(rows).T  # rows = metrics, cols = methods
    return table, fit


# --------------------------------------------------------------------------- #
# Stress testing
# --------------------------------------------------------------------------- #
CRISES: Dict[str, Tuple[str, str]] = {
    "GFC 2008": ("2007-10-09", "2009-03-09"),
    "EU debt crisis 2011": ("2011-07-01", "2011-10-04"),
    "China / oil 2015–16": ("2015-08-01", "2016-02-11"),
    "Q4-2018 selloff": ("2018-09-20", "2018-12-24"),
    "COVID crash 2020": ("2020-02-19", "2020-03-23"),
    "2022 rate shock": ("2022-01-03", "2022-10-12"),
    "2023 banking stress": ("2023-03-06", "2023-03-24"),
}

# Hypothetical instantaneous shocks by asset class.
FACTOR_SCENARIOS: Dict[str, Dict[str, float]] = {
    "Equity −20% correction": {"Equity": -0.20, "Fixed Income": 0.00, "Alternatives": -0.05},
    "Equity −35% crash + flight-to-quality": {"Equity": -0.35, "Fixed Income": 0.05, "Alternatives": -0.10},
    "Rate shock (+200bp)": {"Equity": -0.06, "Fixed Income": -0.09, "Alternatives": -0.03},
    "Inflation shock": {"Equity": -0.10, "Fixed Income": -0.06, "Alternatives": 0.12},
    "Risk-off / flight-to-quality": {"Equity": -0.15, "Fixed Income": 0.04, "Alternatives": 0.03},
    "Liquidity crisis": {"Equity": -0.25, "Fixed Income": -0.05, "Alternatives": -0.15},
}


def historical_stress(prices: pd.DataFrame, weights: pd.Series
                      ) -> Tuple[pd.DataFrame, List[str]]:
    """Replay real crisis windows that fall fully inside the loaded price data.

    Returns (results DataFrame, list of skipped crisis names out of range).
    """
    data_start, data_end = prices.index.min(), prices.index.max()
    w = weights.reindex(prices.columns).fillna(0.0)
    s = w.sum()
    if s > 0:
        w = w / s
    rows, skipped = [], []
    for name, (start, end) in CRISES.items():
        s_ts, e_ts = pd.Timestamp(start), pd.Timestamp(end)
        if s_ts < data_start or e_ts > data_end:
            skipped.append(name)
            continue
        sl = prices.loc[s_ts:e_ts, w.index]
        if len(sl) < 3:
            skipped.append(name)
            continue
        daily = sl.pct_change().dropna()
        port = float((1 + (daily * w.values).sum(axis=1)).prod() - 1)
        # worst single asset over the window
        asset_cum = sl.iloc[-1] / sl.iloc[0] - 1
        worst = asset_cum.idxmin()
        rows.append({
            "Scenario": name, "From": sl.index[0].date(), "To": sl.index[-1].date(),
            "Portfolio": port, "Worst asset": f"{worst} ({asset_cum[worst]*100:.0f}%)",
        })
    return pd.DataFrame(rows), skipped


def apply_factor_shock(weights: pd.Series, class_shocks: Dict[str, float]
                       ) -> Tuple[float, pd.Series]:
    """Instantaneous portfolio P&L for a per-asset-class shock; also per-asset
    contributions."""
    w = weights / weights.sum() if weights.sum() > 0 else weights
    contrib = {t: float(wt) * class_shocks.get(uni.asset_class_of(t), 0.0)
               for t, wt in w.items()}
    return float(sum(contrib.values())), pd.Series(contrib)
