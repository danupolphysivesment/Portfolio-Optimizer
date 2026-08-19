"""Portfolio exposure, concentration and diversification analytics.

Everything here is derivable from three inputs the app already has:

* the portfolio **weights** (a ``pd.Series`` indexed by Yahoo ticker),
* each ETF's **metadata** in :mod:`portlab.universe` (asset class, region,
  listing currency, sleeve note), and
* the return **covariance / correlation** produced by the estimators.

No single-stock *look-through* is attempted — the app has no holdings feed —
so overlap/sector concentration is measured at the fund-sleeve level and is
labelled as such in the UI. The functions are pure so they can be unit-tested
without Streamlit.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from . import universe as uni
from .optimizers import portfolio_vol, risk_contributions

# Exposure dimensions the monitor can break a portfolio down by. Each maps a
# ticker to its bucket label using only universe metadata.
_KEY_FNS: Dict[str, Callable[[str], str]] = {
    "Asset class": uni.asset_class_of,
    "Region": lambda t: getattr(uni.ASSET_BY_TICKER.get(t), "region", "Unknown"),
    "Currency": uni.currency_of,
}
DIMENSIONS: List[str] = list(_KEY_FNS)


def _clean(weights: pd.Series) -> pd.Series:
    """Drop ~zero weights and renormalise so shares sum to 1 (long-only view)."""
    w = weights[weights.abs() > 1e-6].astype(float)
    total = w.sum()
    return w / total if total not in (0.0, np.nan) else w


# --------------------------------------------------------------------------- #
# Exposure breakdowns
# --------------------------------------------------------------------------- #
def group_exposure(weights: pd.Series, dimension: str) -> pd.Series:
    """Sum weights into buckets along one metadata ``dimension``.

    Returns a Series indexed by bucket label, sorted descending.
    """
    key = _KEY_FNS[dimension]
    w = _clean(weights)
    grouped: Dict[str, float] = {}
    for t, val in w.items():
        grouped[key(t)] = grouped.get(key(t), 0.0) + float(val)
    return pd.Series(grouped).sort_values(ascending=False)


def currency_split(weights: pd.Series) -> pd.Series:
    """Coarse USD vs non-USD *listing* exposure (before any FX hedging)."""
    w = _clean(weights)
    usd = float(sum(v for t, v in w.items() if uni.currency_of(t) == "USD"))
    return pd.Series({"USD": usd, "Non-USD": 1.0 - usd})


# --------------------------------------------------------------------------- #
# Concentration
# --------------------------------------------------------------------------- #
def herfindahl(weights: pd.Series) -> float:
    """Herfindahl-Hirschman Index of the weights (sum of squared shares)."""
    w = _clean(weights)
    return float((w ** 2).sum())


def effective_holdings(weights: pd.Series) -> float:
    """Effective number of positions = 1 / HHI (equal-weight ⇒ N)."""
    hhi = herfindahl(weights)
    return float(1.0 / hhi) if hhi > 0 else 0.0


def concentration(weights: pd.Series) -> Dict[str, float]:
    """Top-k NAV shares plus HHI-based diversification measures."""
    w = _clean(weights).abs().sort_values(ascending=False)
    cum = w.cumsum()

    def topk(k: int) -> float:
        return float(cum.iloc[min(k, len(w)) - 1]) if len(w) else 0.0

    return {
        "Positions": int(len(w)),
        "Largest weight": float(w.iloc[0]) if len(w) else 0.0,
        "Top 3": topk(3),
        "Top 5": topk(5),
        "Top 10": topk(10),
        "HHI": herfindahl(weights),
        "Effective N": effective_holdings(weights),
    }


# --------------------------------------------------------------------------- #
# Diversification / risk-based views
# --------------------------------------------------------------------------- #
def diversification_ratio(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Weighted-average asset vol ÷ portfolio vol.

    1.0 means no diversification benefit; higher is better. Uses the same
    covariance the rest of the app estimates.
    """
    w = _clean(weights)
    idx = [t for t in w.index if t in cov.index]
    if len(idx) < 1:
        return 0.0
    w = w[idx]
    sigma = cov.loc[idx, idx]
    asset_vol = np.sqrt(np.diag(sigma.values))
    weighted_avg_vol = float(w.values @ asset_vol)
    port_vol = portfolio_vol(w.values, sigma.values)
    return float(weighted_avg_vol / port_vol) if port_vol > 0 else 0.0


def avg_pairwise_corr(weights: pd.Series, corr: pd.DataFrame) -> float:
    """Weight-blended average of the off-diagonal pairwise correlations."""
    w = _clean(weights)
    idx = [t for t in w.index if t in corr.index]
    if len(idx) < 2:
        return 0.0
    w = w[idx]
    C = corr.loc[idx, idx].values
    ww = np.outer(w.values, w.values)
    np.fill_diagonal(ww, 0.0)
    np.fill_diagonal(C, 0.0)
    denom = ww.sum()
    return float((ww * C).sum() / denom) if denom > 0 else 0.0


def risk_contrib_by_group(weights: pd.Series, cov: pd.DataFrame, dimension: str) -> pd.Series:
    """Aggregate each asset's % contribution to portfolio variance into buckets.

    Complements :func:`group_exposure`: capital weight vs *risk* weight often
    diverge (e.g. a small equity sleeve can dominate portfolio variance).
    """
    w = _clean(weights)
    idx = [t for t in w.index if t in cov.index]
    if not idx:
        return pd.Series(dtype=float)
    rc = risk_contributions(w[idx] / w[idx].sum(), cov.loc[idx, idx])
    key = _KEY_FNS[dimension]
    grouped: Dict[str, float] = {}
    for t, val in rc.items():
        grouped[key(t)] = grouped.get(key(t), 0.0) + float(val)
    return pd.Series(grouped).sort_values(ascending=False)


def portfolio_beta(port_returns: pd.Series, bench_returns: pd.Series) -> float:
    """OLS beta of portfolio returns to a benchmark, on the overlapping dates."""
    df = pd.concat([port_returns, bench_returns], axis=1, join="inner").dropna()
    if len(df) < 3:
        return float("nan")
    p, b = df.iloc[:, 0].values, df.iloc[:, 1].values
    var_b = np.var(b, ddof=1)
    return float(np.cov(p, b, ddof=1)[0, 1] / var_b) if var_b > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Mandate / limit monitoring (RAG flags)
# --------------------------------------------------------------------------- #
def limit_check(
    weights: pd.Series,
    max_single: float,
    max_class: float,
    max_region: float,
    watch_frac: float = 0.9,
) -> List[Dict[str, object]]:
    """Compare the portfolio against concentration limits and flag breaches.

    Returns one row per rule with the worst offender, its value, the limit and
    a red/amber/green ``status`` ("Breach" / "Watch" / "OK"). ``watch_frac`` is
    the fraction of a limit at which a position turns amber.
    """
    w = _clean(weights)
    rows: List[Dict[str, object]] = []

    def add(rule: str, offender: str, value: float, limit: float) -> None:
        if value > limit:
            status = "Breach"
        elif value >= watch_frac * limit:
            status = "Watch"
        else:
            status = "OK"
        rows.append({"Rule": rule, "Largest": offender, "Exposure": value,
                     "Limit": limit, "Status": status})

    if len(w):
        top = w.abs().idxmax()
        add("Single ETF", top, float(w[top]), max_single)
    for dim, limit, rule in (("Asset class", max_class, "Asset class"),
                             ("Region", max_region, "Region")):
        g = group_exposure(weights, dim)
        if len(g):
            add(rule, str(g.index[0]), float(g.iloc[0]), limit)
    return rows
