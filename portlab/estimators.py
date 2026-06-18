"""Return / risk estimation from historical prices.

Everything downstream (optimizers, frontier, metrics) consumes *annualized*
expected returns (a Series) and an *annualized* covariance matrix (a DataFrame),
both indexed by ticker. Keeping a single convention avoids unit bugs.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def infer_periods_per_year(index: pd.Index) -> float:
    """Estimate observations per year from the median spacing of the index."""
    if len(index) < 3:
        return 252.0
    deltas = pd.Series(index).diff().dropna().dt.days
    med = float(np.median(deltas)) if len(deltas) else 1.0
    if med <= 0:
        return 252.0
    if med <= 3:
        return 252.0  # daily / business-daily
    if med <= 10:
        return 52.0  # weekly
    if med <= 45:
        return 12.0  # monthly
    return 4.0  # quarterly


def to_returns(prices: pd.DataFrame, kind: str = "simple") -> pd.DataFrame:
    if kind == "log":
        rets = np.log(prices).diff()
    else:
        rets = prices.pct_change()
    return rets.dropna(how="all")


def mean_returns(
    returns: pd.DataFrame,
    method: str = "Historical mean",
    span: int = 90,
    periods_per_year: Optional[float] = None,
) -> pd.Series:
    """Annualized expected returns."""
    ppy = periods_per_year or infer_periods_per_year(returns.index)
    if method == "EWMA":
        mu = returns.ewm(span=span).mean().iloc[-1]
    else:  # historical sample mean
        mu = returns.mean()
    return mu * ppy


def cov_matrix(
    returns: pd.DataFrame,
    method: str = "Sample",
    span: int = 90,
    periods_per_year: Optional[float] = None,
) -> pd.DataFrame:
    """Annualized covariance matrix."""
    ppy = periods_per_year or infer_periods_per_year(returns.index)
    cols = returns.columns

    if method == "Ledoit-Wolf shrinkage":
        from sklearn.covariance import LedoitWolf

        clean = returns.dropna()
        lw = LedoitWolf().fit(clean.values)
        cov = pd.DataFrame(lw.covariance_, index=cols, columns=cols)
    elif method == "EWMA":
        cov = returns.ewm(span=span).cov().iloc[-len(cols):]
        cov.index = cov.index.droplevel(0)
        cov = cov.loc[cols, cols]
    else:  # sample
        cov = returns.cov()

    cov = cov * ppy
    # Numerical hygiene: symmetrize and nudge onto the PSD cone.
    cov = (cov + cov.T) / 2
    return _nearest_psd(cov)


def _nearest_psd(cov: pd.DataFrame, eps: float = 1e-10) -> pd.DataFrame:
    vals, vecs = np.linalg.eigh(cov.values)
    vals = np.clip(vals, eps, None)
    fixed = (vecs * vals) @ vecs.T
    fixed = (fixed + fixed.T) / 2
    return pd.DataFrame(fixed, index=cov.index, columns=cov.columns)


def correlation_from_cov(cov: pd.DataFrame) -> pd.DataFrame:
    d = np.sqrt(np.diag(cov.values))
    denom = np.outer(d, d)
    corr = cov.values / np.where(denom == 0, 1, denom)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)
