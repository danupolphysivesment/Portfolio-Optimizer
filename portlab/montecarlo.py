"""Forward-looking Monte Carlo simulation with Cholesky-correlated shocks.

We estimate a drift vector and covariance matrix from history, then simulate
future multi-asset paths. Correlation between assets is preserved by drawing
independent standard-normal shocks Z and rotating them through the Cholesky
factor L of the covariance (Σ = L·Lᵀ), so that Cov(L·z) = Σ. The simulated
assets are then combined into a constant-mix (rebalanced) portfolio.

Geometric Brownian motion is used for each asset (log-prices), which keeps
prices positive and compounds correctly. An optional Student-t shock adds fat
tails while still matching Σ (one shared chi-square scalar per step keeps the
cross-asset correlation intact).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _cov_to_corr(Sigma: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(Sigma))
    denom = np.outer(d, d)
    denom[denom == 0] = 1.0
    return Sigma / denom


def _safe_cholesky(Sigma: np.ndarray) -> np.ndarray:
    """Cholesky factor, nudging onto the PSD cone if needed."""
    try:
        return np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(Sigma)
        vals = np.clip(vals, 1e-12, None)
        fixed = (vecs * vals) @ vecs.T
        fixed = (fixed + fixed.T) / 2
        return np.linalg.cholesky(fixed)


def simulate_portfolio(
    mu: pd.Series,
    cov: pd.DataFrame,
    weights: pd.Series,
    horizon_steps: int,
    n_sims: int,
    ppy: float = 252.0,
    initial: float = 10_000.0,
    dist: str = "Normal",
    df: int = 5,
    seed: Optional[int] = 42,
    batch: int = 500,
) -> Dict[str, object]:
    """Simulate a constant-mix portfolio forward.

    Parameters
    ----------
    mu, cov : annualized expected returns / covariance (indexed by ticker).
    weights : portfolio weights (indexed by ticker), rescaled to sum to 1.
    horizon_steps : number of periods to simulate (e.g. 252 ≈ one year daily).
    n_sims : number of Monte Carlo paths.
    dist : "Normal" or "Student-t".
    df : degrees of freedom for the Student-t shock (df > 2).

    Returns a dict with the (n_sims × horizon_steps+1) value paths, terminal
    values, the input correlation matrix, and the realized-vs-input correlation
    error (a check that the Cholesky coupling worked).
    """
    assets = [t for t in weights.index if t in cov.index]
    mu_v = mu.reindex(assets).values.astype(float)
    Sigma = cov.loc[assets, assets].values.astype(float)
    w = weights.reindex(assets).values.astype(float)
    w = w / w.sum()
    n = len(assets)

    dt = 1.0 / ppy
    sqrt_dt = np.sqrt(dt)
    L = _safe_cholesky(Sigma)
    drift = (mu_v - 0.5 * np.diag(Sigma)) * dt  # GBM log-drift per step
    corr_in = _cov_to_corr(Sigma)

    rng = np.random.default_rng(seed)
    value_paths = np.empty((n_sims, horizon_steps + 1), dtype=float)
    value_paths[:, 0] = initial
    corr_err = float("nan")

    done = 0
    while done < n_sims:
        b = min(batch, n_sims - done)
        Z = rng.standard_normal((b, horizon_steps, n))
        # Correlated Brownian increments: Cov = Σ·dt regardless of n.
        incr = (Z @ L.T) * sqrt_dt
        if dist == "Student-t" and df > 2:
            g = rng.chisquare(df, size=(b, horizon_steps, 1))
            incr = incr * np.sqrt((df - 2) / np.maximum(g, 1e-9))
        log_incr = drift + incr
        asset_simple = np.expm1(log_incr)            # per-asset simple returns
        port_ret = asset_simple @ w                  # constant-mix (rebalanced)
        value_paths[done:done + b, 1:] = initial * np.cumprod(1.0 + port_ret, axis=1)

        if done == 0:  # validate Cholesky on the first batch
            sample = asset_simple.reshape(-1, n)
            if sample.shape[0] > n:
                realized = np.corrcoef(sample, rowvar=False)
                corr_err = float(np.nanmax(np.abs(realized - corr_in)))
        done += b

    return {
        "value_paths": value_paths,
        "terminal": value_paths[:, -1].copy(),
        "corr_input": pd.DataFrame(corr_in, index=assets, columns=assets),
        "corr_err": corr_err,
        "cholesky": pd.DataFrame(L, index=assets, columns=assets),
        "assets": assets,
        "horizon_steps": horizon_steps,
        "ppy": ppy,
        "initial": initial,
    }


def percentile_bands(
    value_paths: np.ndarray, qs=(5, 25, 50, 75, 95)
) -> Dict[int, np.ndarray]:
    return {q: np.percentile(value_paths, q, axis=0) for q in qs}


def path_max_drawdowns(value_paths: np.ndarray) -> np.ndarray:
    run_max = np.maximum.accumulate(value_paths, axis=1)
    dd = value_paths / run_max - 1.0
    return dd.min(axis=1)


def terminal_stats(result: dict, target_total_return: Optional[float] = None) -> Dict[str, float]:
    term = result["terminal"]
    init = result["initial"]
    rets = term / init - 1.0
    p5 = np.percentile(rets, 5)
    tail = rets[rets <= p5]
    stats = {
        "Median terminal": float(np.median(term)),
        "Mean terminal": float(term.mean()),
        "Median return": float(np.median(rets)),
        "p5 terminal": float(np.percentile(term, 5)),
        "p95 terminal": float(np.percentile(term, 95)),
        "Prob. of loss": float((term < init).mean()),
        "VaR 95% (terminal)": float(-p5),
        "CVaR 95% (terminal)": float(-tail.mean()) if tail.size else 0.0,
        "Best return": float(rets.max()),
        "Worst return": float(rets.min()),
    }
    if target_total_return is not None:
        stats["Prob. ≥ target"] = float((rets >= target_total_return).mean())
    return stats
