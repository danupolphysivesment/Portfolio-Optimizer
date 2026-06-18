"""Portfolio construction methods.

Conventions
-----------
* ``mu``  : annualized expected returns, pandas Series indexed by ticker.
* ``cov`` : annualized covariance matrix, pandas DataFrame (ticker x ticker).
* All optimizers return a weight Series (indexed by ticker) that sums to 1.

Methods
-------
Heuristic / closed form : equal_weight, inverse_volatility
Risk based              : risk_budgeting, equal_risk_contribution
Mean-variance (scipy)   : min_volatility, max_sharpe, target_return, robust
Views based             : black_litterman (posterior), entropy_pooling (posterior)

Black-Litterman and Entropy Pooling produce a *revised* (mu, cov); the caller
then runs a standard mean-variance optimizer on those posteriors. Helper
``posterior_then_optimize`` wires that together.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Canonical method names surfaced in the UI.
EQUAL_WEIGHT = "Equal Weight"
INVERSE_VOL = "Inverse Volatility"
ERC = "Equal Risk Contribution"
RISK_BUDGET = "Risk Budgeting"
MIN_VOL = "Min Volatility"
MAX_SHARPE = "Max Sharpe"
TARGET_RETURN = "Target Return"
BLACK_LITTERMAN = "Black-Litterman"
ENTROPY_POOLING = "Entropy Pooling"
ROBUST = "Robust (worst-case mean)"

ALL_METHODS = [
    MAX_SHARPE,
    MIN_VOL,
    TARGET_RETURN,
    BLACK_LITTERMAN,
    ENTROPY_POOLING,
    ROBUST,
    RISK_BUDGET,
    INVERSE_VOL,
    ERC,
    EQUAL_WEIGHT,
]

# Methods whose primary inputs are estimated moments only.
NEEDS_MU = {MAX_SHARPE, TARGET_RETURN, BLACK_LITTERMAN, ENTROPY_POOLING, ROBUST}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _series(weights: np.ndarray, index) -> pd.Series:
    w = np.asarray(weights, dtype=float)
    w = np.where(np.abs(w) < 1e-8, 0.0, w)
    s = pd.Series(w, index=index)
    total = s.sum()
    return s / total if total != 0 else s


def portfolio_vol(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(max(weights @ cov @ weights, 0.0)))


def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    return float(weights @ mu)


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Percentage contribution of each asset to total portfolio variance/vol."""
    w = weights.values
    Sigma = cov.loc[weights.index, weights.index].values
    port_var = w @ Sigma @ w
    if port_var <= 0:
        return pd.Series(np.zeros_like(w), index=weights.index)
    marginal = Sigma @ w
    contrib = w * marginal  # absolute contribution to variance
    return pd.Series(contrib / port_var, index=weights.index)


def _bounds(n: int, weight_bounds: Tuple[float, float]) -> List[Tuple[float, float]]:
    lo, hi = weight_bounds
    return [(lo, hi)] * n


def _sum_to_one():
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


# --------------------------------------------------------------------------- #
# heuristic / closed form
# --------------------------------------------------------------------------- #
def equal_weight(tickers) -> pd.Series:
    n = len(tickers)
    return pd.Series(np.repeat(1.0 / n, n), index=tickers)


def inverse_volatility(cov: pd.DataFrame) -> pd.Series:
    vol = np.sqrt(np.diag(cov.values))
    inv = 1.0 / np.where(vol == 0, np.nan, vol)
    inv = np.nan_to_num(inv)
    return _series(inv, cov.index)


# --------------------------------------------------------------------------- #
# risk budgeting / ERC  (convex log-barrier formulation)
# --------------------------------------------------------------------------- #
def risk_budgeting(cov: pd.DataFrame, budgets: Optional[np.ndarray] = None) -> pd.Series:
    """Solve for weights whose risk contributions match ``budgets``.

    Uses the standard convex reformulation
        minimize  0.5 * y' Σ y  -  Σ b_i ln(y_i),   y > 0
    whose stationarity condition  Σy = b / y  gives risk contributions
    proportional to b. The result is rescaled to sum to one (long-only,
    fully invested).
    """
    Sigma = cov.values
    n = Sigma.shape[0]
    if budgets is None:
        budgets = np.repeat(1.0 / n, n)
    budgets = np.asarray(budgets, dtype=float)
    budgets = budgets / budgets.sum()

    def obj(y):
        return 0.5 * y @ Sigma @ y - budgets @ np.log(y)

    def grad(y):
        return Sigma @ y - budgets / y

    x0 = np.repeat(1.0 / n, n)
    res = minimize(
        obj, x0, jac=grad, method="L-BFGS-B",
        bounds=[(1e-9, None)] * n,
        options={"maxiter": 10000, "ftol": 1e-14},
    )
    y = res.x
    return _series(y / y.sum(), cov.index)


def equal_risk_contribution(cov: pd.DataFrame) -> pd.Series:
    return risk_budgeting(cov, budgets=None)


# --------------------------------------------------------------------------- #
# mean-variance family
# --------------------------------------------------------------------------- #
def min_volatility(cov: pd.DataFrame, weight_bounds=(0.0, 1.0)) -> pd.Series:
    Sigma = cov.values
    n = Sigma.shape[0]
    res = minimize(
        lambda w: w @ Sigma @ w,
        np.repeat(1.0 / n, n),
        jac=lambda w: 2 * Sigma @ w,
        method="SLSQP",
        bounds=_bounds(n, weight_bounds),
        constraints=[_sum_to_one()],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return _series(res.x, cov.index)


def max_sharpe(
    mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0, weight_bounds=(0.0, 1.0)
) -> pd.Series:
    m = mu.values
    Sigma = cov.values
    n = len(m)

    def neg_sharpe(w):
        ret = w @ m - rf
        vol = np.sqrt(max(w @ Sigma @ w, 1e-16))
        return -ret / vol

    res = minimize(
        neg_sharpe,
        np.repeat(1.0 / n, n),
        method="SLSQP",
        bounds=_bounds(n, weight_bounds),
        constraints=[_sum_to_one()],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return _series(res.x, cov.index)


def target_return(
    mu: pd.Series, cov: pd.DataFrame, target: float, weight_bounds=(0.0, 1.0)
) -> pd.Series:
    """Minimum-variance portfolio achieving an annualized ``target`` return.

    The target is clipped to the achievable range given the bounds so the
    problem stays feasible.
    """
    m = mu.values
    Sigma = cov.values
    n = len(m)
    lo, hi = weight_bounds
    # Feasible return range under the box constraints (approximate, ignores
    # the simplex coupling but good enough to keep the solver feasible).
    target = float(np.clip(target, m.min(), m.max()))

    res = minimize(
        lambda w: w @ Sigma @ w,
        np.repeat(1.0 / n, n),
        jac=lambda w: 2 * Sigma @ w,
        method="SLSQP",
        bounds=_bounds(n, weight_bounds),
        constraints=[_sum_to_one(), {"type": "eq", "fun": lambda w: w @ m - target}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not res.success:  # fall back to closest feasible (min-var) if infeasible
        return min_volatility(cov, weight_bounds)
    return _series(res.x, cov.index)


def robust_portfolio(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = 0.0,
    kappa: float = 1.0,
    n_obs: int = 252,
    weight_bounds=(0.0, 1.0),
) -> pd.Series:
    """Robust max-Sharpe with box/ellipsoidal uncertainty on the mean.

    The sample mean has estimation-error covariance ~ Σ / T. We penalize the
    objective by the worst-case shortfall ``kappa * sqrt(w' (Σ/T) w)``
    (Ceria–Stubbs style), then maximize the resulting worst-case Sharpe ratio.
    Larger ``kappa`` ⇒ more conservative (shrinks toward min-variance).
    """
    m = mu.values
    Sigma = cov.values
    n = len(m)
    Theta = Sigma / max(n_obs, 1)

    def neg_robust_sharpe(w):
        worst_ret = w @ m - kappa * np.sqrt(max(w @ Theta @ w, 1e-18)) - rf
        vol = np.sqrt(max(w @ Sigma @ w, 1e-16))
        return -worst_ret / vol

    res = minimize(
        neg_robust_sharpe,
        np.repeat(1.0 / n, n),
        method="SLSQP",
        bounds=_bounds(n, weight_bounds),
        constraints=[_sum_to_one()],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    return _series(res.x, cov.index)


# --------------------------------------------------------------------------- #
# Black-Litterman
# --------------------------------------------------------------------------- #
def implied_equilibrium_returns(
    cov: pd.DataFrame, market_weights: pd.Series, risk_aversion: float
) -> pd.Series:
    """Reverse-optimized (CAPM) returns: π = δ Σ w_mkt."""
    w = market_weights.reindex(cov.index).fillna(0.0).values
    pi = risk_aversion * cov.values @ w
    return pd.Series(pi, index=cov.index)


def black_litterman_posterior(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    P: Optional[np.ndarray] = None,
    Q: Optional[np.ndarray] = None,
    omega: Optional[np.ndarray] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Return (posterior mean, posterior covariance).

    With no views, the posterior mean collapses to the equilibrium prior π and
    the posterior covariance is (1 + τ)Σ.
    """
    Sigma = cov.values
    pi = implied_equilibrium_returns(cov, market_weights, risk_aversion).values
    tauSigma = tau * Sigma

    if P is None or Q is None or len(Q) == 0:
        mu_post = pi
        Sigma_post = Sigma + tauSigma
        return pd.Series(mu_post, index=cov.index), pd.DataFrame(
            Sigma_post, index=cov.index, columns=cov.index
        )

    P = np.atleast_2d(P)
    Q = np.asarray(Q, dtype=float).reshape(-1)
    if omega is None:
        # He–Litterman default: Ω = diag(P τΣ P').
        omega = np.diag(np.diag(P @ tauSigma @ P.T))
    omega = np.atleast_2d(omega)

    A = np.linalg.inv(tauSigma)
    B = P.T @ np.linalg.inv(omega) @ P
    M = np.linalg.inv(A + B)  # posterior cov of the *mean*
    mu_post = M @ (A @ pi + P.T @ np.linalg.inv(omega) @ Q)
    Sigma_post = Sigma + M

    return (
        pd.Series(mu_post, index=cov.index),
        pd.DataFrame(Sigma_post, index=cov.index, columns=cov.index),
    )


def build_view_matrices(
    tickers: List[str], views: List[dict]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Translate a list of view dicts into (P, Q).

    Each view is one of:
      {"kind": "absolute", "asset": "SPY", "value": 0.08}
      {"kind": "relative", "long": "SPY", "short": "AGG", "value": 0.03}
    where ``value`` is an annualized return (absolute) or out-performance
    (relative).
    """
    idx = {t: i for i, t in enumerate(tickers)}
    rows, q = [], []
    for v in views:
        row = np.zeros(len(tickers))
        if v.get("kind") == "relative":
            if v["long"] not in idx or v["short"] not in idx:
                continue
            row[idx[v["long"]]] = 1.0
            row[idx[v["short"]]] = -1.0
        else:
            if v.get("asset") not in idx:
                continue
            row[idx[v["asset"]]] = 1.0
        rows.append(row)
        q.append(float(v["value"]))
    if not rows:
        return None, None
    return np.array(rows), np.array(q)


# --------------------------------------------------------------------------- #
# Entropy Pooling (Meucci)
# --------------------------------------------------------------------------- #
def entropy_pooling_posterior(
    returns: pd.DataFrame,
    mean_views: Dict[str, float],
    periods_per_year: float = 252.0,
    prior: Optional[np.ndarray] = None,
) -> Tuple[pd.Series, pd.DataFrame, np.ndarray]:
    """Reweight historical scenarios to honor expected-return views.

    Solves the minimum-relative-entropy (I-projection) problem
        min_q  Σ q_i ln(q_i / p_i)   s.t.  E_q[r_k] = v_k,  Σ q_i = 1
    via its convex dual over the (small) vector of Lagrange multipliers λ:
        G(λ) = ln( Σ_i p_i exp(Σ_k λ_k f_ki) ) − Σ_k λ_k v_k
    with f_ki the (daily) return of asset k in scenario i and v_k the daily
    target. The reweighted probabilities q then imply revised moments, which
    are re-annualized for downstream mean-variance optimization.

    Returns (posterior mean, posterior covariance, scenario probabilities).
    """
    R = returns.dropna()
    tickers = list(R.columns)
    T = len(R)
    if prior is None:
        prior = np.repeat(1.0 / T, T)

    view_assets = [t for t in mean_views if t in tickers]
    if not view_assets:
        mu = (R.values * prior[:, None]).sum(axis=0) * periods_per_year
        cov = _weighted_cov(R.values, prior) * periods_per_year
        return (
            pd.Series(mu, index=tickers),
            pd.DataFrame(cov, index=tickers, columns=tickers),
            prior,
        )

    F = R[view_assets].values  # T x k feature matrix (daily returns)
    v = np.array([mean_views[a] / periods_per_year for a in view_assets])  # daily targets
    log_prior = np.log(prior)

    def dual(lam):
        """Convex dual G(λ) = logΣ p·exp(Fλ) − λ·v, with gradient E_q[F] − v."""
        scores = F @ lam + log_prior
        c = scores.max()
        exp_scores = np.exp(scores - c)
        Z = exp_scores.sum()
        q = exp_scores / Z
        value = (c + np.log(Z)) - lam @ v
        grad = q @ F - v
        return value, grad

    res = minimize(
        dual,
        np.zeros(len(view_assets)),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-14},
    )
    lam = res.x
    scores = F @ lam + log_prior
    exp_scores = np.exp(scores - scores.max())
    q = exp_scores / exp_scores.sum()

    mu = (R.values * q[:, None]).sum(axis=0) * periods_per_year
    cov = _weighted_cov(R.values, q) * periods_per_year
    return (
        pd.Series(mu, index=tickers),
        pd.DataFrame(cov, index=tickers, columns=tickers),
        q,
    )


def _weighted_cov(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = w / w.sum()
    mean = (X * w[:, None]).sum(axis=0)
    Xc = X - mean
    return (Xc * w[:, None]).T @ Xc


# --------------------------------------------------------------------------- #
# unified dispatcher
# --------------------------------------------------------------------------- #
def efficient_frontier(
    mu: pd.Series, cov: pd.DataFrame, n_points: int = 40, weight_bounds=(0.0, 1.0)
) -> pd.DataFrame:
    """Trace the mean-variance efficient frontier between min-vol and max-return.

    Returns a DataFrame with columns [ret, vol] for each target return level.
    """
    lo = float(portfolio_return(min_volatility(cov, weight_bounds).values, mu.values))
    hi = float(mu.max())
    targets = np.linspace(lo, hi, n_points)
    out = []
    for t in targets:
        w = target_return(mu, cov, t, weight_bounds)
        out.append(
            {
                "ret": portfolio_return(w.values, mu.values),
                "vol": portfolio_vol(w.values, cov.values),
            }
        )
    return pd.DataFrame(out)


def optimize(
    method: str,
    *,
    mu: Optional[pd.Series] = None,
    cov: Optional[pd.DataFrame] = None,
    rf: float = 0.0,
    weight_bounds=(0.0, 1.0),
    target: Optional[float] = None,
    budgets: Optional[np.ndarray] = None,
    kappa: float = 1.0,
    n_obs: int = 252,
) -> pd.Series:
    """Run one of the moment-based optimizers. (BL/EP transform moments first
    via their *_posterior helpers, then call this with a base method.)"""
    if method == EQUAL_WEIGHT:
        return equal_weight(cov.index)
    if method == INVERSE_VOL:
        return inverse_volatility(cov)
    if method == ERC:
        return equal_risk_contribution(cov)
    if method == RISK_BUDGET:
        return risk_budgeting(cov, budgets)
    if method == MIN_VOL:
        return min_volatility(cov, weight_bounds)
    if method == MAX_SHARPE:
        return max_sharpe(mu, cov, rf, weight_bounds)
    if method == TARGET_RETURN:
        return target_return(mu, cov, target, weight_bounds)
    if method == ROBUST:
        return robust_portfolio(mu, cov, rf, kappa, n_obs, weight_bounds)
    raise ValueError(f"optimize() does not handle method={method!r} directly.")
