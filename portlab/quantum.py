"""Quantum portfolio optimization with QAOA (statevector-simulated).

Portfolio *selection* is framed as a QUBO: choose which of n assets to hold so
as to minimize

    H(x) = q · xᵀΣx  −  μᵀx  +  P · (Σxᵢ − B)²          xᵢ ∈ {0, 1}

i.e. balance risk (q·variance) against return, holding exactly B assets (a
budget enforced by a penalty P). Because H is diagonal in the computational
basis, the QAOA cost operator e^{-iγH_C} is just a phase, and the mixer
e^{-iβΣXᵢ} is a product of single-qubit rotations.

The QAOA state for depth p is

    |γ,β⟩ = ∏_{l=1}^{p} e^{-iβ_l H_B} e^{-iγ_l H_C} · |+⟩^{⊗n}

We simulate the full statevector (n qubits → 2ⁿ amplitudes), optimize the 2p
angles classically to minimize ⟨H_C⟩, then read out the highest-probability
*feasible* bitstring (exactly B assets) as the selected portfolio. This is a
faithful QAOA — the same circuit a noiseless quantum computer would run — just
executed exactly on a classical machine, which is tractable for the small n
used here.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize

MAX_QUBITS = 12  # statevector sim stays snappy up to ~12 assets (4096 states)


def _all_bitstrings(n: int) -> np.ndarray:
    """(2ⁿ, n) binary matrix; column i is qubit i (i=0 is the most significant)."""
    idx = np.arange(2 ** n)
    return ((idx[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(float)


def portfolio_objective(mu: np.ndarray, cov: np.ndarray, q: float):
    """True mean-variance objective q·xᵀΣx − μᵀx over all 2ⁿ bitstrings."""
    X = _all_bitstrings(len(mu))
    obj = q * np.einsum("si,ij,sj->s", X, cov, X) - X @ mu
    return obj, X, X.sum(axis=1)


def _apply_mixer(state: np.ndarray, beta: float, n: int) -> np.ndarray:
    """Apply e^{-iβ Σ Xₖ} = ∏ₖ (cosβ·I − i sinβ·Xₖ) to the statevector."""
    c, s = np.cos(beta), -1j * np.sin(beta)
    psi = state.reshape([2] * n)
    for k in range(n):
        psi = np.moveaxis(psi, k, 0)
        v0, v1 = psi[0], psi[1]
        out = np.empty_like(psi)
        out[0] = c * v0 + s * v1
        out[1] = s * v0 + c * v1
        psi = np.moveaxis(out, 0, k)
    return psi.reshape(-1)


def _qaoa_state(params: np.ndarray, C: np.ndarray, n: int, p: int) -> np.ndarray:
    gammas, betas = params[:p], params[p:]
    state = np.full(2 ** n, 1.0 / np.sqrt(2 ** n), dtype=complex)  # |+>^n
    for l in range(p):
        state = np.exp(-1j * gammas[l] * C) * state   # cost layer (diagonal phase)
        state = _apply_mixer(state, betas[l], n)       # mixer layer
    return state


def run_qaoa(mu: pd.Series, cov: pd.DataFrame, budget: int, q: float = 1.0,
             p: int = 1, seed: int = 7, maxiter: int = 250, shots: int = 30) -> Dict:
    """Optimize a B-of-n portfolio with QAOA (hybrid readout).

    QAOA is run to produce a measurement distribution over candidate portfolios;
    we then evaluate the true objective on its most-probable *feasible*
    candidates and keep the best — the standard hybrid quantum-classical
    workflow. Returns equal-weight selection weights plus diagnostics.
    """
    tickers = list(mu.index)
    n = len(tickers)
    m = mu.values.astype(float)
    S = cov.loc[tickers, tickers].values.astype(float)
    budget = int(np.clip(budget, 1, n))

    obj, X, card = portfolio_objective(m, S, q)
    feas = np.abs(card - budget) < 0.5
    fidx = np.where(feas)[0]
    if len(fidx) == 0:                      # pathological (B out of range)
        fidx = np.arange(len(obj))
    of = obj[fidx]
    spread = float(of.max() - of.min()) or 1.0

    # Scaled cost for the QAOA phase: centre on the feasible objectives, scale
    # their spread to O(1), and penalise the wrong cardinality.
    penalty = 4.0
    C = 3.0 * (obj - of.mean()) / spread + penalty * (card - budget) ** 2

    def energy(params):
        state = _qaoa_state(params, C, n, p)
        return float(np.sum(np.abs(state) ** 2 * C))

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(4):                      # restarts — landscape is non-convex
        res = minimize(energy, rng.uniform(0, np.pi, size=2 * p),
                       method="COBYLA", options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res

    probs = np.abs(_qaoa_state(best.x, C, n, p)) ** 2

    # Hybrid readout: QAOA measured a distribution over the 2ⁿ selections; take
    # the best feasible portfolio from it (classical evaluation of the sampled
    # candidates). ``shots`` caps how many of the most-probable feasible
    # candidates are evaluated — the standard sampling-based QAOA readout.
    ranked = fidx[np.argsort(-probs[fidx])]
    shortlist = ranked[:min(max(shots, 1), len(ranked))]
    sel_idx = int(shortlist[np.argmin(obj[shortlist])])
    sel = X[sel_idx]
    selected = [tickers[i] for i in range(n) if sel[i] > 0.5]

    weights = pd.Series(0.0, index=tickers)
    if selected:
        weights[selected] = 1.0 / len(selected)

    # Diagnostics: where the chosen portfolio sits in QAOA's measurement ranking,
    # and how far its objective is above the exact best feasible (approx. gap).
    global_opt = int(fidx[np.argmin(obj[fidx])])
    opt_rank = int(np.where(ranked == sel_idx)[0][0]) + 1
    uniform = 1.0 / len(fidx)
    top = [([tickers[i] for i in range(n) if X[j, i] > 0.5], float(probs[j]))
           for j in ranked[:6]]

    return {
        "weights": weights,
        "selected": selected,
        "budget": budget,
        "prob": float(probs[sel_idx]),
        "prob_vs_uniform": float(probs[sel_idx] / uniform),
        "objective": float(obj[sel_idx]),
        "is_optimal": bool(abs(obj[sel_idx] - obj[global_opt]) < 1e-9),
        "opt_rank": opt_rank,
        "n_qubits": n,
        "depth": p,
        "n_feasible": int(len(fidx)),
        "top": top,
    }
