"""Macroeconomic impulse-response analysis.

We model the portfolio jointly with a handful of macro-factor proxies in a
vector autoregression (VAR), then trace **orthogonalized impulse responses**:
how the portfolio reacts, period by period, to a one-standard-deviation shock in
each macro factor. Shocks are identified with a Cholesky ordering of the VAR
residual covariance (macro factors first, the portfolio last — i.e. macro shocks
hit the portfolio contemporaneously, not the other way round).

Factors are liquid Yahoo proxies (indices / futures), transformed to stationary
series: returns for price-like factors, first differences for level factors
(yields, VIX).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import data as dataio


@dataclass(frozen=True)
class MacroFactor:
    name: str
    ticker: str
    transform: str   # "return" or "diff"
    note: str


MACRO_FACTORS: List[MacroFactor] = [
    MacroFactor("Equity market", "^GSPC", "return", "S&P 500 return"),
    MacroFactor("Rates (10Y)", "^TNX", "diff", "Δ 10-year Treasury yield"),
    MacroFactor("US dollar", "DX-Y.NYB", "return", "Dollar index return"),
    MacroFactor("Oil", "CL=F", "return", "WTI crude return"),
    MacroFactor("Volatility", "^VIX", "diff", "Δ VIX (risk sentiment)"),
    MacroFactor("Gold", "GC=F", "return", "Gold return"),
]
_BY_NAME = {f.name: f for f in MACRO_FACTORS}
FACTOR_NAMES = [f.name for f in MACRO_FACTORS]


def load_factor_changes(start: str, end: str, source: str
                        ) -> Tuple[pd.DataFrame, List[str]]:
    """Return (daily stationary factor changes, warnings)."""
    tickers = [f.ticker for f in MACRO_FACTORS]
    levels, warns = dataio.load_prices(tickers, start, end, source=source,
                                       convert_usd=False)
    if levels.empty:
        return pd.DataFrame(), warns
    rename = {f.ticker: f.name for f in MACRO_FACTORS if f.ticker in levels.columns}
    levels = levels.rename(columns=rename)
    out = pd.DataFrame(index=levels.index)
    for f in MACRO_FACTORS:
        if f.name not in levels.columns:
            warns.append(f"Macro factor {f.name} ({f.ticker}) unavailable; skipped.")
            continue
        out[f.name] = (levels[f.name].diff() if f.transform == "diff"
                       else levels[f.name].pct_change())
    return out.dropna(how="all"), warns


def _offset(freq: str):
    # Offset objects work across pandas 1.5 → 3.x (no removed string aliases).
    return pd.offsets.MonthEnd() if freq == "Monthly" else pd.offsets.Week(weekday=6)


def build_frame(port_daily: pd.Series, factor_daily: pd.DataFrame,
                freq: str = "Weekly") -> pd.DataFrame:
    """Align portfolio + factor changes and resample. Portfolio is placed last
    so the Cholesky identification treats it as the most endogenous variable."""
    joined = factor_daily.join(port_daily.rename("Portfolio"), how="inner").dropna()
    off = _offset(freq)
    out = pd.DataFrame(index=joined.resample(off).last().index)
    for c in joined.columns:
        transform = "diff" if (c in _BY_NAME and _BY_NAME[c].transform == "diff") else "return"
        if transform == "diff":
            out[c] = joined[c].resample(off).sum()
        else:
            out[c] = (1 + joined[c]).resample(off).prod() - 1
    factor_cols = [c for c in joined.columns if c != "Portfolio"]
    return out[factor_cols + ["Portfolio"]].dropna()


def run_var_irf(frame: pd.DataFrame, horizon: int = 12, maxlags: int = 8,
                lag: Optional[int] = None, port_name: str = "Portfolio") -> Dict:
    """Fit a VAR and return orthogonalized cumulative IRFs, bands, and FEVD."""
    from statsmodels.tsa.api import VAR

    cols = list(frame.columns)
    model = VAR(frame)
    if lag is None:
        try:
            sel = model.select_order(maxlags)
            lag = int(sel.aic) if getattr(sel, "aic", None) else 1
        except Exception:
            lag = 1
    lag = max(1, int(lag))
    res = model.fit(lag)
    irf = res.irf(horizon)

    port_idx = cols.index(port_name)
    factor_cols = [c for c in cols if c != port_name]
    orth_cum = irf.orth_cum_effects  # (H+1, resp_i, shock_j)
    cum = pd.DataFrame({c: orth_cum[:, port_idx, cols.index(c)] for c in factor_cols})

    se = None
    try:
        cse = irf.cum_effect_stderr(orth=True)
        se = pd.DataFrame({c: cse[:, port_idx, cols.index(c)] for c in factor_cols})
    except Exception:
        se = None

    fevd_share = None
    try:
        dec = res.fevd(horizon).decomp[port_idx]  # (H, n)
        fevd_share = pd.Series(dec[horizon - 1], index=cols)
    except Exception:
        fevd_share = None

    return {"lag": lag, "nobs": int(res.nobs), "cum_irf": cum, "cum_se": se,
            "fevd": fevd_share, "cols": cols, "factor_cols": factor_cols,
            "port_name": port_name}


def static_sensitivities(frame: pd.DataFrame, port_name: str = "Portfolio"
                         ) -> Tuple[pd.Series, float]:
    """OLS of portfolio on contemporaneous factor changes. Returns the response
    to a +1-SD move in each factor (β·σ) and the regression R²."""
    X = frame.drop(columns=[port_name])
    y = frame[port_name].values
    A = np.column_stack([np.ones(len(X)), X.values])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    betas = coef[1:]
    yhat = A @ coef
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    impact = betas * X.std(ddof=1).values
    return pd.Series(impact, index=X.columns), float(r2)
