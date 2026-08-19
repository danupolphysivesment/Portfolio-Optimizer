"""Performance and risk statistics for a return / equity series."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .estimators import infer_periods_per_year


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    return initial * (1.0 + returns.fillna(0.0)).cumprod()


def cagr(returns: pd.Series, periods_per_year: float | None = None) -> float:
    if returns.empty:
        return 0.0
    ppy = periods_per_year or infer_periods_per_year(returns.index)
    growth = float((1.0 + returns.fillna(0.0)).prod())
    years = len(returns) / ppy
    if years <= 0 or growth <= 0:
        return 0.0
    return growth ** (1.0 / years) - 1.0


def ann_volatility(returns: pd.Series, periods_per_year: float | None = None) -> float:
    ppy = periods_per_year or infer_periods_per_year(returns.index)
    return float(returns.std(ddof=1) * np.sqrt(ppy))


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year=None) -> float:
    ppy = periods_per_year or infer_periods_per_year(returns.index)
    excess = returns - rf / ppy
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(ppy))


def sortino_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year=None) -> float:
    ppy = periods_per_year or infer_periods_per_year(returns.index)
    excess = returns - rf / ppy
    downside = excess[excess < 0]
    dd = np.sqrt((downside**2).mean()) if len(downside) else 0.0
    if dd == 0 or np.isnan(dd):
        return 0.0
    return float(excess.mean() / dd * np.sqrt(ppy))


def drawdown_series(returns: pd.Series) -> pd.Series:
    curve = equity_curve(returns)
    peak = curve.cummax()
    return curve / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    return float(dd.min()) if len(dd) else 0.0


def calmar_ratio(returns: pd.Series, periods_per_year=None) -> float:
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return cagr(returns, periods_per_year) / mdd


def historical_var(returns: pd.Series, level: float = 0.95) -> float:
    if returns.empty:
        return 0.0
    return float(-np.percentile(returns.dropna(), (1 - level) * 100))


def historical_cvar(returns: pd.Series, level: float = 0.95) -> float:
    if returns.empty:
        return 0.0
    cutoff = np.percentile(returns.dropna(), (1 - level) * 100)
    tail = returns[returns <= cutoff]
    return float(-tail.mean()) if len(tail) else 0.0


def summary_stats(
    returns: pd.Series, rf: float = 0.0, periods_per_year: float | None = None
) -> Dict[str, float]:
    ppy = periods_per_year or infer_periods_per_year(returns.index)
    return {
        "Total Return": float((1.0 + returns.fillna(0.0)).prod() - 1.0),
        "CAGR": cagr(returns, ppy),
        "Volatility (ann.)": ann_volatility(returns, ppy),
        "Sharpe": sharpe_ratio(returns, rf, ppy),
        "Sortino": sortino_ratio(returns, rf, ppy),
        "Max Drawdown": max_drawdown(returns),
        "Calmar": calmar_ratio(returns, ppy),
        "VaR 95% (period)": historical_var(returns, 0.95),
        "CVaR 95% (period)": historical_cvar(returns, 0.95),
        "Best Period": float(returns.max()) if len(returns) else 0.0,
        "Worst Period": float(returns.min()) if len(returns) else 0.0,
        "Win Rate": float((returns > 0).mean()) if len(returns) else 0.0,
    }
