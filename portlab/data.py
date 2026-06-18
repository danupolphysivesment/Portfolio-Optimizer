"""Price data loading.

Primary source is Yahoo Finance (via yfinance). A deterministic synthetic
generator is provided as an offline fallback so the app stays demonstrable
without a network connection. All public functions return a tidy DataFrame of
adjusted close prices indexed by date with one column per ticker.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

YAHOO = "Yahoo Finance (live)"
SYNTHETIC = "Synthetic (offline demo)"
SOURCES = [YAHOO, SYNTHETIC]


# --------------------------------------------------------------------------- #
# Yahoo Finance
# --------------------------------------------------------------------------- #
def _download_yahoo(tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError("Yahoo Finance returned no data for the requested range.")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:  # single ticker → flat columns
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]

    # Keep only requested tickers that actually came back, in requested order.
    cols = [t for t in tickers if t in close.columns]
    close = close[cols]
    return close


# --------------------------------------------------------------------------- #
# Synthetic fallback (correlated geometric Brownian motion)
# --------------------------------------------------------------------------- #
# Rough, plausible annual drift / vol per asset class for the demo generator.
_SYNTH_PROFILE = {
    "Equity": (0.08, 0.17),
    "Fixed Income": (0.03, 0.06),
    "Alternatives": (0.05, 0.15),
    "Unknown": (0.06, 0.14),
}


def _synthetic_prices(tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
    from .universe import asset_class_of

    rng = np.random.default_rng(12345)
    dates = pd.bdate_range(start=start, end=end)
    n_days = len(dates)
    n = len(tickers)
    if n_days < 2:
        raise RuntimeError("Date range too short for synthetic data.")

    mu = np.array([_SYNTH_PROFILE[asset_class_of(t)][0] for t in tickers])
    vol = np.array([_SYNTH_PROFILE[asset_class_of(t)][1] for t in tickers])

    # Block correlation: assets in the same class are more correlated.
    classes = [asset_class_of(t) for t in tickers]
    corr = np.full((n, n), 0.15)
    for i in range(n):
        for j in range(n):
            if i == j:
                corr[i, j] = 1.0
            elif classes[i] == classes[j]:
                corr[i, j] = 0.6
    # Equities and alternatives carry mild positive cross-correlation.
    chol = np.linalg.cholesky(corr + 1e-6 * np.eye(n))

    dt = 1 / 252
    z = rng.standard_normal((n_days - 1, n)) @ chol.T
    daily = (mu - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z
    log_prices = np.vstack([np.zeros(n), np.cumsum(daily, axis=0)])
    prices = 100.0 * np.exp(log_prices)
    return pd.DataFrame(prices, index=dates, columns=list(tickers))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_prices(
    tickers: Sequence[str],
    start: str,
    end: str,
    source: str = YAHOO,
) -> Tuple[pd.DataFrame, List[str]]:
    """Return (prices, warnings).

    Prices are cleaned: forward-filled within each series, then any column or
    row that is still entirely missing is dropped. Tickers that returned no
    usable data are reported in ``warnings`` and excluded from the result.
    """
    tickers = list(dict.fromkeys(tickers))  # de-dupe, keep order
    warnings: List[str] = []
    if not tickers:
        return pd.DataFrame(), ["No tickers selected."]

    if source == SYNTHETIC:
        prices = _synthetic_prices(tickers, start, end)
    else:
        prices = _download_yahoo(tickers, start, end)

    prices = prices.sort_index()
    # Forward fill small gaps, then require a real first observation.
    prices = prices.ffill()
    missing = [t for t in tickers if t not in prices.columns]
    for t in missing:
        warnings.append(f"No data returned for {t}; excluded.")

    # Drop columns that are still all-NaN, and align on the common window.
    prices = prices.dropna(axis=1, how="all")
    dropped = [t for t in prices.columns if prices[t].isna().all()]
    if dropped:
        warnings.append(f"Dropped empty series: {', '.join(dropped)}")
        prices = prices.drop(columns=dropped)

    # Restrict to rows where every surviving asset has a price (common history).
    prices = prices.dropna(axis=0, how="any")
    if len(prices) < 30:
        warnings.append(
            "Fewer than 30 overlapping observations — estimates will be unreliable."
        )
    return prices, warnings
