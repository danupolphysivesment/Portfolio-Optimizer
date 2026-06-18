"""Investable universe: liquid ETFs grouped into three asset classes.

Every ticker here is a real, liquid, US-listed ETF that yfinance can price with a
long history. The universe is intentionally broad enough to build a diversified
multi-asset portfolio while staying small enough to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

EQUITY = "Equity"
FIXED_INCOME = "Fixed Income"
ALTERNATIVES = "Alternatives"

ASSET_CLASSES: List[str] = [EQUITY, FIXED_INCOME, ALTERNATIVES]


@dataclass(frozen=True)
class Asset:
    ticker: str
    name: str
    asset_class: str


# At least three ETFs per asset class (15 total).
ASSETS: List[Asset] = [
    # ---- Equity -------------------------------------------------------------
    Asset("SPY", "S&P 500 — US large cap", EQUITY),
    Asset("QQQ", "Nasdaq 100 — US growth / tech", EQUITY),
    Asset("IWM", "Russell 2000 — US small cap", EQUITY),
    Asset("VEA", "Developed markets ex-US", EQUITY),
    Asset("VWO", "Emerging markets", EQUITY),
    # ---- Fixed Income -------------------------------------------------------
    Asset("AGG", "US aggregate bond", FIXED_INCOME),
    Asset("TLT", "20+ year Treasury", FIXED_INCOME),
    Asset("IEF", "7-10 year Treasury", FIXED_INCOME),
    Asset("LQD", "Investment-grade corporate", FIXED_INCOME),
    Asset("HYG", "High-yield corporate", FIXED_INCOME),
    Asset("TIP", "Inflation-protected Treasury (TIPS)", FIXED_INCOME),
    # ---- Alternatives -------------------------------------------------------
    Asset("GLD", "Gold", ALTERNATIVES),
    Asset("DBC", "Broad commodities", ALTERNATIVES),
    Asset("VNQ", "US real estate (REITs)", ALTERNATIVES),
    Asset("IGF", "Global infrastructure", ALTERNATIVES),
]

ASSET_BY_TICKER: Dict[str, Asset] = {a.ticker: a for a in ASSETS}


def assets_in_class(asset_class: str) -> List[Asset]:
    return [a for a in ASSETS if a.asset_class == asset_class]


def label(ticker: str) -> str:
    a = ASSET_BY_TICKER.get(ticker)
    return f"{ticker} — {a.name}" if a else ticker


def asset_class_of(ticker: str) -> str:
    a = ASSET_BY_TICKER.get(ticker)
    return a.asset_class if a else "Unknown"


def default_selection() -> List[str]:
    """A sensible diversified starter basket spanning all three classes."""
    return ["SPY", "VEA", "VWO", "AGG", "TLT", "LQD", "GLD", "VNQ"]
