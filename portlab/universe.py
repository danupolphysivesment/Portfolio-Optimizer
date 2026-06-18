"""Investable universe: ETFs grouped into three asset classes.

Each asset stores its Yahoo Finance ticker (used for pricing), the original
Bloomberg ticker (for reference), the listing currency, a region tag (for
filtering), and a short note. Non-USD listings (Europe/Xetra EUR, Tokyo JPY,
London GBp) are converted to USD at load time so cross-currency portfolios are
coherent — see ``data.load_prices``.

Yahoo symbols were verified to return data and to match the intended fund (e.g.
the iShares STOXX Europe 600 sector ETFs map to the EXxx.DE Xetra lines).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# Asset classes
EQUITY = "Equity"
FIXED_INCOME = "Fixed Income"
ALTERNATIVES = "Alternatives"
ASSET_CLASSES: List[str] = [EQUITY, FIXED_INCOME, ALTERNATIVES]

# Regions (used for the optional picker filter)
WORLD = "World"
US = "US"
EUROPE = "Europe"
ASIA = "Asia"
GLOBAL = "Global"


@dataclass(frozen=True)
class Asset:
    ticker: str          # Yahoo Finance symbol (pricing)
    name: str            # full fund name
    asset_class: str
    region: str
    currency: str        # listing currency: USD / EUR / JPY / GBp
    bb: str              # Bloomberg ticker (reference)
    note: str            # short description


# fmt: off
ASSETS: List[Asset] = [
    # ===== EQUITY · World =================================================
    Asset("ACWI", "iShares MSCI ACWI ETF",            EQUITY, WORLD, "USD", "ACWI US Equity", "MSCI All-Country World"),
    Asset("AAXJ", "iShares MSCI All Country Asia ex-Japan ETF", EQUITY, WORLD, "USD", "AAXJ US Equity", "MSCI Asia ex-Japan"),
    Asset("URTH", "iShares MSCI World ETF",           EQUITY, WORLD, "USD", "URTH US Equity", "MSCI World (Developed)"),

    # ===== EQUITY · US broad ==============================================
    Asset("SPY",  "SPDR S&P 500 ETF Trust",           EQUITY, US, "USD", "SPY US Equity",  "S&P 500"),
    Asset("QQQM", "Invesco NASDAQ 100 ETF",           EQUITY, US, "USD", "QQQM US Equity", "Nasdaq 100"),
    Asset("QQQ",  "Invesco QQQ Trust",                EQUITY, US, "USD", "",               "Nasdaq 100 (QQQ)"),
    Asset("IWM",  "iShares Russell 2000 ETF",         EQUITY, US, "USD", "",               "US Small Cap"),

    # ===== EQUITY · US sectors (SPDR Select) ==============================
    Asset("XLK",  "Technology Select Sector SPDR Fund",     EQUITY, US, "USD", "XLK US Equity", "US Technology"),
    Asset("XLF",  "Financial Select Sector SPDR Fund",      EQUITY, US, "USD", "XLF US Equity", "US Financials"),
    Asset("XLV",  "Health Care Select Sector SPDR Fund",    EQUITY, US, "USD", "XLV US Equity", "US Health Care"),
    Asset("XLY",  "Consumer Discretionary Select SPDR",     EQUITY, US, "USD", "XLY US Equity", "US Consumer Discretionary"),
    Asset("XLP",  "Consumer Staples Select Sector SPDR",    EQUITY, US, "USD", "XLP US Equity", "US Consumer Staples"),
    Asset("XLE",  "Energy Select Sector SPDR Fund",         EQUITY, US, "USD", "XLE US Equity", "US Energy"),
    Asset("XLI",  "Industrial Select Sector SPDR Fund",     EQUITY, US, "USD", "XLI US Equity", "US Industrials"),
    Asset("XLU",  "Utilities Select Sector SPDR Fund",      EQUITY, US, "USD", "XLU US Equity", "US Utilities"),
    Asset("XLB",  "Materials Select Sector SPDR Fund",      EQUITY, US, "USD", "XLB US Equity", "US Materials"),
    Asset("XLRE", "Real Estate Select Sector SPDR Fund",    EQUITY, US, "USD", "XLRE US Equity","US Real Estate"),
    Asset("XLC",  "Communication Services Select SPDR",     EQUITY, US, "USD", "XLC US Equity", "US Communication Services"),

    # ===== EQUITY · Europe (STOXX Europe 600 sectors + broad) =============
    Asset("EXH1.DE", "iShares STOXX Europe 600 Oil & Gas",        EQUITY, EUROPE, "EUR", "SXEPEX GY Equity", "Europe Energy"),
    Asset("EXV1.DE", "iShares STOXX Europe 600 Banks",            EQUITY, EUROPE, "EUR", "SX7PEX GY Equity", "Europe Banks"),
    Asset("EXH5.DE", "iShares STOXX Europe 600 Insurance",        EQUITY, EUROPE, "EUR", "SXIPEX GY Equity", "Europe Insurance"),
    Asset("EXV4.DE", "iShares STOXX Europe 600 Health Care",      EQUITY, EUROPE, "EUR", "SXDPEX GY Equity", "Europe Health Care"),
    Asset("EXV3.DE", "iShares STOXX Europe 600 Technology",       EQUITY, EUROPE, "EUR", "SX8PEX GY Equity", "Europe Technology"),
    Asset("EXH4.DE", "iShares STOXX Europe 600 Industrial G&S",   EQUITY, EUROPE, "EUR", "SXNPEX GY Equity", "Europe Industrials"),
    Asset("EXH7.DE", "iShares STOXX Europe 600 Personal & Household", EQUITY, EUROPE, "EUR", "SXQPEX GY Equity", "Europe Consumer Discretionary"),
    Asset("EXH3.DE", "iShares STOXX Europe 600 Food & Beverage",  EQUITY, EUROPE, "EUR", "SX3PEX GY Equity", "Europe Consumer Staples"),
    Asset("EXV8.DE", "iShares STOXX Europe 600 Construction & Materials", EQUITY, EUROPE, "EUR", "SXOPEX GY Equity", "Europe Materials"),
    Asset("EXH9.DE", "iShares STOXX Europe 600 Utilities",        EQUITY, EUROPE, "EUR", "SX6PEX GY Equity", "Europe Utilities"),
    Asset("EXI5.DE", "iShares STOXX Europe 600 Real Estate",      EQUITY, EUROPE, "EUR", "SREEEX GY Equity", "Europe Real Estate"),
    Asset("EXV2.DE", "iShares STOXX Europe 600 Telecommunications", EQUITY, EUROPE, "EUR", "SXKPEX GY Equity", "Europe Communication Services"),
    Asset("EXSA.DE", "iShares STOXX Europe 600 UCITS ETF",        EQUITY, EUROPE, "EUR", "SXXPIEX GY Equity", "STOXX Europe 600 (broad)"),
    Asset("EXW1.DE", "iShares Core EURO STOXX 50",                EQUITY, EUROPE, "EUR", "SX5EEX GY Equity", "Eurozone Large Cap"),
    Asset("SXRJ.DE", "iShares MSCI EMU Small Cap",                EQUITY, EUROPE, "EUR", "SXRJ GY Equity",  "Eurozone Small Cap"),

    # ===== EQUITY · Asia ==================================================
    Asset("INDA",   "iShares MSCI India ETF",         EQUITY, ASIA, "USD", "INDA US Equity", "India (MSCI)"),
    Asset("INDY",   "iShares India 50 ETF",           EQUITY, ASIA, "USD", "INDY US Equity", "India (Nifty 50)"),
    Asset("1306.T", "NEXT FUNDS TOPIX ETF",           EQUITY, ASIA, "JPY", "1306 JP Equity", "Japan (TOPIX)"),
    Asset("1321.T", "NEXT FUNDS Nikkei 225 ETF",      EQUITY, ASIA, "JPY", "1321 JP Equity", "Japan (Nikkei 225)"),
    Asset("EWJ",    "iShares MSCI Japan ETF",         EQUITY, ASIA, "USD", "EWJ US Equity",  "Japan (MSCI)"),
    Asset("EWY",    "iShares MSCI South Korea ETF",   EQUITY, ASIA, "USD", "EWY US Equity",  "South Korea"),
    Asset("EWT",    "iShares MSCI Taiwan ETF",        EQUITY, ASIA, "USD", "EWT US Equity",  "Taiwan"),
    Asset("VNM",    "VanEck Vietnam ETF",             EQUITY, ASIA, "USD", "VNM US Equity",  "Vietnam"),
    Asset("THD",    "iShares MSCI Thailand ETF",      EQUITY, ASIA, "USD", "THD US Equity",  "Thailand"),

    # Developed/EM broad (retained, complements regional sleeves)
    Asset("VEA", "Vanguard FTSE Developed Markets ETF", EQUITY, WORLD, "USD", "", "Developed ex-US"),
    Asset("VWO", "Vanguard FTSE Emerging Markets ETF",  EQUITY, WORLD, "USD", "", "Emerging Markets"),

    # ===== FIXED INCOME ===================================================
    Asset("IBTA.L", "iShares $ Treasury Bond 1-3yr UCITS ETF",  FIXED_INCOME, US, "USD", "IBTA LN Equity", "UST 1-3y"),
    Asset("CBU7.L", "iShares $ Treasury Bond 3-7yr UCITS ETF",  FIXED_INCOME, US, "USD", "CBU7 LN Equity", "UST 3-7y"),
    Asset("CBU0.L", "iShares $ Treasury Bond 7-10yr UCITS ETF", FIXED_INCOME, US, "USD", "CBU0 LN Equity", "UST 7-10y"),
    Asset("FLOA.L", "iShares $ Floating Rate Bond UCITS ETF",   FIXED_INCOME, US, "USD", "FLOA LN Equity", "US IG Floating Rate"),
    Asset("SDIG.L", "iShares $ Short Duration Corp Bond UCITS", FIXED_INCOME, US, "USD", "SDIG LN Equity", "US IG 0-5y"),
    Asset("IGIB",   "iShares 5-10 Yr IG Corporate Bond ETF",   FIXED_INCOME, US, "USD", "IGIB US Equity", "US IG 5-10y"),
    Asset("LQDA.L", "iShares $ Corp Bond UCITS ETF",           FIXED_INCOME, US, "USD", "LQDA LN Equity", "US IG broad"),
    Asset("SDHY.L", "iShares $ Short Duration HY Corp Bond",    FIXED_INCOME, US, "USD", "SDHY LN Equity", "US HY 0-5y"),
    Asset("IMBA.L", "iShares US Mortgage Backed Securities",    FIXED_INCOME, US, "USD", "IMBA LN Equity", "US Agency MBS"),
    Asset("JGIAX",  "JPMorgan Income Fund",                    FIXED_INCOME, US, "USD", "JGIAX US Equity","JPM Income (active)"),
    # Retained US fixed-income staples (fill duration/inflation gaps)
    Asset("AGG", "iShares Core US Aggregate Bond ETF",   FIXED_INCOME, US, "USD", "", "US Aggregate Bond"),
    Asset("TLT", "iShares 20+ Year Treasury Bond ETF",   FIXED_INCOME, US, "USD", "", "UST 20y+"),
    Asset("TIP", "iShares TIPS Bond ETF",                FIXED_INCOME, US, "USD", "", "US TIPS"),
    Asset("HYG", "iShares iBoxx $ High Yield Corp Bond",  FIXED_INCOME, US, "USD", "", "US HY broad"),

    # ===== ALTERNATIVES ===================================================
    Asset("CRBL.L", "Lyxor Commodities Refinitiv/CoreCommodity CRB TR", ALTERNATIVES, GLOBAL, "GBp", "LYTR GY Equity", "Commodities (CRB)"),
    Asset("GLD",    "SPDR Gold Trust",                   ALTERNATIVES, GLOBAL, "USD", "GLD US Equity", "Gold"),
    Asset("USL",    "United States 12 Month Oil Fund",   ALTERNATIVES, US,     "USD", "USL US Equity", "Oil (12-month)"),
    Asset("REET",   "iShares Global REIT ETF",           ALTERNATIVES, GLOBAL, "USD", "REET US Equity","Global REITs"),
    # Retained alternative staples
    Asset("DBC", "Invesco DB Commodity Index Tracking Fund", ALTERNATIVES, GLOBAL, "USD", "", "Broad Commodities"),
    Asset("VNQ", "Vanguard Real Estate ETF",                 ALTERNATIVES, US,     "USD", "", "US REITs"),
    Asset("IGF", "iShares Global Infrastructure ETF",        ALTERNATIVES, GLOBAL, "USD", "", "Global Infrastructure"),
]
# fmt: on

ASSET_BY_TICKER: Dict[str, Asset] = {a.ticker: a for a in ASSETS}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def assets_in_class(asset_class: str) -> List[Asset]:
    return [a for a in ASSETS if a.asset_class == asset_class]


def all_regions() -> List[str]:
    seen = []
    for a in ASSETS:
        if a.region not in seen:
            seen.append(a.region)
    return seen


def label(ticker: str) -> str:
    a = ASSET_BY_TICKER.get(ticker)
    return f"{ticker} · {a.note}" if a else ticker


def display_name(asset: Asset) -> str:
    """Picker option string; unique per asset (ticker is unique)."""
    return f"{asset.ticker} · {asset.note}"


def asset_class_of(ticker: str) -> str:
    a = ASSET_BY_TICKER.get(ticker)
    return a.asset_class if a else "Unknown"


def currency_of(ticker: str) -> str:
    a = ASSET_BY_TICKER.get(ticker)
    return a.currency if a else "USD"


def default_selection() -> List[str]:
    """A diversified, all-USD-listed starter basket (robust without FX)."""
    return ["SPY", "QQQM", "EWJ", "INDA", "IGIB", "LQDA.L", "GLD", "REET"]
