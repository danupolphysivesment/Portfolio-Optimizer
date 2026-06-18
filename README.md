# 📈 Quant Portfolio Lab

A Streamlit application for **portfolio construction** and **backtesting** on real
historical ETF data. Built as a teaching / research tool for quantitative
portfolio management.

## What it does

A broad ETF universe (**65 funds**) across three asset classes:

| Asset class   | Coverage |
|---------------|----------|
| **Equity (44)**       | Global (ACWI, URTH, AAXJ), US broad (SPY, QQQM, IWM), **11 US sector SPDRs** (XLK…XLC), **15 STOXX Europe 600 sector + broad** (EXxx.DE), Asia regional (India, Japan TOPIX/Nikkei, Korea, Taiwan, Vietnam, Thailand), developed/EM (VEA, VWO) |
| **Fixed Income (14)** | USD Treasury ladder 1-3y→7-10y (IBTA.L, CBU7.L, CBU0.L), 20y+ (TLT), IG (SDIG.L, IGIB, LQDA.L), HY (SDHY.L, HYG), floating (FLOA.L), MBS (IMBA.L), TIPS (TIP), aggregate (AGG), active (JGIAX) |
| **Alternatives (7)**  | Commodities (CRBL.L, DBC), gold (GLD), oil (USL), REITs (REET, VNQ), infrastructure (IGF) |

Tickers shown are **Yahoo Finance** symbols; the in-app universe table also lists
the original **Bloomberg** ticker, region, and currency for each fund.

### Multi-currency → USD
The list spans US (USD), Xetra (EUR), Tokyo (JPY), and London (USD/GBp) listings.
A sidebar toggle (**on** by default) converts every non-USD series to USD using
daily `EURUSD`/`GBPUSD`/`JPYUSD` FX (GBp pence handled), so cross-currency
portfolios and their covariances are comparable for a USD investor.

### 🧮 Optimizer
Choose any subset of assets and build a portfolio with one of **ten** methods,
all estimated from historical returns:

1. **Equal Weight** — 1/N baseline
2. **Inverse Volatility** — weight ∝ 1/σ
3. **Equal Risk Contribution** — risk parity (convex log-barrier solver)
4. **Risk Budgeting** — custom risk shares per asset
5. **Min Volatility** — global minimum-variance portfolio
6. **Max Sharpe** — tangency portfolio
7. **Target Return** — min-variance at a chosen expected return
8. **Black-Litterman** — equilibrium prior blended with subjective views
9. **Entropy Pooling** — Meucci scenario re-weighting under mean views
10. **Robust (worst-case mean)** — Ceria–Stubbs estimation-error-aware optimization

Outputs: weights, asset-class allocation, **risk-contribution** decomposition,
ex-ante return/vol/Sharpe, and the **efficient frontier** with your portfolio,
the max-Sharpe and min-vol portfolios, and each asset plotted.

### 🧪 Backtester
- **Fixed weights** — hold a target allocation with periodic rebalancing
  (Monthly → Annual or buy-&-hold), transaction costs in bps, weight drift, and
  a benchmark (60/40, SPY, or equal weight).
- **Walk-forward optimization** — re-estimate moments and re-optimize on each
  rebalance date using only a trailing window: an honest **out-of-sample** test
  with no look-ahead bias.

Metrics: CAGR, annualized vol, Sharpe, Sortino, max drawdown, Calmar, VaR/CVaR,
win rate, turnover, and total cost drag. Equity curve, drawdown, and weight-drift
charts included.

### 📚 Asset Universe & 📖 Methods
Reference tabs: explore normalized price history and the return-correlation
heatmap, and read the math behind every optimizer.

## Run it

```bash
cd quant-portfolio-lab
pip install -r requirements.txt
streamlit run app.py
```

The default data source is **Yahoo Finance (live)** via `yfinance`. If you have no
network connection, switch the sidebar **Data source** to *Synthetic (offline
demo)* to explore the app with deterministic simulated data.

## Project layout

```
quant-portfolio-lab/
├── app.py                 # Streamlit UI (sidebar settings + 4 workbenches)
├── requirements.txt
├── .streamlit/config.toml
└── portlab/
    ├── universe.py        # ETF universe across 3 asset classes
    ├── data.py            # yfinance loader + synthetic fallback
    ├── estimators.py      # annualized mean & covariance (sample/EWMA/Ledoit-Wolf)
    ├── optimizers.py      # all 10 construction methods + efficient frontier
    ├── metrics.py         # performance & risk statistics
    └── backtest.py        # fixed-weight & walk-forward engines
```

## Notes & caveats
- Estimates are backward-looking; mean-variance optimizers are sensitive to
  expected-return error — that is exactly why Robust, Black-Litterman, and the
  risk-based methods are included for comparison.
- Covariances are nudged onto the PSD cone for numerical stability.
- This is a research/educational tool, **not investment advice**.
