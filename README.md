# QuantLab

**Question:** how non-Normal are real returns, and does mean-variance
optimisation survive estimation error?

## How to reproduce

```
python setup_quantlab.py          # only if starting from scratch
python -m venv .venv
.venv\Scripts\activate            # Mac/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest                            # all tests should pass
python make_figures.py            # downloads data on first run, saves report/figures/*.png
jupyter lab                       # notebooks/ holds the analysis
```

The first call to `load_prices()` downloads data from Yahoo Finance and caches it in
`data/`. Later calls read the cache. Use `load_prices(refresh=True)` to download again.

## Project layout

| Path | Purpose |
|---|---|
| `src/quantlab/universe.py` | tickers, start and end dates |
| `src/quantlab/data_loader.py` | download, validate, cache (raw copy in `data/raw`) |
| `src/quantlab/validation.py` | extreme moves, stale prices, calendar gaps, first valid dates |
| `src/quantlab/returns.py` | log and simple returns, annualisation |
| `src/quantlab/diagnostics.py` | skewness, kurtosis, Jarque-Bera, tails, ACF, Student-t, correlations |
| `src/quantlab/stats.py`, `linalg_tools.py` | covariance, OLS (normal equations, QR, SVD), PCA |
| `src/quantlab/optimizers.py` | closed-form Markowitz, tangency, numerical check, long-only |
| `src/quantlab/covariance.py` | sample, Ledoit-Wolf, EWMA, Marchenko-Pastur |
| `src/quantlab/simulation.py` | estimation-error study with a known true covariance |
| `src/quantlab/backtest.py`, `metrics.py` | walk-forward engine, strategies, performance statistics |
| `notebooks/` | `01a`, `01b`, `01c` stylised facts; `02` maths drills; `03` estimation error and backtest |
| `report/` | `writeup.tex`, `figures/` |

## Conventions

- Prices: adjusted close (`auto_adjust=True`)
- Returns: log for statistics, simple for portfolio arithmetic
- Annualising: mean x 252, volatility x sqrt(252)
- No look-ahead: anything used at date t is knowable at t-1 (tested)
- Random functions take a `seed` and use `np.random.default_rng(seed)`

## Data

11 October 2026: (fill in: see the timestamp of the file in `data/processed`)
Universe: 25 US large-cap stocks (AAPL, MSFT, AMZN, JPM, WFC, BAC, XOM, CVX, PG, KO, PEP, WMT,
HD, MCD, DIS, NKE, UNH, AMGN, GILD, INTC, CSCO, ORCL, CAT, UNP, NEE) and the S&P 500 (^GSPC)
Period: 2000-01-03 to 2025-12-30 (6,538 prices, 6,537 daily returns)
Excluded tickers: none. All 26 series are complete from the first date.

Checks: no stale price runs; one calendar gap over 5 days (2001-09-17, the closure after
11 September 2001); about 60 daily moves beyond 20%, all at dates consistent with known events
(for example AAPL on 2000-09-29), none resembling an unadjusted split. Nothing was removed.

## Results

**Stylised facts.** Excess kurtosis is positive for every series (S&P 500: 10.6; AAPL: 114.6);
Jarque-Bera rejects Normality for all 26. There are 65 to 155 times more 4-sigma days than a
Normal predicts. Student-t fits have 2.2 to 3.6 degrees of freedom. Returns are nearly
uncorrelated (mean lag-1 autocorrelation -0.049) but squared returns are strongly
autocorrelated (0.193). Average pairwise correlation rises from about 0.26-0.30 in calm periods
to 0.63-0.72 in the 2008-09 and 2020 crises.

**Estimation error.** Only 2 to 3 eigenvalues of the stock correlation matrix lie above the
Marchenko-Pastur noise edge. Ledoit-Wolf shrinkage cuts the excess variance of minimum-variance
portfolios from 69% to 34% at n = 60 in simulation, and the benefit vanishes as n grows.

**Backtest** (window 504 days, monthly rebalancing, 10 bps costs, net):

| Strategy | Return | Vol. | Sharpe | Max DD | Turnover/yr |
|---|---|---|---|---|---|
| 1/N | 14.6% | 18.6% | 0.823 | -43.8% | 0.58 |
| MinVar, sample | 7.5% | 14.5% | 0.571 | -40.2% | 3.30 |
| MinVar, Ledoit-Wolf | 8.0% | 14.4% | 0.609 | -37.5% | 2.83 |
| MinVar, LW long-only | 9.5% | 14.5% | 0.698 | -35.3% | 1.47 |
| Shrunk mean-variance | 11.0% | 16.3% | 0.725 | -37.5% | 8.43 |

1/N ranks first in all four sensitivity settings. Ledoit-Wolf beats the sample covariance in
all four, and long-only beats unconstrained. Sharpe standard errors are about 0.22 to 0.24, so
the differences are within sampling noise. Shrunk mean-variance fell back to minimum variance
in 152 of 288 rebalances.

## Limitations

- Survivorship bias: the stocks are today's large companies, so failed firms are missing.
- One market, 25 stocks, one 26-year period.
- The simulation uses Normal data and a constant one-factor covariance.
- Sharpe ratios have large standard errors; no paired significance test was run.
- Window, rebalancing, cost and shrinkage settings were fixed in advance; see the sensitivity table.
- Costs are proportional to turnover only (no market impact, short borrowing cost or taxes); the
  risk-free rate is zero.
