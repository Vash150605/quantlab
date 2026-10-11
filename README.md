# QuantLab

**Question:** how non-Normal are real returns, and does mean-variance
optimisation survive estimation error?

## How to reproduce

```
pip install -e ".[dev]"
pytest
jupyter lab
```

## Conventions

- Prices: adjusted close (`auto_adjust=True`)
- Returns: log for statistics, simple for portfolio arithmetic
- Annualising: mean x 252, volatility x sqrt(252)
- No look-ahead: anything used at date t is knowable at t-1
- Random functions take a `seed` and use `np.random.default_rng(seed)`

## Data

Download date: (fill in)
Universe: (fill in)
Excluded tickers: (fill in)

## Results

(fill in)

## Limitations

(fill in: survivorship bias, one market, one period, ...)
