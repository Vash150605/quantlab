"""Performance statistics for a series of simple daily returns."""
from __future__ import annotations

import numpy as np
import pandas as pd


def wealth(r: pd.Series) -> pd.Series:
    """Growth of 1 unit: cumulative product of (1 + r)."""
    return (1 + r).cumprod()


def drawdown(r: pd.Series) -> pd.Series:
    """Fall of wealth from its running peak, as a negative fraction (0 at a new high)."""
    w = wealth(r)
    return w / w.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    """Worst peak-to-trough fall. Example: r = [0.1, -0.5, 0.2] gives -0.5."""
    return float(drawdown(r).min())


def ann_return(r: pd.Series, periods: int = 252) -> float:
    """Compound annual growth rate: (prod(1 + r)) ** (periods / n) - 1."""
    return float((1 + r).prod() ** (periods / len(r)) - 1)


def ann_vol(r: pd.Series, periods: int = 252) -> float:
    """Annualised volatility: std x sqrt(periods)."""
    return float(r.std(ddof=1) * np.sqrt(periods))


def sharpe(r: pd.Series, periods: int = 252) -> float:
    """Annualised Sharpe ratio with a zero risk-free rate: sqrt(periods) x mean / std."""
    return float(np.sqrt(periods) * r.mean() / r.std(ddof=1))


def summary(results: dict[str, pd.DataFrame], periods: int = 252) -> pd.DataFrame:
    """One row per strategy from walk_forward outputs (columns: ret, gross, turnover)."""
    rows = {}
    for name, df in results.items():
        rows[name] = {
            "ann_return": ann_return(df["ret"], periods),
            "ann_vol": ann_vol(df["ret"], periods),
            "sharpe": sharpe(df["ret"], periods),
            "max_drawdown": max_drawdown(df["ret"]),
            "turnover_per_year": df["turnover"].sum() / len(df) * periods,
            "ann_return_gross": ann_return(df["gross"], periods),
            "sharpe_gross": sharpe(df["gross"], periods),
        }
    return pd.DataFrame.from_dict(rows, orient="index")