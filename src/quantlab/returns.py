from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns: ln(P_t / P_{t-1}). Use for statistics."""
    return np.log(prices / prices.shift(1)).dropna(how="all")


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns: P_t / P_{t-1} - 1. Use for portfolio arithmetic."""
    return prices.pct_change().dropna(how="all")


def annualise(returns: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    """Annualised mean and volatility (mean x 252, vol x sqrt(252))."""
    return pd.DataFrame(
        {
            "mean": returns.mean() * periods,
            "vol": returns.std() * np.sqrt(periods),
        }
    )