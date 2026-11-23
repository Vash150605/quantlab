from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns: ln(P_t / P_{t-1}). Use for statistics."""
    previous_prices = prices.shift(1)
    returns = np.log(prices / previous_prices)
    return returns.dropna(how="all")


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns: P_t / P_{t-1} - 1. Use for portfolio arithmetic."""
    returns = prices.pct_change()
    return returns.dropna(how="all")


def annualise(returns: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    """
    Annualised mean and volatility 
    Annualised mean :  daily mean return * 252
    Annualised volatility : daily volatility * sqrt(252)
    """
    daily_mean = returns.mean()
    daily_volatility = returns.std()

    annual_mean = daily_mean * periods
    annual_volatility = daily_volatility * np.sqrt(periods)
    return pd.DataFrame(
        {
            "mean": annual_mean,
            "vol": annual_volatility,
        }
    )