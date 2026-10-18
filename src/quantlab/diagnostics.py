"""Measure how far returns are from a Normal distribution."""
from __future__ import annotations
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox

import numpy as np
import pandas as pd
from scipy import stats


def moment_table(returns: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    """Annualised mean and vol, plus skewness and excess kurtosis per column."""
    return pd.DataFrame(
        {
            "mean_ann": returns.mean() * periods,
            "vol_ann": returns.std() * np.sqrt(periods),
            "skew": returns.apply(stats.skew),
            "excess_kurt": returns.apply(stats.kurtosis),  # Normal = 0
        }
    )


def jarque_bera_table(returns: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Jarque-Bera statistic and p-value per column. Null: the data are Normal."""
    rows = {}
    for col in returns.columns:
        stat, p = stats.jarque_bera(returns[col].dropna())
        rows[col] = {"JB": stat, "p_value": p, "reject": p < alpha}
    return pd.DataFrame.from_dict(rows, orient="index")


def tail_counts(returns: pd.DataFrame, k: float = 4.0) -> pd.DataFrame:
    """Days beyond k standard deviations, versus the count a Normal would give."""
    z = (returns - returns.mean()) / returns.std()
    observed = (z.abs() > k).sum()
    expected = 2 * stats.norm.sf(k) * len(returns)
    out = pd.DataFrame({"observed": observed, "expected": expected})
    out["ratio"] = out["observed"] / out["expected"]
    return out

def rolling_vol(returns: pd.DataFrame, window: int = 21, periods: int = 252) -> pd.DataFrame:
    """Annualised volatility over a moving window (21 trading days is about a month)."""
    return returns.rolling(window).std() * np.sqrt(periods)


def autocorr_table(returns: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
    """Autocorrelation of returns and of squared returns, with Ljung-Box p-values.

    acf_ret / acf_sq: lag-1 autocorrelation of returns / squared returns.
    lb_p_ret / lb_p_sq: Ljung-Box p-value over the first `lags` lags.
    Null of Ljung-Box: no autocorrelation. A small p-value means autocorrelation exists.
    """
    rows = {}
    for col in returns.columns:
        x = returns[col].dropna()
        rows[col] = {
            "acf_ret": acf(x, nlags=1)[1],
            "acf_sq": acf(x**2, nlags=1)[1],
            "lb_p_ret": acorr_ljungbox(x, lags=[lags])["lb_pvalue"].iloc[0],
            "lb_p_sq": acorr_ljungbox(x**2, lags=[lags])["lb_pvalue"].iloc[0],
        }
    return pd.DataFrame.from_dict(rows, orient="index")