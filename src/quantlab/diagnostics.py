"""Measure how far returns are from a Normal distribution."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf


def moment_table(returns: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    """Annualised mean and volatility, plus skewness and excess kurtosis, per column."""
    return pd.DataFrame(
        {
            "mean_ann": returns.mean() * periods,
            "vol_ann": returns.std() * np.sqrt(periods),
            "skew": returns.apply(stats.skew),
            "excess_kurt": returns.apply(stats.kurtosis),  # a Normal has 0
        }
    )


def jarque_bera_table(returns: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Jarque-Bera test per column. Null hypothesis: the data are Normal."""
    rows = {}
    for col in returns.columns:
        stat, p_value = stats.jarque_bera(returns[col].dropna())
        rows[col] = {"JB": stat, "p_value": p_value, "reject": p_value < alpha}
    return pd.DataFrame.from_dict(rows, orient="index")


def tail_counts(returns: pd.DataFrame, k: float = 4.0) -> pd.DataFrame:
    """Days beyond k standard deviations, compared with what a Normal would give."""
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

    acf_ret / acf_sq     : lag-1 autocorrelation of returns / squared returns
    lb_p_ret / lb_p_sq   : Ljung-Box p-value over the first `lags` lags
    A small p-value means the series is autocorrelated.
    """
    rows = {}
    for col in returns.columns:
        x = returns[col].dropna()
        squared = x**2
        rows[col] = {
            "acf_ret": acf(x, nlags=1)[1],
            "acf_sq": acf(squared, nlags=1)[1],
            "lb_p_ret": acorr_ljungbox(x, lags=[lags])["lb_pvalue"].iloc[0],
            "lb_p_sq": acorr_ljungbox(squared, lags=[lags])["lb_pvalue"].iloc[0],
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def fit_t(returns: pd.DataFrame) -> pd.DataFrame:
    """Fit a Student-t to each column: degrees of freedom, location, scale.

    A small df means fat tails (a Normal is the limit as df grows very large).
    """
    rows = {}
    for col in returns.columns:
        df, loc, scale = stats.t.fit(returns[col].dropna())
        rows[col] = {"df": df, "loc": loc, "scale": scale}
    return pd.DataFrame.from_dict(rows, orient="index")


def avg_pairwise_corr(returns: pd.DataFrame) -> float:
    """Average of the off-diagonal entries of the correlation matrix."""
    C = returns.corr().to_numpy()
    p = len(C)
    off_diagonal_sum = C.sum() - p      # remove the p ones on the diagonal
    return float(off_diagonal_sum / (p * (p - 1)))


def corr_by_period(returns: pd.DataFrame, periods: dict) -> pd.Series:
    """Average pairwise correlation inside each (start, end) period."""
    result = {}
    for name, (start, end) in periods.items():
        result[name] = avg_pairwise_corr(returns.loc[start:end])
    return pd.Series(result)