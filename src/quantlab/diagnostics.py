"""Measure how far returns are from a Normal distribution."""
from __future__ import annotations

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