"""Walk-forward backtest with transaction costs and turnover. No look-ahead.

Returns passed in are SIMPLE returns (portfolio return = weighted sum of asset returns).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.covariance import ledoit_wolf, sample_cov
from quantlab.optimizers import min_variance_long_only, min_variance_weights, tangency_weights


def walk_forward(
    returns: pd.DataFrame,
    weight_fn,
    window: int = 252,
    rebalance: int = 21,
    cost: float = 0.001,
) -> pd.DataFrame:
    """Run a strategy through history.

    weight_fn(X) -> weights summing to 1, where X is the (window x assets) array of returns.
    On a rebalance day t the weights use ONLY rows t-window .. t-1 and are applied to day t.
    Between rebalances the weights drift with returns. Turnover at a rebalance is
    sum|w_new - w_drifted|, and the cost (cost x turnover) is deducted from that day's return.

    Output (indexed by date, starting after the first window):
    ret = net of costs, gross = before costs, turnover, cost.
    """
    R = returns.to_numpy()
    n, p = R.shape
    gross = np.full(n, np.nan)
    net = np.full(n, np.nan)
    turn = np.zeros(n)
    w = None
    for t in range(window, n):
        if (t - window) % rebalance == 0:
            w_new = weight_fn(R[t - window : t])            # rows up to t-1 only
            turn[t] = np.abs(w_new - (w if w is not None else 0.0)).sum()
            w = w_new
        gross[t] = w @ R[t]
        net[t] = gross[t] - cost * turn[t]
        w = w * (1 + R[t])                                  # drift with the day's returns
        w = w / w.sum()
    out = pd.DataFrame(
        {"ret": net, "gross": gross, "turnover": turn, "cost": cost * turn},
        index=returns.index,
    )
    return out.iloc[window:]


# ---- strategies: each maps a (window x assets) array of simple returns to weights ----

def equal_weight(X: np.ndarray) -> np.ndarray:
    """1/N: no estimation at all."""
    return np.ones(X.shape[1]) / X.shape[1]


def minvar_sample(X: np.ndarray) -> np.ndarray:
    """Minimum variance from the sample covariance (shorting allowed)."""
    return min_variance_weights(sample_cov(X))


def minvar_lw(X: np.ndarray) -> np.ndarray:
    """Minimum variance from the Ledoit-Wolf covariance (shorting allowed)."""
    return min_variance_weights(ledoit_wolf(X)[0])


def minvar_lw_long_only(X: np.ndarray) -> np.ndarray:
    """Minimum variance from the Ledoit-Wolf covariance, long-only."""
    return min_variance_long_only(ledoit_wolf(X)[0])


def make_shrunk_mv(k: float = 0.5, rf: float = 0.0, max_gross: float = 3.0):
    """Tangency-type weights with Ledoit-Wolf covariance and shrunk means.

    Means are pulled toward their cross-sectional average: mu_s = (1-k) mu + k mean(mu).
    Because the tangency weights are scaled by 1' S^-1 (mu_s - rf), they explode when that
    sum is close to zero and are invalid when it is not positive. So the strategy falls back
    to minimum variance when the sum is not positive or when the gross exposure
    sum|w| would exceed max_gross. The returned function counts its calls and fallbacks in
    `.stats`. These rules were fixed before looking at any real-data results.
    """
    stats = {"calls": 0, "fallbacks": 0}

    def shrunk_mv(X: np.ndarray) -> np.ndarray:
        stats["calls"] += 1
        cov = ledoit_wolf(X)[0]
        mu = X.mean(axis=0)
        mu_s = (1 - k) * mu + k * mu.mean()
        x = np.linalg.solve(cov, mu_s - rf)
        if x.sum() > 0:
            w = x / x.sum()
            if np.abs(w).sum() <= max_gross:
                return w
        stats["fallbacks"] += 1
        return min_variance_weights(cov)

    shrunk_mv.stats = stats
    return shrunk_mv