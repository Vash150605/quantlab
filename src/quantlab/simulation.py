"""Simulation with a known true covariance: how bad are the estimators, really?"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.covariance import ewma_cov, ledoit_wolf, sample_cov
from quantlab.optimizers import min_variance_weights


def one_factor_cov(p: int = 25, seed: int = 0, factor_vol: float = 0.01) -> np.ndarray:
    """True covariance (daily units): common factor + idiosyncratic noise.

    Sigma = factor_vol^2 * b b' + diag(e^2), with betas b in [0.5, 1.5] and e in [0.008, 0.015].
    """
    rng = np.random.default_rng(seed)
    b = rng.uniform(0.5, 1.5, p)
    e = rng.uniform(0.008, 0.015, p)
    return factor_vol**2 * np.outer(b, b) + np.diag(e**2)


ESTIMATORS = {
    "sample": sample_cov,
    "ledoit_wolf": lambda X: ledoit_wolf(X)[0],
    "ewma": ewma_cov,
}


def excess_variance_study(
    true_cov: np.ndarray,
    ns=(60, 120, 250, 500, 1000),
    reps: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """For each n and estimator, average over `reps` draws of:

    frob:  ||Sigma_hat - Sigma_true||_F
    cond:  condition number of Sigma_hat
    ratio: w(Sigma_hat)' Sigma_true w(Sigma_hat) / (w*' Sigma_true w*), w = min-variance weights.
    All estimators see the SAME draws. A ratio of 1 is perfect.
    """
    p = len(true_cov)
    w_star = min_variance_weights(true_cov)
    v_star = w_star @ true_cov @ w_star
    chol = np.linalg.cholesky(true_cov)
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        acc = {name: {"frob": [], "cond": [], "ratio": []} for name in ESTIMATORS}
        for _ in range(reps):
            X = rng.standard_normal((n, p)) @ chol.T      # rows ~ N(0, true_cov)
            for name, est in ESTIMATORS.items():
                S = est(X)
                w = min_variance_weights(S)
                acc[name]["frob"].append(np.linalg.norm(S - true_cov))
                acc[name]["cond"].append(np.linalg.cond(S))
                acc[name]["ratio"].append(w @ true_cov @ w / v_star)
        for name, d in acc.items():
            rows.append(
                {
                    "n": n,
                    "estimator": name,
                    "frob": np.mean(d["frob"]),
                    "cond": np.mean(d["cond"]),
                    "ratio": np.mean(d["ratio"]),
                    "ratio_se": np.std(d["ratio"], ddof=1) / np.sqrt(reps),
                }
            )
    return pd.DataFrame(rows)