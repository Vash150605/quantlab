"""Mean, covariance and correlation: slow literal versions and fast vectorised ones."""
from __future__ import annotations

import numpy as np


def _check(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n observations x p variables)")
    if X.shape[0] < 2:
        raise ValueError("need at least 2 observations")
    return X


def cov_loops(X: np.ndarray) -> np.ndarray:
    """Sample covariance straight from the definition.

    S[i, j] = 1/(n-1) * sum_t (X[t, i] - mean_i) * (X[t, j] - mean_j)
    Example: X = [[1, 2], [3, 4], [5, 9]] gives [[4, 7], [7, 13]].
    """
    X = _check(X)
    n, p = X.shape
    mean = [sum(X[t, i] for t in range(n)) / n for i in range(p)]
    S = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            S[i, j] = sum((X[t, i] - mean[i]) * (X[t, j] - mean[j]) for t in range(n)) / (n - 1)
    return S


def cov_vec(X: np.ndarray) -> np.ndarray:
    """Same quantity in matrix form: S = Xc' Xc / (n-1)."""
    X = _check(X)
    Xc = X - X.mean(axis=0)
    return Xc.T @ Xc / (len(X) - 1)


def corr_from_cov(S: np.ndarray) -> np.ndarray:
    """Correlation matrix: S[i, j] / (sd_i * sd_j)."""
    S = np.asarray(S, dtype=float)
    sd = np.sqrt(np.diag(S))
    if np.any(sd == 0):
        raise ValueError("a variable has zero variance; correlation is undefined")
    return S / np.outer(sd, sd)