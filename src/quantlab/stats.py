"""Mean, covariance and correlation: a slow version (loops) and a fast version (matrices)."""
from __future__ import annotations

import numpy as np


def _check(X: np.ndarray) -> np.ndarray:
    """Make sure X is a 2-D float array with at least 2 rows."""
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

    # mean of each column
    mean = []
    for i in range(p):
        total = 0.0
        for t in range(n):
            total += X[t, i]
        mean.append(total / n)

        # covariance of every pair of columns
    S = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            total = 0.0
            for t in range(n):
                total += (X[t, i] - mean[i]) * (X[t, j] - mean[j])
            S[i, j] = total / (n - 1)
    return S

def cov_vec(X: np.ndarray) -> np.ndarray:
    """Same answer with matrices: S = Xc' Xc / (n-1), where Xc is X minus its column means."""
    X = _check(X)
    centred = X - X.mean(axis=0)
    return centred.T @ centred / (len(X) - 1)


def corr_from_cov(S: np.ndarray) -> np.ndarray:
    """Correlation matrix: S[i, j] / (sd_i * sd_j)."""
    S = np.asarray(S, dtype=float)
    sd = np.sqrt(np.diag(S))
    if np.any(sd == 0):
        raise ValueError("a variable has zero variance; correlation is undefined")
    return S / np.outer(sd, sd)