"""Covariance estimators and the Marchenko-Pastur reference distribution."""
from __future__ import annotations

import numpy as np

from quantlab.stats import cov_vec


def sample_cov(X: np.ndarray) -> np.ndarray:
    """Sample covariance, Xc'Xc / (n-1)."""
    return cov_vec(X)


def ledoit_wolf_loop(X: np.ndarray):
    """Ledoit-Wolf shrinkage toward m*I, written with a literal loop over t (slow).

    With demeaned rows x_t and S = (1/n) sum x_t x_t' (note 1/n):
    m = tr(S)/p,  d2 = ||S - mI||^2,  b2_bar = (1/n^2) sum_t ||x_t x_t' - S||^2,
    b2 = min(b2_bar, d2),  delta = b2/d2,  Sigma* = delta*m*I + (1-delta)*S,
    where ||A||^2 = tr(AA')/p.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / n
    m = np.trace(S) / p
    d2 = np.sum((S - m * np.eye(p)) ** 2) / p
    b2_bar = sum(np.sum((np.outer(x, x) - S) ** 2) / p for x in Xc) / n**2
    b2 = min(b2_bar, d2)
    delta = b2 / d2
    return delta * m * np.eye(p) + (1 - delta) * S, delta


def ledoit_wolf(X: np.ndarray):
    """Same as ledoit_wolf_loop, without the loop. Returns (Sigma_star, delta)."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / n
    m = np.trace(S) / p
    d2 = np.sum((S - m * np.eye(p)) ** 2) / p
    b2_bar = (np.sum(np.sum(Xc**2, axis=1) ** 2) / n - np.sum(S**2)) / (n * p)
    b2 = min(b2_bar, d2)
    delta = b2 / d2
    return delta * m * np.eye(p) + (1 - delta) * S, delta


def ewma_cov(X: np.ndarray, lam: float = 0.94, init: int = 30) -> np.ndarray:
    """RiskMetrics EWMA covariance: S_t = lam * S_{t-1} + (1 - lam) * x_t x_t'.

    Starts from the sample covariance of the first `init` rows, then updates through the
    LAST row, so the result is the estimate to use for the next day. Pass only the
    estimation window in a backtest: the demeaning uses the whole array given.
    """
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    S = np.cov(Xc[:init], rowvar=False)
    for t in range(init, len(Xc)):
        S = lam * S + (1 - lam) * np.outer(Xc[t], Xc[t])
    return S


def mp_edges(n: int, p: int, sigma2: float = 1.0) -> tuple[float, float]:
    """Marchenko-Pastur support for the eigenvalues of a sample covariance of pure noise."""
    q = p / n
    return sigma2 * (1 - np.sqrt(q)) ** 2, sigma2 * (1 + np.sqrt(q)) ** 2


def mp_density(x: np.ndarray, n: int, p: int, sigma2: float = 1.0) -> np.ndarray:
    """Marchenko-Pastur density: sqrt((l+ - x)(x - l-)) / (2 pi sigma2 q x) inside the edges."""
    q = p / n
    lo, hi = mp_edges(n, p, sigma2)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    inside = (x > lo) & (x < hi)
    out[inside] = np.sqrt((hi - x[inside]) * (x[inside] - lo)) / (2 * np.pi * sigma2 * q * x[inside])
    return out