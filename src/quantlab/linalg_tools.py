"""Least squares three ways, and PCA using the SVD."""
from __future__ import annotations

import numpy as np


def ols_normal(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Normal equations: solve (X'X) b = X'y. Fast, but loses accuracy on awkward data."""
    return np.linalg.solve(X.T @ X, X.T @ y)


def ols_qr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """QR: write X = Q R, then solve R b = Q'y. More accurate than the normal equations."""
    Q, R = np.linalg.qr(X)
    return np.linalg.solve(R, Q.T @ y)


def ols_svd(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """SVD: write X = U S V', then b = V S^-1 U' y. The most robust of the three."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ ((U.T @ y) / s)


def pca_svd(X: np.ndarray):
    """PCA from the SVD of the centred data.

    Returns three things:
      eigvals - variance of each component (largest first)
      Vt      - the directions (one per row)
      scores  - the data written in terms of those directions
    """
    centred = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(centred, full_matrices=False)
    eigvals = s**2 / (len(X) - 1)
    scores = centred @ Vt.T
    return eigvals, Vt, scores