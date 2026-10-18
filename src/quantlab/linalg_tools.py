"""Least squares three ways, and PCA via the SVD."""
from __future__ import annotations

import numpy as np


def ols_normal(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Normal equations: solve (X'X) b = X'y. Fast, but squares the condition number."""
    return np.linalg.solve(X.T @ X, X.T @ y)


def ols_qr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """QR: X = QR, then solve R b = Q'y. More stable."""
    Q, R = np.linalg.qr(X)
    return np.linalg.solve(R, Q.T @ y)


def ols_svd(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """SVD: X = U S V', so b = V S^-1 U' y. Most robust."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ ((U.T @ y) / s)


def pca_svd(X: np.ndarray):
    """PCA by SVD of the centred data.

    Returns (eigvals, Vt, scores): variance per component, directions (rows of Vt),
    and the data projected on the directions.
    """
    Xc = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    eigvals = s**2 / (len(X) - 1)
    scores = Xc @ Vt.T
    return eigvals, Vt, scores