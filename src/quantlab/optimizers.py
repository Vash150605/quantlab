"""Markowitz portfolios: closed forms from Lagrange multipliers, and a numerical check.

Notation: mu = expected returns (p,), cov = covariance matrix (p, p), w = weights (p,).
The closed forms allow short positions (negative weights); there is no long-only constraint.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Global minimum-variance portfolio: w = S^-1 1 / (1' S^-1 1).

    Example: cov = diag(0.04, 0.01) gives w = [0.2, 0.8].
    """
    x = np.linalg.solve(cov, np.ones(len(cov)))  # solve, don't invert
    return x / x.sum()


def target_return_weights(mu: np.ndarray, cov: np.ndarray, m: float) -> np.ndarray:
    """Minimum variance subject to sum(w) = 1 and mu'w = m.

    w = lam * S^-1 1 + gam * S^-1 mu, with A = 1'S^-1 1, B = 1'S^-1 mu, C = mu'S^-1 mu,
    D = A*C - B^2, lam = (C - B*m)/D, gam = (A*m - B)/D.
    """
    ones = np.ones(len(mu))
    Si1, Sim = np.linalg.solve(cov, ones), np.linalg.solve(cov, mu)
    A, B, C = ones @ Si1, ones @ Sim, mu @ Sim
    D = A * C - B**2
    lam, gam = (C - B * m) / D, (A * m - B) / D
    return lam * Si1 + gam * Sim


def tangency_weights(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0) -> np.ndarray:
    """Maximum-Sharpe portfolio: w proportional to S^-1 (mu - rf), scaled to sum to 1."""
    x = np.linalg.solve(cov, mu - rf)
    return x / x.sum()


def solve_numeric(mu: np.ndarray, cov: np.ndarray, m: float, long_only: bool = False) -> np.ndarray:
    """Same problem as target_return_weights, solved by scipy (SLSQP). Used as a check."""
    p = len(mu)
    cons = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},
        {"type": "eq", "fun": lambda w: w @ mu - m},
    ]
    bounds = [(0.0, 1.0)] * p if long_only else None
    res = minimize(
        lambda w: w @ cov @ w,
        np.ones(p) / p,
        jac=lambda w: 2 * cov @ w,
        constraints=cons,
        bounds=bounds,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"solver failed: {res.message}")
    return res.x


def portfolio_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    """Return (expected return, volatility) of a portfolio."""
    return float(w @ mu), float(np.sqrt(w @ cov @ w))


def frontier(mu: np.ndarray, cov: np.ndarray, targets: np.ndarray):
    """Efficient frontier: for each target return, the minimum-variance weights.

    Returns (vols, weights) with weights of shape (len(targets), p).
    """
    W = np.array([target_return_weights(mu, cov, m) for m in targets])
    vols = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    return vols, W

def min_variance_long_only(cov: np.ndarray) -> np.ndarray:
    """Minimum variance with sum(w) = 1 and 0 <= w <= 1. No closed form; solved by SLSQP."""
    p = len(cov)
    res = minimize(
        lambda w: w @ cov @ w,
        np.ones(p) / p,
        jac=lambda w: 2 * cov @ w,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        bounds=[(0.0, 1.0)] * p,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"solver failed: {res.message}")
    return np.clip(res.x, 0.0, None) / np.clip(res.x, 0.0, None).sum()