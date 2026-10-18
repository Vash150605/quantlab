import numpy as np
import pytest

from quantlab.optimizers import (
    frontier,
    min_variance_weights,
    portfolio_stats,
    solve_numeric,
    tangency_weights,
    target_return_weights,
)


def _problem(p=6, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((p, p))
    cov = A @ A.T / p + 0.1 * np.eye(p)  # symmetric, positive definite
    mu = rng.uniform(0.02, 0.15, p)
    return mu, cov


def test_min_variance_by_hand():
    w = min_variance_weights(np.diag([0.04, 0.01]))
    np.testing.assert_allclose(w, [0.2, 0.8])


def test_min_variance_sums_to_one():
    _, cov = _problem()
    assert np.isclose(min_variance_weights(cov).sum(), 1.0)


def test_target_return_constraints_hold():
    mu, cov = _problem()
    w = target_return_weights(mu, cov, 0.08)
    assert abs(w.sum() - 1) < 1e-10
    assert abs(w @ mu - 0.08) < 1e-10


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_closed_form_equals_solver(seed):
    mu, cov = _problem(seed=seed)
    m = float(np.median(mu))
    np.testing.assert_allclose(
        target_return_weights(mu, cov, m), solve_numeric(mu, cov, m), atol=1e-5
    )


def test_min_variance_is_leftmost_point_of_frontier():
    mu, cov = _problem()
    w0 = min_variance_weights(cov)
    m0 = w0 @ mu
    assert np.allclose(target_return_weights(mu, cov, m0), w0)
    vols, _ = frontier(mu, cov, np.linspace(m0 - 0.05, m0 + 0.05, 21))
    assert np.isclose(vols.min(), np.sqrt(w0 @ cov @ w0), atol=1e-6)


def test_tangency_has_highest_sharpe():
    mu, cov = _problem()
    rf = 0.01
    w = tangency_weights(mu, cov, rf)
    sharpe = lambda x: (x @ mu - rf) / np.sqrt(x @ cov @ x)
    rng = np.random.default_rng(5)
    for _ in range(200):
        other = rng.standard_normal(len(mu))
        other /= other.sum()
        assert sharpe(w) >= sharpe(other) - 1e-9


def test_long_only_solver_has_no_negative_weights():
    mu, cov = _problem()
    w = solve_numeric(mu, cov, float(np.median(mu)), long_only=True)
    assert w.min() > -1e-8 and abs(w.sum() - 1) < 1e-8


def test_singular_covariance_raises():
    cov = np.array([[1.0, 1.0], [1.0, 1.0]])  # two identical assets
    with pytest.raises(np.linalg.LinAlgError):
        min_variance_weights(cov)


def test_portfolio_stats():
    mu, vol = portfolio_stats(np.array([0.5, 0.5]), np.array([0.10, 0.20]), np.diag([0.04, 0.04]))
    assert np.isclose(mu, 0.15) and np.isclose(vol, np.sqrt(0.02))