import numpy as np
import pytest
from sklearn.covariance import LedoitWolf

from quantlab.covariance import (
    ewma_cov,
    ledoit_wolf,
    ledoit_wolf_loop,
    mp_density,
    mp_edges,
    sample_cov,
)


def _trapz(y, x):
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))


def _X(n=200, p=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p)) @ rng.standard_normal((p, p))


def test_loop_equals_vectorised_lw():
    X = _X()
    S1, d1 = ledoit_wolf_loop(X)
    S2, d2 = ledoit_wolf(X)
    np.testing.assert_allclose(S1, S2, atol=1e-12)
    assert np.isclose(d1, d2)


def test_lw_matches_sklearn():
    X = _X(seed=1)
    lw = LedoitWolf().fit(X)
    S, delta = ledoit_wolf(X)
    np.testing.assert_allclose(S, lw.covariance_, atol=1e-8)
    assert np.isclose(delta, lw.shrinkage_, atol=1e-8)


def test_lw_delta_in_unit_interval_and_psd():
    S, delta = ledoit_wolf(_X(n=30, p=25, seed=2))
    assert 0 <= delta <= 1
    assert np.linalg.eigvalsh(S).min() > 0     # invertible even when n is close to p


def test_lw_is_better_conditioned_than_sample():
    X = _X(n=40, p=30, seed=3)
    assert np.linalg.cond(ledoit_wolf(X)[0]) < np.linalg.cond(sample_cov(X))


def test_ewma_equals_explicit_weighted_sum():
    X = _X(n=80, p=4, seed=4)
    lam, init = 0.94, 30
    Xc = X - X.mean(axis=0)
    expected = lam ** (len(X) - init) * np.cov(Xc[:init], rowvar=False)
    for t in range(init, len(X)):
        expected = expected + (1 - lam) * lam ** (len(X) - 1 - t) * np.outer(Xc[t], Xc[t])
    np.testing.assert_allclose(ewma_cov(X, lam, init), expected)


def test_ewma_reacts_to_recent_volatility():
    rng = np.random.default_rng(5)
    X = np.vstack([rng.standard_normal((300, 3)) * 0.01, rng.standard_normal((30, 3)) * 0.05])
    assert np.trace(ewma_cov(X)) > 2 * np.trace(sample_cov(X))


def test_ewma_symmetric_psd():
    S = ewma_cov(_X(n=150, p=5, seed=6))
    np.testing.assert_allclose(S, S.T)
    assert np.linalg.eigvalsh(S).min() > -1e-12


def test_mp_edges_values():
    lo, hi = mp_edges(250, 25)
    assert np.isclose(lo, (1 - np.sqrt(0.1)) ** 2) and np.isclose(hi, (1 + np.sqrt(0.1)) ** 2)
    assert np.allclose(mp_edges(250, 25, sigma2=4.0), 4 * np.array([lo, hi]))


def test_mp_density_integrates_to_one():
    n, p = 1000, 250
    lo, hi = mp_edges(n, p)
    x = np.linspace(lo, hi, 20001)
    assert np.isclose(_trapz(mp_density(x, n, p), x), 1.0, atol=1e-3)


def test_noise_eigenvalues_sit_inside_mp_edges():
    n, p = 2000, 50
    X = np.random.default_rng(7).standard_normal((n, p))
    eig = np.linalg.eigvalsh(np.corrcoef(X, rowvar=False))
    lo, hi = mp_edges(n, p)
    assert eig.min() > lo * 0.95 and eig.max() < hi * 1.05