import numpy as np
from sklearn.decomposition import PCA

from quantlab.linalg_tools import ols_normal, ols_qr, ols_svd, pca_svd


def _data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((200, 4))
    beta = np.array([1.0, -2.0, 0.5, 3.0])
    return X, beta, X @ beta + 0.1 * rng.standard_normal(200)


def test_ols_three_ways_agree_with_lstsq():
    X, _, y = _data()
    ref = np.linalg.lstsq(X, y, rcond=None)[0]
    for f in (ols_normal, ols_qr, ols_svd):
        np.testing.assert_allclose(f(X, y), ref, atol=1e-10)


def test_ols_recovers_true_beta_without_noise():
    X, beta, _ = _data()
    for f in (ols_normal, ols_qr, ols_svd):
        np.testing.assert_allclose(f(X, X @ beta), beta, atol=1e-8)


def test_qr_beats_normal_equations_when_ill_conditioned():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(500)
    X = np.column_stack([a, a + 1e-7 * rng.standard_normal(500)])
    beta = np.array([1.0, 1.0])
    y = X @ beta
    err = lambda f: np.linalg.norm(f(X, y) - beta)
    assert err(ols_qr) < err(ols_normal)


def test_pca_matches_eigendecomposition():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 6)) @ rng.standard_normal((6, 6))
    eigvals, _, _ = pca_svd(X)
    ref = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1]
    np.testing.assert_allclose(eigvals, ref)


def test_pca_matches_sklearn_up_to_sign():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((300, 5)) @ rng.standard_normal((5, 5))
    eigvals, Vt, _ = pca_svd(X)
    sk = PCA().fit(X)
    np.testing.assert_allclose(eigvals, sk.explained_variance_)
    np.testing.assert_allclose(np.abs(Vt), np.abs(sk.components_), atol=1e-8)


def test_pca_scores_have_variance_eigvals():
    X = np.random.default_rng(3).standard_normal((400, 4)) @ np.random.default_rng(4).standard_normal((4, 4))
    eigvals, _, scores = pca_svd(X)
    np.testing.assert_allclose(scores.var(axis=0, ddof=1), eigvals)