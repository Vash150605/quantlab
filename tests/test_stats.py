import numpy as np
import pytest

from quantlab.stats import corr_from_cov, cov_loops, cov_vec


def test_cov_by_hand():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 9.0]])
    expected = np.array([[4.0, 7.0], [7.0, 13.0]])
    np.testing.assert_allclose(cov_loops(X), expected)
    np.testing.assert_allclose(cov_vec(X), expected)


def test_cov_three_ways():
    X = np.random.default_rng(0).standard_normal((200, 5))
    np.testing.assert_allclose(cov_loops(X), cov_vec(X))
    np.testing.assert_allclose(cov_vec(X), np.cov(X, rowvar=False))


def test_cov_symmetric_and_psd():
    X = np.random.default_rng(1).standard_normal((100, 6))
    S = cov_vec(X)
    np.testing.assert_allclose(S, S.T)
    assert np.linalg.eigvalsh(S).min() > -1e-12


def test_cov_singular_when_fewer_obs_than_assets():
    X = np.random.default_rng(2).standard_normal((5, 10))
    assert np.linalg.matrix_rank(cov_vec(X)) == 4  # n - 1


def test_cov_single_asset():
    X = np.random.default_rng(3).standard_normal((50, 1))
    assert np.isclose(cov_vec(X)[0, 0], np.var(X, ddof=1))


def test_cov_rejects_bad_input():
    with pytest.raises(ValueError):
        cov_vec(np.ones(5))
    with pytest.raises(ValueError):
        cov_vec(np.ones((1, 3)))


def test_corr_matches_numpy():
    X = np.random.default_rng(4).standard_normal((300, 4))
    np.testing.assert_allclose(corr_from_cov(cov_vec(X)), np.corrcoef(X, rowvar=False))


def test_corr_constant_column_raises():
    X = np.random.default_rng(5).standard_normal((20, 3))
    X[:, 1] = 7.0
    with pytest.raises(ValueError):
        corr_from_cov(cov_vec(X))