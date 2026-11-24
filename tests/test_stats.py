import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quantlab.diagnostics import (
    autocorr_table,
    avg_pairwise_corr,
    fit_t,
    moment_table,
    rolling_vol,
    tail_counts,
)
from quantlab.linalg_tools import ols_normal, ols_qr, ols_svd, pca_svd
from quantlab.stats import corr_from_cov, cov_loops, cov_vec


# ---------- stats.py ----------

def test_cov_by_hand():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 9.0]])
    expected = np.array([[4.0, 7.0], [7.0, 13.0]])

    np.testing.assert_allclose(cov_loops(X), expected)
    np.testing.assert_allclose(cov_vec(X), expected)


def test_cov_matches_numpy():
    X = np.random.default_rng(0).standard_normal((200, 5))

    np.testing.assert_allclose(cov_loops(X), cov_vec(X))
    np.testing.assert_allclose(cov_vec(X), np.cov(X, rowvar=False))


def test_cov_is_symmetric():
    X = np.random.default_rng(1).standard_normal((100, 6))
    S = cov_vec(X)

    np.testing.assert_allclose(S, S.T)
    assert np.linalg.eigvalsh(S).min() > -1e-12     # no negative variances


def test_cov_rejects_bad_input():
    with pytest.raises(ValueError):
        cov_vec(np.ones(5))                          # 1-D
    with pytest.raises(ValueError):
        cov_vec(np.ones((1, 3)))                     # only one row


def test_corr_matches_numpy():
    X = np.random.default_rng(4).standard_normal((300, 4))

    np.testing.assert_allclose(corr_from_cov(cov_vec(X)), np.corrcoef(X, rowvar=False))


def test_corr_constant_column_raises():
    X = np.random.default_rng(5).standard_normal((20, 3))
    X[:, 1] = 7.0                                    # zero variance

    with pytest.raises(ValueError):
        corr_from_cov(cov_vec(X))


# ---------- linalg_tools.py ----------

def make_regression_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 4))
    beta = np.array([1.0, -2.0, 0.5, 3.0])
    y = X @ beta + 0.1 * rng.standard_normal(200)
    return X, beta, y


def test_ols_three_ways_match_numpy():
    X, _, y = make_regression_data()
    reference = np.linalg.lstsq(X, y, rcond=None)[0]

    for method in (ols_normal, ols_qr, ols_svd):
        np.testing.assert_allclose(method(X, y), reference, atol=1e-10)


def test_ols_recovers_true_beta_without_noise():
    X, beta, _ = make_regression_data()

    for method in (ols_normal, ols_qr, ols_svd):
        np.testing.assert_allclose(method(X, X @ beta), beta, atol=1e-8)


def test_pca_matches_eigendecomposition():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 6)) @ rng.standard_normal((6, 6))

    eigvals, _, _ = pca_svd(X)
    reference = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1]

    np.testing.assert_allclose(eigvals, reference)


def test_pca_scores_have_variance_eigvals():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((400, 4)) @ rng.standard_normal((4, 4))

    eigvals, _, scores = pca_svd(X)

    np.testing.assert_allclose(scores.var(axis=0, ddof=1), eigvals)


# ---------- diagnostics.py ----------

def test_tail_counts_single_outlier():
    x = np.zeros(100)
    x[50] = 1.0
    df = pd.DataFrame({"A": x})

    out = tail_counts(df, k=4)

    assert out.loc["A", "observed"] == 1
    assert np.isclose(out.loc["A", "expected"], 2 * stats.norm.sf(4) * 100)


def test_heavy_tails_have_high_kurtosis():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "normal": rng.standard_normal(20000),
            "heavy": rng.standard_t(3, 20000),
        }
    )

    out = moment_table(df)

    assert abs(out.loc["normal", "excess_kurt"]) < 0.3
    assert out.loc["heavy", "excess_kurt"] > 2


def test_autocorr_detects_volatility_clustering():
    rng = np.random.default_rng(0)
    n = 20000
    # volatility switches between 1% and 5% every 50 days
    sigma = np.where((np.arange(n) // 50) % 2 == 0, 0.01, 0.05)
    df = pd.DataFrame(
        {
            "iid": rng.standard_normal(n) * 0.02,
            "clustered": rng.standard_normal(n) * sigma,
        }
    )

    out = autocorr_table(df)

    assert abs(out.loc["iid", "acf_sq"]) < 0.05
    assert out.loc["clustered", "acf_sq"] > 0.1
    assert out.loc["clustered", "lb_p_sq"] < 0.01


def test_rolling_vol():
    df = pd.DataFrame({"A": np.tile([0.01, -0.01], 50)})

    out = rolling_vol(df, window=10)

    assert out["A"].iloc[:9].isna().all()            # first 9 rows have no full window
    expected = np.std([0.01, -0.01] * 5, ddof=1) * np.sqrt(252)
    assert np.isclose(out["A"].iloc[20], expected)


def test_avg_pairwise_corr_by_hand():
    a = np.random.default_rng(0).standard_normal(500)
    same = pd.DataFrame({"x": a, "y": a})
    mixed = pd.DataFrame({"x": a, "y": a, "z": -a})

    assert np.isclose(avg_pairwise_corr(same), 1.0)
    assert np.isclose(avg_pairwise_corr(mixed), -1 / 3)    # (1 - 1 - 1) / 3


def test_fit_t_finds_fat_and_thin_tails():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "fat": rng.standard_t(4, 20000),
            "normal": rng.standard_normal(20000),
        }
    )

    out = fit_t(df)

    assert 3.0 < out.loc["fat", "df"] < 5.5
    assert out.loc["normal", "df"] > 20
