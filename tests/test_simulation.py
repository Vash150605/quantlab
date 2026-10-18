import numpy as np

from quantlab.simulation import excess_variance_study, one_factor_cov


def test_one_factor_cov_is_valid():
    S = one_factor_cov(10, seed=1)
    np.testing.assert_allclose(S, S.T)
    assert np.linalg.eigvalsh(S).min() > 0


def test_study_is_reproducible_and_ratio_at_least_one():
    S = one_factor_cov(8, seed=2)
    a = excess_variance_study(S, ns=(60,), reps=20, seed=3)
    b = excess_variance_study(S, ns=(60,), reps=20, seed=3)
    assert a.equals(b)
    assert (a["ratio"] >= 1 - 1e-9).all()     # the true min-variance portfolio cannot be beaten