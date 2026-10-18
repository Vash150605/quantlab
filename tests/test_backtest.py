import numpy as np
import pandas as pd

from quantlab.backtest import (
    equal_weight, make_shrunk_mv, minvar_lw, minvar_lw_long_only, minvar_sample, walk_forward,
)
from quantlab.covariance import ledoit_wolf
from quantlab.optimizers import min_variance_long_only, min_variance_weights


def _returns(n=300, p=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame(rng.normal(0.0004, 0.01, (n, p)), index=idx, columns=[f"A{i}" for i in range(p)])


def test_output_shape_and_start():
    r = _returns()
    out = walk_forward(r, equal_weight, window=60, rebalance=20)
    assert len(out) == len(r) - 60
    assert out.index[0] == r.index[60]
    assert not out.isna().any().any()


def test_first_day_hand_check_with_costs():
    r = _returns(n=100, p=2)
    out = walk_forward(r, equal_weight, window=10, rebalance=1000, cost=0.01)
    day0 = r.iloc[10].mean()
    assert np.isclose(out["gross"].iloc[0], day0)
    assert np.isclose(out["turnover"].iloc[0], 1.0)          # from cash into fully invested
    assert np.isclose(out["ret"].iloc[0], day0 - 0.01)
    assert out["turnover"].iloc[1:].sum() == 0               # no rebalance, no turnover


def test_weights_drift_between_rebalances():
    r = _returns(n=50, p=2, seed=1)
    out = walk_forward(r, equal_weight, window=10, rebalance=1000, cost=0.0)
    R = r.to_numpy()
    w = np.array([0.5, 0.5])
    w = w * (1 + R[10])
    w = w / w.sum()
    assert np.isclose(out["gross"].iloc[1], w @ R[11])


def test_turnover_on_rebalance_equals_drift_distance():
    r = _returns(n=80, p=3, seed=2)
    out = walk_forward(r, equal_weight, window=20, rebalance=10, cost=0.0)
    R = r.to_numpy()
    w = np.ones(3) / 3
    for t in range(20, 30):
        w = w * (1 + R[t]); w = w / w.sum()
    assert np.isclose(out["turnover"].iloc[10], np.abs(np.ones(3) / 3 - w).sum())


def test_costs_lower_net_returns():
    r = _returns()
    free = walk_forward(r, minvar_lw, window=60, rebalance=20, cost=0.0)
    costly = walk_forward(r, minvar_lw, window=60, rebalance=20, cost=0.002)
    assert np.allclose(free["gross"], costly["gross"])
    assert costly["ret"].sum() < free["ret"].sum()


def test_weight_fn_sees_only_the_past():
    r = _returns(n=120, p=3, seed=3)
    R = r.to_numpy()
    seen = []

    def spy(X):
        seen.append(X.copy())
        return np.ones(X.shape[1]) / X.shape[1]

    walk_forward(r, spy, window=30, rebalance=15)
    for k, X in enumerate(seen):
        t = 30 + 15 * k
        assert X.shape[0] == 30
        np.testing.assert_array_equal(X, R[t - 30 : t])      # rows t-30 .. t-1, never row t


def test_no_look_ahead_future_change_does_not_alter_past():
    r = _returns(n=200, p=4, seed=4)
    t0 = 150
    r2 = r.copy()
    r2.iloc[t0] = r2.iloc[t0] + 0.5                           # shock to one future day
    for fn in (equal_weight, minvar_sample, minvar_lw, make_shrunk_mv()):
        a = walk_forward(r, fn, window=60, rebalance=10)
        b = walk_forward(r2, fn, window=60, rebalance=10)
        before = a.index < r.index[t0]
        pd.testing.assert_frame_equal(a[before], b[before])   # everything before t0 identical
        assert not np.isclose(a["gross"].loc[r.index[t0]], b["gross"].loc[r.index[t0]])


def test_strategies_return_valid_weights():
    X = _returns(n=120, p=5, seed=5).to_numpy()
    for fn in (equal_weight, minvar_sample, minvar_lw, minvar_lw_long_only, make_shrunk_mv()):
        assert np.isclose(fn(X).sum(), 1.0)
    assert minvar_lw_long_only(X).min() >= -1e-12


def test_shrunk_mv_falls_back_when_all_means_negative():
    rng = np.random.default_rng(6)
    X = rng.normal(-0.01, 0.01, (200, 4))                     # strongly negative means
    expected = min_variance_weights(ledoit_wolf(X)[0])
    np.testing.assert_allclose(make_shrunk_mv()(X), expected)


def test_long_only_matches_closed_form_when_closed_form_is_positive():
    cov = np.diag([0.04, 0.01, 0.02])
    np.testing.assert_allclose(min_variance_long_only(cov), min_variance_weights(cov), atol=1e-6)


def test_shrunk_mv_gross_cap_triggers_fallback():
    X = _returns(n=200, p=5, seed=7).to_numpy()
    always_fallback = make_shrunk_mv(max_gross=0.5)           # gross is never below 1
    np.testing.assert_allclose(always_fallback(X), minvar_lw(X))
    assert always_fallback.stats == {"calls": 1, "fallbacks": 1}
    never_fallback = make_shrunk_mv(max_gross=1e9)
    w = never_fallback(X)
    assert np.isclose(w.sum(), 1.0)
    assert never_fallback.stats["fallbacks"] in (0, 1)