import numpy as np
import pandas as pd

from quantlab.diagnostics import avg_pairwise_corr, corr_by_period, fit_t


def test_avg_pairwise_corr_by_hand():
    a = np.random.default_rng(0).standard_normal(500)
    same = pd.DataFrame({"x": a, "y": a})
    assert np.isclose(avg_pairwise_corr(same), 1.0)
    mixed = pd.DataFrame({"x": a, "y": a, "z": -a})
    assert np.isclose(avg_pairwise_corr(mixed), -1 / 3)          # (1 - 1 - 1) / 3


def test_independent_columns_have_near_zero_correlation():
    df = pd.DataFrame(np.random.default_rng(1).standard_normal((5000, 6)))
    assert abs(avg_pairwise_corr(df)) < 0.03


def test_corr_by_period_detects_a_crisis_block():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2020-01-01", periods=400)
    calm = rng.standard_normal((200, 5))
    factor = rng.standard_normal((200, 1))
    crisis = 0.9 * factor + 0.4 * rng.standard_normal((200, 5))     # strong common factor
    df = pd.DataFrame(np.vstack([calm, crisis]), index=idx)
    out = corr_by_period(df, {"calm": (idx[0], idx[199]), "crisis": (idx[200], idx[-1])})
    assert out["crisis"] > out["calm"] + 0.5


def test_fit_t_recovers_fat_and_thin_tails():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"fat": rng.standard_t(4, 20000), "normal": rng.standard_normal(20000)})
    out = fit_t(df)
    assert 3.0 < out.loc["fat", "df"] < 5.5
    assert out.loc["normal", "df"] > 20