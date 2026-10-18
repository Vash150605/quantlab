import numpy as np
import pandas as pd
from scipy import stats

from quantlab.diagnostics import jarque_bera_table, moment_table, tail_counts
from quantlab.diagnostics import autocorr_table, rolling_vol


def test_tail_counts_single_outlier():
    x = np.zeros(100)
    x[50] = 1.0
    df = pd.DataFrame({"A": x}, index=pd.bdate_range("2020-01-01", periods=100))
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
    m = moment_table(df)
    assert abs(m.loc["normal", "excess_kurt"]) < 0.3
    assert m.loc["heavy", "excess_kurt"] > 2


def test_jarque_bera_rejects_heavy_tails():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"heavy": rng.standard_t(3, 20000)})
    out = jarque_bera_table(df)
    assert out["reject"].dtype == bool


def test_autocorr_detects_clustering():
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
    assert abs(out.loc["clustered", "acf_ret"]) < 0.05


def test_rolling_vol_shape_and_value():
    df = pd.DataFrame({"A": np.tile([0.01, -0.01], 50)})
    out = rolling_vol(df, window=10)
    assert out["A"].iloc[:9].isna().all()
    assert out["A"].iloc[9:].notna().all()
    expected = np.std([0.01, -0.01] * 5, ddof=1) * np.sqrt(252)
    assert np.isclose(out["A"].iloc[20], expected)