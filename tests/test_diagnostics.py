import numpy as np
import pandas as pd
from scipy import stats

from quantlab.diagnostics import jarque_bera_table, moment_table, tail_counts


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