import numpy as np
import pandas as pd

from quantlab.returns import log_returns, simple_returns


def _prices():
    idx = pd.bdate_range("2020-01-01", periods=3)
    return pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=idx)


def test_simple_returns_values():
    r = simple_returns(_prices())
    assert np.allclose(r["A"].values, [0.10, -0.10])


def test_log_returns_values():
    r = log_returns(_prices())
    assert np.allclose(r["A"].values, [np.log(1.1), np.log(0.9)])


def test_log_simple_relationship():
    p = _prices()
    assert np.allclose(np.exp(log_returns(p)) - 1, simple_returns(p))