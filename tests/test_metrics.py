import numpy as np
import pandas as pd

from quantlab.metrics import ann_return, ann_vol, drawdown, max_drawdown, sharpe, summary, wealth


def test_max_drawdown_by_hand():
    r = pd.Series([0.1, -0.5, 0.2])
    assert np.isclose(max_drawdown(r), -0.5)
    np.testing.assert_allclose(drawdown(r).to_numpy(), [0.0, -0.5, 0.66 / 1.1 - 1])


def test_drawdown_zero_for_rising_wealth():
    assert max_drawdown(pd.Series([0.01, 0.02, 0.03])) == 0.0


def test_sharpe_by_hand():
    r = pd.Series([0.02, 0.0])
    assert np.isclose(sharpe(r), np.sqrt(252) * 0.01 / np.std([0.02, 0.0], ddof=1))
    assert np.isclose(sharpe(pd.Series([0.01, -0.01] * 5)), 0.0)


def test_ann_return_and_vol():
    assert np.isclose(ann_return(pd.Series([0.1, 0.1]), periods=1), 0.1)
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 500))
    assert np.isclose(ann_vol(r), r.std(ddof=1) * np.sqrt(252))
    assert np.isclose(wealth(pd.Series([0.1, 0.1])).iloc[-1], 1.21)


def test_summary_columns_and_turnover():
    df = pd.DataFrame(
        {"ret": [0.01, 0.0, 0.01, 0.0], "gross": [0.011, 0.0, 0.011, 0.0], "turnover": [1.0, 0, 0, 0]}
    )
    out = summary({"s": df})
    assert list(out.columns) == [
        "ann_return", "ann_vol", "sharpe", "max_drawdown",
        "turnover_per_year", "ann_return_gross", "sharpe_gross",
    ]
    assert np.isclose(out.loc["s", "turnover_per_year"], 1.0 / 4 * 252)
    assert out.loc["s", "ann_return_gross"] > out.loc["s", "ann_return"]