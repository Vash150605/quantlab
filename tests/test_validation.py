import numpy as np
import pandas as pd

from quantlab.validation import extreme_moves, stale_runs


def _returns():
    idx = pd.bdate_range("2020-01-01", periods=8)
    return pd.DataFrame(
        {
            "A": [0.01, -0.30, 0.01, 0.0, 0.02, 0.0, 0.01, 0.0],
            "B": [0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.01],
        },
        index=idx,
    )


def test_extreme_moves_finds_the_big_one():
    out = extreme_moves(_returns())
    assert len(out) == 1
    assert out.iloc[0]["ticker"] == "A"
    assert np.isclose(out.iloc[0]["log_return"], -0.30)


def test_stale_runs_flags_only_long_runs():
    out = stale_runs(_returns(), min_run=5)
    assert out["ticker"].tolist() == ["B"]
    assert out["stale_days"].iloc[0] == 5