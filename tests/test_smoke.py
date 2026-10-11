"""Checks that the environment and the package are wired up correctly."""
import numpy as np
import pandas as pd

import quantlab


def test_package_imports():
    assert quantlab.__version__ == "0.1.0"


def test_parquet_roundtrip(tmp_path):
    idx = pd.bdate_range("2020-01-01", periods=5)
    df = pd.DataFrame({"A": np.arange(5.0)}, index=idx)
    path = tmp_path / "x.parquet"
    df.to_parquet(path)
    out = pd.read_parquet(path)
    pd.testing.assert_frame_equal(df, out, check_freq=False)
