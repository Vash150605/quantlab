import numpy as np
import pandas as pd
import pytest

import quantlab.data_loader as dl
from quantlab.data_loader import _cache_path, validate_prices


def make_prices(n=50, k=3, seed=0):
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    vals = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, k)), axis=0))
    return pd.DataFrame(vals, index=idx, columns=list("ABC")[:k])


# ---------- validate_prices: bad data must be refused ----------
def test_empty_raises():
    with pytest.raises(ValueError, match="no data"):
        validate_prices(pd.DataFrame())


def test_duplicate_dates_raise():
    p = make_prices()
    p = pd.concat([p, p.iloc[[0]]]).sort_index()
    with pytest.raises(ValueError, match="unique"):
        validate_prices(p)


def test_unsorted_index_raises():
    with pytest.raises(ValueError, match="sorted"):
        validate_prices(make_prices().iloc[::-1])


def test_too_many_missing_raises():
    p = make_prices()
    p.iloc[:10, 0] = np.nan  # 20% missing
    with pytest.raises(ValueError, match="missing"):
        validate_prices(p)


def test_non_positive_raises():
    p = make_prices()
    p.iloc[5, 0] = -1.0
    with pytest.raises(ValueError, match="non-positive"):
        validate_prices(p)


# ---------- validate_prices: small problems are cleaned ----------
def test_short_gap_is_filled():
    p = make_prices()
    p.iloc[10:12, 1] = np.nan  # 2-day gap, 4% missing
    out = validate_prices(p)
    assert not out.isna().any().any() and len(out) == len(p)


def test_gap_longer_than_3_days_is_not_invented():
    p = make_prices(n=200)
    p.iloc[10:15, 1] = np.nan  # 5-day gap (2.5% missing, so it passes the 5% check)
    out = validate_prices(p)
    assert len(out) == len(p) - 2  # 3 days filled, the last 2 rows dropped


# ---------- cache ----------
def test_cache_key_ignores_ticker_order():
    assert _cache_path(["A", "B"], "2020", "2021") == _cache_path(["B", "A"], "2020", "2021")


def test_cache_key_changes_with_dates():
    assert _cache_path(["A"], "2020", "2021") != _cache_path(["A"], "2020", "2022")


def _fake_network(monkeypatch, tmp_path, frame):
    """Send cache and raw files to tmp_path and replace the download with a counter."""
    monkeypatch.setattr(dl, "CACHE_DIR", tmp_path / "processed")
    monkeypatch.setattr(dl, "RAW_DIR", tmp_path / "raw")
    calls = {"n": 0}

    def fake_download(*a, **k):
        calls["n"] += 1
        return pd.concat({"Close": frame}, axis=1)

    monkeypatch.setattr(dl.yf, "download", fake_download)
    return calls


def test_second_load_uses_cache_not_network(tmp_path, monkeypatch):
    calls = _fake_network(monkeypatch, tmp_path, make_prices())
    a = dl.load_prices(["A", "B", "C"], "2020-01-01", "2020-03-01")
    b = dl.load_prices(["A", "B", "C"], "2020-01-01", "2020-03-01")
    assert calls["n"] == 1
    pd.testing.assert_frame_equal(a, b, check_freq=False, check_names=False)


def test_refresh_forces_new_download(tmp_path, monkeypatch):
    calls = _fake_network(monkeypatch, tmp_path, make_prices())
    dl.load_prices(["A", "B", "C"], "2020-01-01", "2020-03-01")
    dl.load_prices(["A", "B", "C"], "2020-01-01", "2020-03-01", refresh=True)
    assert calls["n"] == 2


def test_raw_download_is_saved_untouched(tmp_path, monkeypatch):
    _fake_network(monkeypatch, tmp_path, make_prices())
    dl.load_prices(["A", "B", "C"], "2020-01-01", "2020-03-01")
    assert len(list((tmp_path / "raw").glob("raw_*.parquet"))) == 1
