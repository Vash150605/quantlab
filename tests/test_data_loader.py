import numpy as np
import pandas as pd
import pytest

import quantlab.data_loader as dl
from quantlab.data_loader import _cache_path, validate_prices
from quantlab.returns import log_returns
from quantlab.validation import calendar_gaps, extreme_moves, first_valid_dates


def make_prices(n=50, k=3, seed=0):
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    vals = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, k)), axis=0))
    return pd.DataFrame(vals, index=idx, columns=list("ABC")[:k])


@pytest.fixture
def fake_net(tmp_path, monkeypatch):
    """Point the loader at temporary folders and replace yfinance by a counting fake."""
    monkeypatch.setattr(dl, "CACHE_DIR", tmp_path / "processed")
    monkeypatch.setattr(dl, "RAW_DIR", tmp_path / "raw", raising=False)
    state = {"calls": 0, "data": make_prices()}

    def fake_download(*args, **kwargs):
        state["calls"] += 1
        return pd.concat({"Close": state["data"]}, axis=1)

    monkeypatch.setattr(dl.yf, "download", fake_download)
    return state


# ---- validate_prices ----

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
    p.iloc[:10, 0] = np.nan                       # 20% missing
    with pytest.raises(ValueError, match="missing"):
        validate_prices(p)


def test_non_positive_raises():
    p = make_prices()
    p.iloc[5, 0] = -1.0
    with pytest.raises(ValueError, match="non-positive"):
        validate_prices(p)


def test_short_gap_is_filled():
    p = make_prices()
    p.iloc[10:12, 1] = np.nan                     # 4% missing, 2-day gap
    out = validate_prices(p)
    assert not out.isna().any().any() and len(out) == len(p)


def test_gap_longer_than_three_days_is_not_filled():
    p = make_prices(n=200)
    p.iloc[10:15, 1] = np.nan                     # 5-day gap, 2.5% missing
    out = validate_prices(p)
    assert not out.isna().any().any()
    assert len(out) == len(p) - 2                 # 3 days filled, the last 2 rows dropped
    assert out.iloc[:, 1].loc[p.index[10:13]].nunique() == 1   # filled with the last price


def test_validate_does_not_modify_input():
    p = make_prices()
    p.iloc[3, 0] = np.nan
    before = p.copy()
    validate_prices(p)
    pd.testing.assert_frame_equal(p, before)


# ---- cache key ----

def test_cache_key_ignores_ticker_order():
    assert _cache_path(["A", "B"], "2020", "2021") == _cache_path(["B", "A"], "2020", "2021")


def test_cache_key_changes_with_dates():
    assert _cache_path(["A"], "2020", "2021") != _cache_path(["A"], "2020", "2022")


def test_cache_key_changes_with_tickers():
    assert _cache_path(["A"], "2020", "2021") != _cache_path(["A", "B"], "2020", "2021")


# ---- load_prices (no network) ----

def test_second_load_uses_cache_not_network(fake_net):
    a = dl.load_prices(["A", "B", "C"], "2020-01-01", "2020-03-01")
    b = dl.load_prices(["A", "B", "C"], "2020-01-01", "2020-03-01")
    assert fake_net["calls"] == 1
    pd.testing.assert_frame_equal(a, b, check_freq=False, check_names=False)


def test_refresh_forces_new_download(fake_net):
    dl.load_prices(["A", "B", "C"], "2020-01-01", "2020-03-01")
    dl.load_prices(["A", "B", "C"], "2020-01-01", "2020-03-01", refresh=True)
    assert fake_net["calls"] == 2


def test_single_ticker_works(fake_net, monkeypatch):
    # a lone ticker can come back as a Series instead of a table
    def one_ticker(*args, **kwargs):
        fake_net["calls"] += 1
        return pd.DataFrame({"Close": make_prices(k=1)["A"]})   # raw["Close"] is a Series

    monkeypatch.setattr(dl.yf, "download", one_ticker)
    out = dl.load_prices(["A"], "2020-01-01", "2020-03-01")
    assert isinstance(out, pd.DataFrame) and list(out.columns) == ["A"]


def test_loaded_prices_have_no_nan(fake_net):
    fake_net["data"].iloc[10:12, 1] = np.nan
    out = dl.load_prices(["A", "B", "C"], "2020-01-01", "2020-03-01")
    assert not out.isna().any().any()


def test_parquet_roundtrip_is_identical(fake_net):
    first = dl.load_prices(["A", "B", "C"], "2020-01-01", "2020-03-01")
    path = _cache_path(["A", "B", "C"], "2020-01-01", "2020-03-01")
    assert path.exists()
    pd.testing.assert_frame_equal(pd.read_parquet(path), first, check_freq=False)


def test_raw_download_is_saved_next_to_processed(fake_net):
    if not hasattr(dl, "RAW_DIR"):
        pytest.skip("this data_loader does not save raw files")
    dl.load_prices(["A", "B", "C"], "2020-01-01", "2020-03-01")
    assert len(list(dl.RAW_DIR.glob("raw_*.parquet"))) == 1


# ---- checks on the data (validation.py) ----

def test_split_like_jump_is_flagged():
    p = make_prices()
    p.iloc[25:, 0] = p.iloc[25:, 0] * 0.5          # an unadjusted 2-for-1 split
    flagged = extreme_moves(log_returns(p), threshold=0.25)
    assert len(flagged) == 1
    assert flagged.iloc[0]["ticker"] == "A"
    assert flagged.iloc[0]["date"] == p.index[25]


def test_calendar_gaps_flags_long_gap_only():
    idx = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-06", "2020-01-20"])
    gaps = calendar_gaps(pd.DataFrame({"A": range(4)}, index=idx), max_days=5)
    assert gaps["date"].tolist() == [pd.Timestamp("2020-01-20")]
    assert gaps["gap_days"].tolist() == [14]


def test_first_valid_dates_reports_late_starter():
    p = make_prices()
    p.iloc[:10, 1] = np.nan
    out = first_valid_dates(p)
    assert out.loc["A", "first_date"] == p.index[0]
    assert out.loc["B", "first_date"] == p.index[10]
    assert np.isclose(out.loc["B", "missing_frac"], 10 / 50)