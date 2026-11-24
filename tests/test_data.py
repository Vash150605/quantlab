import numpy as np
import pandas as pd
import pytest

import quantlab.data_loader as dl
from quantlab.data_loader import validate_prices, _cache_path
from quantlab.returns import log_returns, simple_returns, annualise
from quantlab.validation import (
    extreme_moves,
    stale_runs,
    calendar_gaps,
    first_valid_dates
)

#Helper function to create a sample prices DataFrame
def make_prices(n=50, k=3):
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(0)

    values = 100 * np.exp(
        np.cumsum(rng.normal(0, 0.01, (n, k)), axis=0)
    )
    return pd.DataFrame(
        values, 
        index=dates, 
        columns=list("ABC")[:k]
    )

def make_returns():
    dates = pd.bdate_range("2020-01-01", periods=8)
    return pd.DataFrame(
                {
            "A": [0.01, -0.30, 0.01, 0, 0.02, 0, 0.01, 0],
            "B": [0, 0, 0, 0, 0, 0.01, 0, 0.01]
        },
        index=dates
    )

@pytest.fixture
def fake_download(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CACHE_DIR", tmp_path / "processed")
    monkeypatch.setattr(dl, "RAW_DIR", tmp_path / "raw")

    calls = {"count": 0}

    def fake(*args, **kwargs):
        calls["count"] += 1
        return pd.concat({"Close": make_prices()}, axis=1)

    monkeypatch.setattr(dl.yf, "download", fake)

    return calls


# data_loader.py tests
def test_empty_prices():
    with pytest.raises(ValueError, match="no data"):
        validate_prices(pd.DataFrame())


def test_unsorted_dates():
    prices = make_prices().iloc[::-1]

    with pytest.raises(ValueError, match="sorted"):
        validate_prices(prices)


def test_duplicate_dates():
    prices = make_prices()
    prices = pd.concat([prices, prices.iloc[[0]]]).sort_index()

    with pytest.raises(ValueError, match="unique"):
        validate_prices(prices)


def test_too_many_missing_values():
    prices = make_prices()
    prices.iloc[:10, 0] = np.nan

    with pytest.raises(ValueError, match="missing"):
        validate_prices(prices)


def test_non_positive_prices():
    prices = make_prices()
    prices.iloc[5, 0] = -1

    with pytest.raises(ValueError, match="non-positive"):
        validate_prices(prices)


def test_short_missing_gap_is_filled():
    prices = make_prices()
    prices.iloc[10:12, 1] = np.nan

    result = validate_prices(prices)

    assert not result.isna().any().any()
    assert len(result) == len(prices)


def test_long_missing_gap_is_partly_dropped():
    prices = make_prices(n=200)
    prices.iloc[10:15, 1] = np.nan

    result = validate_prices(prices)

    assert not result.isna().any().any()
    assert len(result) == len(prices) - 2

# Cache and download tests
def test_cache_key_ignores_ticker_order():
    first = _cache_path(["A", "B"], "2020", "2021")
    second = _cache_path(["B", "A"], "2020", "2021")

    assert first == second


def test_cache_is_used_on_second_download(fake_download):
    first = dl.load_prices(
        ["A", "B", "C"], "2020-01-01", "2020-03-01"
    )

    second = dl.load_prices(
        ["A", "B", "C"], "2020-01-01", "2020-03-01"
    )

    assert fake_download["count"] == 1

    pd.testing.assert_frame_equal(
        first, second,
        check_freq=False,
        check_names=False
    )


def test_refresh_downloads_again(fake_download):
    dl.load_prices(["A", "B", "C"], "2020-01-01", "2020-03-01")

    dl.load_prices(
        ["A", "B", "C"],
        "2020-01-01",
        "2020-03-01",
        refresh=True
    )

    assert fake_download["count"] == 2



# returns.py tests

def test_simple_returns():
    prices = pd.DataFrame({"A": [100, 110, 99]})

    result = simple_returns(prices)

    assert np.allclose(result["A"], [0.10, -0.10])


def test_log_returns():
    prices = pd.DataFrame({"A": [100, 110, 99]})

    result = log_returns(prices)

    expected = [np.log(1.1), np.log(0.9)]

    assert np.allclose(result["A"], expected)


def test_log_and_simple_returns_relationship():
    prices = pd.DataFrame({"A": [100, 110, 99]})

    log_result = log_returns(prices)
    simple_result = simple_returns(prices)

    assert np.allclose(
        np.exp(log_result) - 1,
        simple_result
    )


def test_annualise():
    returns = pd.DataFrame({"A": [0.01, 0.03]})

    result = annualise(returns)

    expected_mean = 0.02 * 252
    expected_vol = np.std([0.01, 0.03], ddof=1) * np.sqrt(252)

    assert np.isclose(result.loc["A", "mean"], expected_mean)
    assert np.isclose(result.loc["A", "vol"], expected_vol)


# validation.py tests

def test_extreme_moves():
    result = extreme_moves(make_returns())

    assert len(result) == 1
    assert result.iloc[0]["ticker"] == "A"
    assert np.isclose(result.iloc[0]["log_return"], -0.30)


def test_stale_runs():
    result = stale_runs(make_returns(), min_run=5)

    assert result["ticker"].tolist() == ["B"]
    assert result["stale_days"].iloc[0] == 5


def test_calendar_gaps():
    dates = pd.to_datetime([
        "2020-01-01",
        "2020-01-02",
        "2020-01-06",
        "2020-01-20"
    ])

    prices = pd.DataFrame({"A": [100, 101, 102, 103]}, index=dates)

    result = calendar_gaps(prices, max_days=5)

    assert result["date"].tolist() == [pd.Timestamp("2020-01-20")]
    assert result["gap_days"].tolist() == [14]


def test_first_valid_dates():
    prices = make_prices()
    prices.iloc[:10, 1] = np.nan

    result = first_valid_dates(prices)

    assert result.loc["A", "first_date"] == prices.index[0]
    assert result.loc["B", "first_date"] == prices.index[10]
    assert np.isclose(result.loc["B", "missing_frac"], 10 / 50)