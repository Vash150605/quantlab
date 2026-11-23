from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yfinance as yf

from quantlab.universe import ALL, START, END

# Always the project's data/processed folder, wherever you run the code from
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

def _cache_path(tickers, start, end) -> Path:
    """Same request always maps to the same filename. Order of tickers does not matter."""
    key = "|".join(sorted(tickers)) + f"|{start}|{end}"
    return CACHE_DIR / f"prices_{hashlib.md5(key.encode()).hexdigest()[:10]}.parquet"


def validate_prices(prices: pd.DataFrame, max_missing: float = 0.05) -> pd.DataFrame:
    """Check that prices are valid and return a cleaned copy."""
    if prices.empty:
        raise ValueError("no data returned for the given tickers and data range")
    
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Index must be sorted in ascending order")
    
    if prices.index.has_duplicates:
        raise ValueError("Index must not have duplicates: it must have unique dates")
    
    if prices.isnull().mean().max() > max_missing:
        raise ValueError(f"Prices have more than {max_missing*100}% missing values")
    
    if (prices.dropna()<=0).any().any():
        raise ValueError("non-positive prices found")

    # fill missing values with the last available price, but limit to 3 consecutive days,
    # then drop any remaining missing values
    cleaned_prices = prices.ffill(limit=3).dropna()
    return cleaned_prices


def load_prices(
    tickers=ALL, 
    start: str = START, 
    end: str = END, 
    refresh: bool = False
) -> pd.DataFrame:
    """Download adjusted close prices and cache them as parquet files.
        Returned a DataFrame with dates as index and tickers as columns."""

    path = _cache_path(tickers, start, end)

    #checked whether the cache data already exists
    if path.exists() and not refresh:
        prices = pd.read_parquet(path)   # loading the existing cache file
        return prices   # return the cached prices without downloading again

    prices = yf.download(tickers=list(tickers), start=start, end=end, auto_adjust = True,progress=False)

    # keep an untouched copy of the download 
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / path.name.replace("prices_", "raw_")
    prices.to_parquet(raw_path)


    prices = prices['Close']
    # a single ticker can come back as a Series, then turn into DataFrame 
    if isinstance(prices, pd.Series):    
        prices = prices.to_frame(name=tickers[0])

    cleaned_prices = validate_prices(prices)

    # save the cleaned prices to the cache 
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_prices.to_parquet(path)

    return cleaned_prices

