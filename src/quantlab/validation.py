"""Checks on returns. They report problems; they never change the data."""
from __future__ import annotations

import pandas as pd


def extreme_moves(returns: pd.DataFrame, threshold: float = 0.20) -> pd.DataFrame:
    """
    Find returns that are exceed the threshold in either direction.
    Returns a DataFrame with columns: date, ticker, log_return
    """
    stacked = returns.stack()   # convert the DataFrame into single Series
    extreme_returns = stacked[stacked.abs() > threshold]

    result = extreme_returns.reset_index()  # convert the Series back into a DataFrame with columns: date, ticker, log_return

    result.columns = ["date", "ticker", "log_return"]
    return result.sort_values(
        "log_return", 
        key= lambda values: values.abs(), 
        ascending=False
    )


def stale_runs(returns: pd.DataFrame, min_run: int = 5) -> pd.DataFrame:
    """
    Find the stock with consective zero returns.
    A zero returns means the price did not change from the previous day.
    This can be a sign of stale data, especially if it happens for many day in a row.
    """
    rows = []

    for ticker in returns.columns:
        is_zero = returns[ticker] == 0
        group_id = (is_zero != is_zero.shift()).cumsum() # Assign a group number whenever the condition changes
        run_length = is_zero.groupby(group_id).transform("sum")  # Calculate the length of each consecutive group
        stale_days = is_zero & (run_length >= min_run)
        total_stale_days = stale_days.sum()
        if total_stale_days>0:
            rows.append((ticker, int(total_stale_days)))
    result = pd.DataFrame(
        rows,
        columns=["ticker", "stale_days"]
    )
    return result

def calendar_gaps(prices: pd.DataFrame, max_days: int = 5) -> pd.DataFrame:
    """
    Find dates where the gap from the previous available date
    exceeds the specified number of calendar days.
    """
    date_differences = prices.index.to_series().diff()  #Calculate the difference between consecutive dates
    gap_days = date_differences.dt.days  #Convert the differences to number of days
    large_gaps = gap_days[gap_days > max_days]  #Filter for gaps that exceed the specified number of days

    # Create a DataFrame with the gap dates and gap lengths
    result = pd.DataFrame({
        "date": large_gaps.index,
        "gap_days": large_gaps.values
    })

    return result


def first_valid_dates(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Find the first available price date and missing-data
    percentage for each ticker.
    """
    # Find the first date with a valid price
    first_dates = prices.apply(
        lambda column: column.first_valid_index()
    )

    # Calculate the percentage of missing values
    missing_fractions = prices.isnull().mean()

    # Combine the results into a DataFrame
    result = pd.DataFrame({
        "first_date": first_dates,
        "missing_frac": missing_fractions
    })

    return result
