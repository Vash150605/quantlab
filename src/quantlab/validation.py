"""Checks on returns. They report problems; they never change the data."""
from __future__ import annotations

import pandas as pd


def extreme_moves(returns: pd.DataFrame, threshold: float = 0.20) -> pd.DataFrame:
    """List every daily return whose size exceeds the threshold."""
    stacked = returns.stack()
    out = stacked[stacked.abs() > threshold].reset_index()
    out.columns = ["date", "ticker", "log_return"]
    return out.sort_values("log_return", key=abs, ascending=False)


def stale_runs(returns: pd.DataFrame, min_run: int = 5) -> pd.DataFrame:
    """Find tickers with min_run or more zero returns in a row (min_run + 1 identical prices)."""
    rows = []
    for ticker in returns.columns:
        is_zero = returns[ticker].eq(0)
        group_id = (is_zero != is_zero.shift()).cumsum()
        run_length = is_zero.groupby(group_id).transform("sum")
        flagged = is_zero & (run_length >= min_run)
        if flagged.any():
            rows.append((ticker, int(flagged.sum())))
    return pd.DataFrame(rows, columns=["ticker", "stale_days"])