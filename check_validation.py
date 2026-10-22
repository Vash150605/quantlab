from quantlab.data_loader import load_prices
from quantlab.returns import log_returns
from quantlab.validation import extreme_moves, stale_runs, calendar_gaps, first_valid_dates

r = log_returns(load_prices())

print("Extreme moves (|log return| > 0.20):")
print(extreme_moves(r).to_string(index=False))
print()
print("Stale runs:")
print(stale_runs(r))

prices = load_prices()
print()
print("Calendar gaps (> 5 days):")
print(calendar_gaps(prices).to_string(index=False))
print()
print("First valid date and missing fraction:")
print(first_valid_dates(prices))