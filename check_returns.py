from quantlab.data_loader import load_prices
from quantlab.returns import log_returns, annualise

prices = load_prices()
r = log_returns(prices)

print(r.shape)
print(annualise(r).round(3))
print("Largest daily moves:")
print(r.abs().max().sort_values(ascending=False).head(5))