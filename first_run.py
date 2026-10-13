import logging
from quantlab.data_loader import load_prices

logging.basicConfig(level=logging.INFO)
prices = load_prices()  # uses ALL, START, END from universe.py

print(prices.shape)
print(prices.head())
print(prices.tail())