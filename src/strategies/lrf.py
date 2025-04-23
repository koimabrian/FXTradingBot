# src/strategies/lrf.py
from backtesting import Strategy
import numpy as np
import pandas as pd
import logging

class LRFStrategy(Strategy):
    lrf_period = 20

    def init(self):
        self.logger = logging.getLogger(__name__)
        self.lrf_forecast = self.I(self.get_lrf_forecast, self.data.Close, self.lrf_period)

    def get_lrf_forecast(self, prices, period):
        """Calculate Linear Regression Forecast."""
        try:
            closes = pd.Series(prices)
            x = np.arange(len(closes))
            coefficients = np.polyfit(x, closes, 1)
            slope, intercept = coefficients
            forecasted_price = slope * period + intercept
            return np.full_like(prices, forecasted_price)
        except Exception as e:
            self.logger.error(f"LRF calculation failed: {str(e)}")
            return np.full_like(prices, np.nan)

    def next(self):
        price_tolerance = self.data.Close[-1] * 0.001  # 0.1% tolerance
        if self.lrf_forecast[-1] > self.data.Close[-1] + price_tolerance:
            self.buy()
        elif self.lrf_forecast[-1] < self.data.Close[-1] - price_tolerance:
            self.sell()