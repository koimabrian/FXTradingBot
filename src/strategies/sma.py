# src/strategies/sma.py (assumed existing implementation)
from backtesting import Strategy
import pandas as pd

class SMAStrategy(Strategy):
    sma_period = 20

    def init(self):
        self.sma = self.I(self.get_sma, self.data.Close, self.sma_period)

    def get_sma(self, prices, period):
        return pd.Series(prices).rolling(window=period).mean().values

    def next(self):
        if self.data.Close[-1] < self.sma[-1]:
            self.buy()
        elif self.data.Close[-1] > self.sma[-1]:
            self.sell()