# src/strategies/rsi.py
from backtesting import Strategy
from backtesting.lib import crossover
import ta
import pandas as pd

def crossunder(x, y):
    return crossover(y, x)

class RSIStrategy(Strategy):
    period = 20
    volatility_factor = 1.5
    lot_size = 0.01

    def init(self):
        df = pd.DataFrame({
            'open': self.data.Open,
            'high': self.data.High,
            'low': self.data.Low,
            'close': self.data.Close,
            'volume': self.data.Volume
        })
        self.rsi = self.I(lambda x: x, ta.momentum.RSIIndicator(close=df['close'], window=14).rsi())

    def next(self):
        if crossover(self.rsi, 30):
            self.buy(size=self.lot_size)
        elif crossunder(self.rsi, 70):
            self.sell(size=self.lot_size)