# src/strategies/pmo.py
from backtesting import Strategy
from backtesting.lib import crossover
import ta
import pandas as pd

def crossunder(x, y):
    return crossover(y, x)

class PMOStrategy(Strategy):
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
        self.ao = self.I(lambda x: x, ta.momentum.AwesomeOscillator(high=df['high'], low=df['low']))

    def next(self):
        if crossover(self.ao, 0):
            self.buy(size=self.lot_size)
        elif crossunder(self.ao, 0):
            self.sell(size=self.lot_size)