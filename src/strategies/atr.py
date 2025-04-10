# src/strategies/atr.py
from backtesting import Strategy
from backtesting.lib import crossover
import ta
import pandas as pd

def crossunder(x, y):
    return crossover(y, x)

class ATRStrategy(Strategy):
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
        self.atr = self.I(lambda x: x, ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range())
        self.sma = self.I(ta.trend.sma_indicator, df['close'], self.params.period)

    def next(self):
        if (crossover(self.data.Close, self.sma) and
            self.atr[-1] > self.atr[-2]):
            self.buy(size=self.lot_size)
        elif (crossunder(self.data.Close, self.sma) and
              self.atr[-1] > self.atr[-2]):
            self.sell(size=self.lot_size)