# src/strategies/volume_profile.py
from backtesting import Strategy
from backtesting.lib import crossover
import ta
import pandas as pd

def crossunder(x, y):
    return crossover(y, x)

class VolumeProfileStrategy(Strategy):
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
        self.vwma = self.I(ta.volume.vwma, df['close'], df['volume'], self.params.period)

    def next(self):
        if crossover(self.data.Close, self.vwma):
            self.buy(size=self.lot_size)
        elif crossunder(self.data.Close, self.vwma):
            self.sell(size=self.lot_size)