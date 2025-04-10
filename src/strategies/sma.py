# src/strategies/sma.py
from backtesting import Strategy
from backtesting.lib import crossover
import ta
import pandas as pd

def crossunder(x, y):
    return crossover(y, x)

class SMAStrategy(Strategy):
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
        self.sma_short = self.I(ta.trend.sma_indicator, df['close'], self.params.period // 2)
        self.sma_long = self.I(ta.trend.sma_indicator, df['close'], self.params.period)

    def next(self):
        if crossover(self.sma_short, self.sma_long):
            self.buy(size=self.lot_size)
        elif crossunder(self.sma_short, self.sma_long):
            self.sell(size=self.lot_size)