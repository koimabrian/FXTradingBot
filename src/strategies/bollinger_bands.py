# src/strategies/bollinger_bands.py
from backtesting import Strategy
from backtesting.lib import crossover
import ta
import pandas as pd

def crossunder(x, y):
    return crossover(y, x)

class BollingerBandsStrategy(Strategy):
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
        bb = ta.volatility.BollingerBands(close=df['close'], window=self.params.period)
        self.bb_upper = self.I(lambda x: x, bb.bollinger_hband())
        self.bb_lower = self.I(lambda x: x, bb.bollinger_lband())
        self.bb_middle = self.I(lambda x: x, bb.bollinger_mavg())

    def next(self):
        if crossover(self.data.Close, self.bb_lower):
            self.buy(size=self.lot_size)
        elif crossunder(self.data.Close, self.bb_upper):
            self.sell(size=self.lot_size)