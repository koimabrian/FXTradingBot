# src/strategies/jfkps.py
from backtesting import Strategy
from backtesting.lib import crossover
import ta
import pandas as pd

def crossunder(x, y):
    return crossover(y, x)

class JFKPSStrategy(Strategy):
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
        keltner = ta.volatility.KeltnerChannel(high=df['high'], low=df['low'], close=df['close'], window=self.params.period)
        self.keltner_upper = self.I(lambda x: x, keltner.keltner_channel_hband())
        self.keltner_lower = self.I(lambda x: x, keltner.keltner_channel_lband())
        self.psar = self.I(lambda x: x, ta.trend.PSARIndicator(high=df['high'], low=df['low'], close=df['close']).psar())

    def next(self):
        if (crossover(self.data.Close, self.keltner_upper) and
            self.psar[-1] < self.data.Close[-1]):
            self.buy(size=self.lot_size)
        elif (crossunder(self.data.Close, self.keltner_lower) and
              self.psar[-1] > self.data.Close[-1]):
            self.sell(size=self.lot_size)