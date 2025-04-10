# src/strategies/macd.py
from backtesting import Strategy
from backtesting.lib import crossover
import ta
import pandas as pd

def crossunder(x, y):
    return crossover(y, x)

class MACDStrategy(Strategy):
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
        macd = ta.trend.MACD(close=df['close'])
        self.macd_line = self.I(lambda x: x, macd.macd())
        self.signal_line = self.I(lambda x: x, macd.macd_signal())

    def next(self):
        if crossover(self.macd_line, self.signal_line):
            self.buy(size=self.lot_size)
        elif crossunder(self.macd_line, self.signal_line):
            self.sell(size=self.lot_size)