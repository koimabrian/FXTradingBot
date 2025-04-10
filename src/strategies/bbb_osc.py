# src/strategies/bbb_osc.py
from backtesting import Strategy
from backtesting.lib import crossover
import ta
import pandas as pd

def crossunder(x, y):
    return crossover(y, x)

class BBBOSCStrategy(Strategy):
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
        stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14)
        self.stoch_k = self.I(lambda x: x, stoch.stoch())
        self.stoch_d = self.I(lambda x: x, stoch.stoch_signal())

    def next(self):
        if (self.data.Close[-1] <= self.bb_lower[-1] and
            crossover(self.stoch_k, self.stoch_d) and
            self.stoch_k[-1] < 20):
            self.buy(size=self.lot_size)
        elif (self.data.Close[-1] >= self.bb_upper[-1] and
              crossunder(self.stoch_k, self.stoch_d) and
              self.stoch_k[-1] > 80):
            self.sell(size=self.lot_size)