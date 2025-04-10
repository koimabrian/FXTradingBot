# src/strategies/purple_cloud.py
from backtesting import Strategy
from backtesting.lib import crossover
import ta
import pandas as pd

def crossunder(x, y):
    """Returns True if x crosses below y."""
    return crossover(y, x)

class PurpleCloudStrategy(Strategy):
    period = 20
    volatility_factor = 1.5
    lot_size = 0.01  # Micro lot

    def init(self):
        df = pd.DataFrame({
            'open': self.data.Open,
            'high': self.data.High,
            'low': self.data.Low,
            'close': self.data.Close,
            'volume': self.data.Volume
        })
        ichimoku = ta.trend.IchimokuIndicator(high=df['high'], low=df['low'])
        self.tenkan_sen = self.I(lambda x: x, ichimoku.ichimoku_conversion_line())
        self.kijun_sen = self.I(lambda x: x, ichimoku.ichimoku_base_line())
        self.senkou_span_a = self.I(lambda x: x, ichimoku.ichimoku_a())
        self.senkou_span_b = self.I(lambda x: x, ichimoku.ichimoku_b())

    def next(self):
        if (crossover(self.tenkan_sen, self.kijun_sen) and
            self.data.Close[-1] > self.senkou_span_a[-1] and
            self.data.Close[-1] > self.senkou_span_b[-1]):
            self.buy(size=self.lot_size)
        elif (crossunder(self.tenkan_sen, self.kijun_sen) and
              self.data.Close[-1] < self.senkou_span_a[-1] and
              self.data.Close[-1] < self.senkou_span_b[-1]):
            self.sell(size=self.lot_size)