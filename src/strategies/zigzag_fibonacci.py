# src/strategies/zigzag_fibonacci.py
from backtesting import Strategy
import numpy as np
import logging

class ZigZagFibonacciStrategy(Strategy):
    zigzag_depth = 12
    zigzag_deviation = 5
    zigzag_backstep = 3
    fib_levels = [0.382, 0.5, 0.618]
    fib_pip_tolerance = 10
    volatility_rank = None  # To be set by the main loop

    def init(self):
        self.logger = logging.getLogger(__name__)
        self.zigzag_data = None

    def get_zigzag_fibonacci(self, highs, lows):
        """Calculate ZigZag and Fibonacci levels."""
        try:
            zigzag = []
            last_high = last_low = None
            direction = 0
            for i in range(self.zigzag_depth, len(highs) - self.zigzag_backstep):
                if all(highs[i] >= highs[i-j] for j in range(1, self.zigzag_depth + 1)) and \
                   all(highs[i] >= highs[i+j] for j in range(1, self.zigzag_backstep + 1)):
                    if direction != 1 and (last_high is None or highs[i] > last_high * (1 + self.zigzag_deviation / 100)):
                        zigzag.append(('high', highs[i], i))
                        last_high = highs[i]
                        direction = 1
                elif all(lows[i] <= lows[i-j] for j in range(1, self.zigzag_depth + 1)) and \
                     all(lows[i] <= lows[i+j] for j in range(1, self.zigzag_backstep + 1)):
                    if direction != -1 and (last_low is None or lows[i] < last_low * (1 - self.zigzag_deviation / 100)):
                        zigzag.append(('low', lows[i], i))
                        last_low = lows[i]
                        direction = -1

            if len(zigzag) < 2:
                self.logger.warning("Insufficient ZigZag points for Fibonacci")
                return None

            last_point = zigzag[-1]
            second_last_point = zigzag[-2]
            swing_high = max(last_point[1], second_last_point[1])
            swing_low = min(last_point[1], second_last_point[1])
            swing_direction = 'down' if last_point[0] == 'low' else 'up'

            fib_range = swing_high - swing_low
            fib_levels = {level: swing_low + fib_range * level for level in self.fib_levels}
            return {'direction': swing_direction, 'levels': fib_levels, 'high': swing_high, 'low': swing_low}
        except Exception as e:
            self.logger.error(f"ZigZag/Fibonacci calculation failed: {str(e)}")
            return None

    def next(self):
        self.zigzag_data = self.get_zigzag_fibonacci(self.data.High, self.data.Low)
        if self.zigzag_data is None:
            return

        # Fibonacci zone check
        pip_size = 0.01 if "JPY" in self.symbol else 0.0001
        pip_tolerance = 2.0 if "JPY" in self.symbol else 20.0
        in_fib_zone = any(abs(self.data.Close[-1] - level) < pip_tolerance * pip_size for level in self.zigzag_data['levels'].values())
        fib_required = self.volatility_rank is not None and self.volatility_rank >= 3
        fib_condition = in_fib_zone or not fib_required or (self.volatility_rank is not None and self.volatility_rank < 3)

        if not fib_condition:
            return

        if self.zigzag_data['direction'] == 'up':
            self.buy()
        elif self.zigzag_data['direction'] == 'down':
            self.sell()