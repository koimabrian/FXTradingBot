# src/stop_take/fixed_sltp.py (assumed existing implementation)
import logging

class FixedSLTP:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.profit_threshold = config.get('profit_threshold', 5)  # Default to 5 USD

    def custom_sl(self, price, data, direction):
        # Not used in the original strategy
        return price - 0.01 if direction == 'buy' else price + 0.01

    def custom_tp(self, price, data, direction, entry_price):
        # Not used in the original strategy
        return price + 0.02 if direction == 'buy' else price - 0.02

    def check_close(self, position):
        """Check if position should be closed based on profit threshold."""
        total_profit = position.profit + position.swap
        if total_profit >= self.profit_threshold:
            self.logger.info(f"Closing position {position.ticket} for {position.symbol} at profit: {total_profit:.2f}")
            return True
        return False