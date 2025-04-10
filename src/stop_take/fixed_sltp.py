# src/stop_take/fixed_sltp.py
import logging
import os
import json

class FixedSLTP:
    def __init__(self, config):
        self._logger = logging.getLogger(__name__)
        self.config = config
        self.timeframe = config['backtest']['timeframe']
        self.sl_distance = self._load_sl_distance()
        self.tp_multiplier = self._load_tp_multiplier()

    def _load_sl_distance(self):
        params_path = os.path.join(os.path.dirname(__file__), '../../strategy_params.json')
        if os.path.isfile(params_path):
            try:
                with open(params_path, 'r') as file:
                    params = json.load(file)
                return params.get('sl_strategies', {}).get('fixed', {}).get(self.timeframe, {}).get('distance', 0.0005)
            except Exception as e:
                self._logger.error(f"Error loading SL distance: {str(e)}")
                return 0.0005
        return 0.0005

    def _load_tp_multiplier(self):
        params_path = os.path.join(os.path.dirname(__file__), '../../strategy_params.json')
        if os.path.isfile(params_path):
            try:
                with open(params_path, 'r') as file:
                    params = json.load(file)
                return params.get('tp_strategies', {}).get('fixed', {}).get(self.timeframe, {}).get('multiplier', 2.0)
            except Exception as e:
                self._logger.error(f"Error loading TP multiplier: {str(e)}")
                return 2.0
        return 2.0

    def calculate_sl(self, data, entry_price, direction):
        return entry_price - self.sl_distance if direction == 'buy' else entry_price + self.sl_distance

    def calculate_tp(self, data, entry_price, direction, sl_price):
        return entry_price + (self.sl_distance * self.tp_multiplier) if direction == 'buy' else entry_price - (self.sl_distance * self.tp_multiplier)