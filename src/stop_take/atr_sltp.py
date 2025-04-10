# src/stop_take/atr_sltp.py
import pandas as pd
import logging
import os
import json

class ATRSLTP:
    def __init__(self, config):
        self._logger = logging.getLogger(__name__)
        self.config = config
        self.timeframe = config['backtest']['timeframe']
        self.atr_period = config['parameters'].get('atr_period', 14)
        self.multiplier = self._load_multiplier()

    def _load_multiplier(self):
        params_path = os.path.join(os.path.dirname(__file__), '../../strategy_params.json')
        if os.path.isfile(params_path):
            try:
                with open(params_path, 'r') as file:
                    params = json.load(file)
                return params.get('sl_strategies', {}).get('atr', {}).get(self.timeframe, {}).get('multiplier', 1.5)
            except Exception as e:
                self._logger.error(f"Error loading ATR multiplier: {str(e)}")
                return 1.5
        return 1.5

    def init(self, data):
        self.atr = pd.Series(data['High'] - data['Low']).rolling(window=self.atr_period).mean()

    def calculate_sl(self, data, entry_price, direction):
        atr_value = self.atr.iloc[-1] if hasattr(self, 'atr') else 0.0005 * 2
        return entry_price - (atr_value * self.multiplier) if direction == 'buy' else entry_price + (atr_value * self.multiplier)

    def calculate_tp(self, data, entry_price, direction, sl_price):
        atr_value = self.atr.iloc[-1] if hasattr(self, 'atr') else 0.0005 * 2
        return entry_price + (atr_value * 2.0) if direction == 'buy' else entry_price - (atr_value * 2.0)