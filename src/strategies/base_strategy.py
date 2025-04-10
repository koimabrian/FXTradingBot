# src/strategies/base_strategy.py
from backtesting import Strategy
from backtesting.lib import crossover
import pandas as pd
import logging
import traceback
import os
import json

class BaseStrategy(Strategy):
    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)  # Call the parent Strategy constructor
        self._logger = logging.getLogger(__name__)
        # Access config and params from the class attributes set by Backtest
        self.config = getattr(self, '_config', None)
        self.timeframe = self.config['backtest']['timeframe'] if self.config else 'M1'
        self.params = getattr(self, '_params', None) or self._load_default_params()
        self.sl_distance = self.params.get('sl_distance', 0.0005)
        self.tp_multiplier = self.params.get('tp_multiplier', 2.0)

    def _load_default_params(self):
        params_path = os.path.join(os.path.dirname(__file__), '../../strategy_params.json')
        if os.path.isfile(params_path):
            try:
                with open(params_path, 'r') as file:
                    params = json.load(file)
                strategy_params = params.get('strategies', {}).get(self.__class__.__name__.lower(), {})
                return strategy_params.get(self.timeframe, {})
            except Exception as e:
                self._logger.error(f"Error loading parameters for {self.__class__.__name__}: {str(e)}")
                return {}
        return {}

    def init(self):
        raise NotImplementedError("Subclasses must implement init()")

    def next(self):
        raise NotImplementedError("Subclasses must implement next()")

    def calculate_lot_size(self, current_price):
        equity = self.config['backtest']['cash'] if self.config else 10000
        margin = self.config['backtest']['margin'] if self.config else 0.02
        return min(0.1, equity * margin / current_price)