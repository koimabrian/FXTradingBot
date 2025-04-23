# src/strategies/garch_volatility.py
from backtesting import Strategy
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm
import logging

class GARCHVolatilityStrategy(Strategy):
    garch_lookback = 200
    var_confidence = 0.95
    max_var_percent = 0.01
    top_volatile_pairs = 10
    min_profit_pips = 5
    min_profit_xauusd = 5
    exotic_pairs = ["USDZARm", "XAUUSDm"]
    volatility_dict = {}  # To be set by the main loop
    volatility_threshold = 0.0001  # To be set by the main loop

    def init(self):
        self.logger = logging.getLogger(__name__)
        self.volatility = None
        self.var = None
        self.es = None

    def get_garch_volatility_var_es(self, prices):
        """Calculate GARCH volatility, VaR, and ES."""
        try:
            returns = pd.Series(prices).pct_change().dropna() * 100
            if len(returns) < 10 or returns.var() < 1e-10:
                self.logger.warning("Insufficient data or variance for GARCH, using fallback volatility")
                volatility = returns.std() / 100 if returns.std() != 0 else 0.0001
                var_monetary = volatility * 1000 * 0.1
                es_monetary = var_monetary * 1.25
                return volatility, var_monetary, es_monetary

            model = arch_model(returns, vol='Garch', p=1, q=1, mean='Zero', dist='Normal', rescale=True)
            res = model.fit(disp='off', options={'maxiter': 200})
            forecast = res.forecast(horizon=1)
            volatility = np.sqrt(forecast.variance.iloc[-1].values[0]) / 100

            mean_return = 0
            z_score = norm.ppf(1 - self.var_confidence)
            var_return = mean_return + z_score * volatility
            es_return = mean_return - (volatility * norm.pdf(z_score) / (1 - self.var_confidence))

            position_value = 1000 * (0.05 if self.symbol in self.exotic_pairs else 0.1)
            var_monetary = position_value * abs(var_return)
            es_monetary = position_value * abs(es_return)

            return volatility, var_monetary, es_monetary
        except Exception as e:
            self.logger.error(f"GARCH/VaR/ES calculation failed: {str(e)}")
            volatility = pd.Series(prices).pct_change().std() / 100 if len(prices) > 1 else 0.0001
            var_monetary = volatility * 1000 * 0.1
            es_monetary = var_monetary * 1.25
            return volatility, var_monetary, es_monetary

    def estimate_price_movement(self, volatility, current_price):
        """Estimate price movement based on volatility."""
        pip_size = 0.0001
        if "JPY" in self.symbol:
            pip_size = 0.01
        elif self.symbol == "XAUUSDm":
            pip_size = 1.0

        expected_movement_units = current_price * volatility
        expected_movement_pips = expected_movement_units / pip_size

        spread = 2.0
        if "JPY" in self.symbol:
            spread = 0.02
        elif self.symbol == "XAUUSDm":
            spread = 2.0

        return expected_movement_pips, spread

    def next(self):
        self.volatility, self.var, self.es = self.get_garch_volatility_var_es(self.data.Close)
        if self.volatility is None or self.var is None:
            return

        max_var_allowed = 100000 * self.max_var_percent  # Assume initial cash of 100000
        if self.volatility < self.volatility_threshold or self.var > max_var_allowed:
            return

        expected_movement, spread = self.estimate_price_movement(self.volatility, self.data.Close[-1])
        min_profit = self.min_profit_xauusd if self.symbol == "XAUUSDm" else self.min_profit_pips
        if expected_movement < spread + min_profit:
            return

        # If all filters pass, allow the trade (signal will come from other strategies)
        self.buy()  # Placeholder; actual signal will be determined by combined strategy