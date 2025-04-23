# backtest_runner.py
import logging
import MetaTrader5 as mt5
import pandas as pd
import traceback
from backtesting import Backtest
from src.strategies.purple_cloud import PurpleCloudStrategy
from src.strategies.jfkps import JFKPSStrategy
from src.strategies.bbb_osc import BBBOSCStrategy
from src.strategies.doda_stochastic import DodaStochasticStrategy
from src.strategies.pmo import PMOStrategy
from src.strategies.rsi import RSIStrategy
from src.strategies.macd import MACDStrategy
from src.strategies.bollinger_bands import BollingerBandsStrategy
from src.strategies.fibonacci_retracement import FibonacciRetracementStrategy
from src.strategies.atr import ATRStrategy
from src.strategies.stochastic_oscillator import StochasticOscillatorStrategy
from src.strategies.ichimoku_cloud import IchimokuCloudStrategy
from src.strategies.volume_profile import VolumeProfileStrategy
from src.strategies.pivot_points import PivotPointsStrategy
from src.strategies.sma import SMAStrategy
from src.strategies.lrf import LRFStrategy
from src.strategies.zigzag_fibonacci import ZigZagFibonacciStrategy
from src.strategies.garch_volatility import GARCHVolatilityStrategy
from src.stop_take.dynamic_sltp import DynamicSLTPStrategy
from src.stop_take.fixed_sltp import FixedSLTP
from src.stop_take.atr_sltp import ATRSLTP
from src.utils.data_manager import DataManager
from multiprocessing import Pool, Manager
import webbrowser
import time
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm
import itertools
import os
import numpy as np

# Import dash and plotly for interactive dashboard
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Suppress numpy and backtesting warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

class CombinedStrategy:
    def __init__(self, strategies, sltp_strategy):
        self.strategies = strategies
        self.sltp_strategy = sltp_strategy
        self.logger = logging.getLogger(__name__)

    def init(self, data, symbol, volatility_dict=None, volatility_threshold=0.0001):
        self.data = data
        self.symbol = symbol
        self.volatility_dict = volatility_dict if volatility_dict is not None else {}
        self.volatility_threshold = volatility_threshold
        self.open_positions = 0

        # Initialize all strategies
        for strategy in self.strategies:
            strategy.data = data
            strategy.symbol = symbol
            if isinstance(strategy, ZigZagFibonacciStrategy):
                strategy.volatility_rank = list(self.volatility_dict.keys()).index(symbol) if symbol in self.volatility_dict else len(self.volatility_dict)
            if isinstance(strategy, GARCHVolatilityStrategy):
                strategy.volatility_dict = self.volatility_dict
                strategy.volatility_threshold = self.volatility_threshold
            strategy.init()

    def execute(self, current_price):
        if self.open_positions >= 20:
            self.logger.info(f"Max positions (20) reached. Skipping trade for {self.symbol}.")
            return self.open_positions

        # Collect signals from all strategies
        buy_strength = 0
        sell_strength = 0
        filters_passed = True

        for strategy in self.strategies:
            # Simulate the strategy's next() to determine its signal
            strategy.data.Close = np.append(strategy.data.Close, current_price)
            strategy.data.High = np.append(strategy.data.High, current_price)
            strategy.data.Low = np.append(strategy.data.Low, current_price)
            strategy.data.Open = np.append(strategy.data.Open, current_price)
            strategy.data.index = np.append(strategy.data.index, strategy.data.index[-1] + pd.Timedelta(minutes=1))

            strategy.next()

            # Check the strategy's position to determine its signal
            if strategy.position.is_long:
                buy_strength += 1
            elif strategy.position.is_short:
                sell_strength += 1

            # Check filters (e.g., GARCHVolatilityStrategy)
            if isinstance(strategy, GARCHVolatilityStrategy):
                if strategy.volatility < strategy.volatility_threshold or strategy.var > 100000 * strategy.max_var_percent:
                    filters_passed = False
                expected_movement, spread = strategy.estimate_price_movement(strategy.volatility, current_price)
                min_profit = strategy.min_profit_xauusd if strategy.symbol == "XAUUSDm" else strategy.min_profit_pips
                if expected_movement < spread + min_profit:
                    filters_passed = False

            # Reset the strategy's position for the next iteration
            strategy.position.close()

        if not filters_passed:
            self.logger.info(f"{self.symbol} - Filters not passed (volatility, VaR, or profitability).")
            return self.open_positions

        # Execute trade based on combined signal strength
        if buy_strength > sell_strength:
            self.logger.info(f"{self.symbol} - Combined BUY signal (Buy: {buy_strength}, Sell: {sell_strength})")
            lot_size = 0.05 if self.symbol in ["USDZARm", "XAUUSDm"] else 0.1
            # Simulate a buy order (in backtesting, this would be handled by the framework)
            self.open_positions += 1
        elif sell_strength > buy_strength:
            self.logger.info(f"{self.symbol} - Combined SELL signal (Buy: {buy_strength}, Sell: {sell_strength})")
            lot_size = 0.05 if self.symbol in ["USDZARm", "XAUUSDm"] else 0.1
            # Simulate a sell order
            self.open_positions += 1
        else:
            self.logger.info(f"{self.symbol} - No trade: Buy({buy_strength}), Sell({sell_strength})")

        return self.open_positions

    def check_close(self, position):
        """Delegate position closure to the SL/TP strategy."""
        return self.sltp_strategy.check_close(position)

class BacktestRunner:
    def __init__(self, config, strategy_params):
        self._logger = logging.getLogger(__name__)
        self.config = config
        self.timeframes = config['timeframes']
        self.strategies = config['strategies']
        self.sl_strategies = config['sl_strategies']
        self.tp_strategies = config['tp_strategies']
        self.cash = config['backtest']['cash']
        self.commission = config['backtest']['commission']
        self.margin = config['backtest']['margin']
        self.sl = config.get('sl')
        self.tp = config.get('tp')
        self.backtest_module = config.get('backtest_module', config.get('default_backtest_module', 'mt5_backtest'))
        self.signal = config.get('signal', config.get('default_signal', 'sma'))
        self.parallel = config.get('parallel', False)
        self.custom_params = config.get('custom_params', {})
        self.strategy_params = strategy_params

        # Initialize results, best_params, strategy_rankings, and sltp_rankings
        if self.parallel:
            self.manager = Manager()
            self.results = self.manager.dict()
            self.best_params = self.manager.dict()
            self.strategy_rankings = self.manager.list()
            self.sltp_rankings = self.manager.list()
        else:
            self.manager = None
            self.results = {}
            self.best_params = {}
            self.strategy_rankings = []
            self.sltp_rankings = []

        self.data_manager = DataManager(data_dir='data', logger=self._logger)
        self.symbols = self.get_available_symbols()
        self.initialize_mt5()

    def get_available_symbols(self):
        """Get the list of symbols for which data is available in the data folder."""
        symbols = set()
        data_dir = 'data'
        if not os.path.exists(data_dir):
            self._logger.error("Data directory does not exist")
            return []

        for filename in os.listdir(data_dir):
            if filename.endswith('_data.csv'):
                symbol = filename.split('_')[0]
                symbols.add(symbol)
        symbols = list(symbols)
        self._logger.info(f"Found data for symbols: {symbols}")
        return symbols

    def initialize_mt5(self):
        if self.backtest_module == 'mt5_backtest':
            try:
                if not mt5.initialize():
                    self._logger.error("Failed to initialize MT5")
                    raise Exception("MT5 initialization failed")
                mode = self.config['mt5']['mode']
                credentials = self.config['mt5'][mode]
                if not mt5.login(credentials['login'], password=credentials['password'], server=credentials['server']):
                    self._logger.error(f"Failed to login to MT5: {mt5.last_error()}")
                    raise Exception("MT5 login failed")
                self._logger.info(f"MT5 initialized in {mode} mode with login {credentials['login']}")
                for symbol in self.symbols:
                    if not mt5.symbol_select(symbol, True):
                        self._logger.error(f"Failed to add {symbol} to Market Watch")
                        raise Exception(f"Failed to add {symbol} to Market Watch")
                    self._logger.info(f"Added {symbol} to Market Watch")
            except Exception as e:
                self._logger.error(f"Error initializing MT5: {str(e)}")
                self._logger.error(traceback.format_exc())
                with open('terminal_log.txt', 'a') as f:
                    f.write(f"Error initializing MT5: {str(e)}\n")
                    f.write(traceback.format_exc() + "\n")
                raise

    def load_data(self, symbol, timeframe):
        """Load data using DataManager."""
        try:
            return self.data_manager.load_data_for_backtest(symbol, timeframe)
        except Exception as e:
            self._logger.error(f"Failed to load data for {symbol} on {timeframe}: {str(e)}")
            with open('terminal_log.txt', 'a') as f:
                f.write(f"Failed to load data for {symbol} on {timeframe}: {str(e)}\n")
                f.write(traceback.format_exc() + "\n")
            return None

    def get_strategy_instance(self, strategy_name):
        strategy_map = {
            'purple_cloud': PurpleCloudStrategy,
            'jfkps': JFKPSStrategy,
            'bbb_osc': BBBOSCStrategy,
            'doda_stochastic': DodaStochasticStrategy,
            'pmo': PMOStrategy,
            'rsi': RSIStrategy,
            'macd': MACDStrategy,
            'bollinger_bands': BollingerBandsStrategy,
            'fibonacci_retracement': FibonacciRetracementStrategy,
            'atr': ATRStrategy,
            'stochastic_oscillator': StochasticOscillatorStrategy,
            'ichimoku_cloud': IchimokuCloudStrategy,
            'volume_profile': VolumeProfileStrategy,
            'pivot_points': PivotPointsStrategy,
            'sma': SMAStrategy,
            'lrf': LRFStrategy,
            'zigzag_fibonacci': ZigZagFibonacciStrategy,
            'garch_volatility': GARCHVolatilityStrategy
        }
        strategy_class = strategy_map.get(strategy_name.lower())
        if not strategy_class:
            self._logger.error(f"Unknown strategy: {strategy_name}")
            raise ValueError(f"Unknown strategy: {strategy_name}")
        params = self.strategy_params.get('strategies', {}).get(strategy_name.lower(), {}).get(self.timeframe, {})
        for key, value in self.custom_params.items():
            if key.startswith(strategy_name.lower()):
                param_key = key.split('_', 1)[1]
                params[param_key] = float(value)
        strategy_class._config = self.config
        strategy_class._params = params
        return strategy_class

    def get_sltp_instance(self, sl_strategy_name, tp_strategy_name):
        sl_strategy = FixedSLTP(self.config) if sl_strategy_name == 'fixed' else \
                      ATRSLTP(self.config) if sl_strategy_name == 'atr' else \
                      DynamicSLTPStrategy(self.config) if sl_strategy_name == 'dynamic' else FixedSLTP(self.config)
        tp_strategy = FixedSLTP(self.config) if tp_strategy_name == 'fixed' else \
                      DynamicSLTPStrategy(self.config) if tp_strategy_name == 'dynamic' else FixedSLTP(self.config)
        return sl_strategy, tp_strategy

    def run_backtest(self, symbol, timeframe, strategy_name, sl_strategy_name, tp_strategy_name, strategy_params=None, sl_params=None, tp_params=None):
        self._logger.info(f"Running backtest for {symbol} on {timeframe} with strategy {strategy_name}, SL strategy {sl_strategy_name}, TP strategy {tp_strategy_name}, strategy params {strategy_params}, SL params {sl_params}, TP params {tp_params}")
        data = self.load_data(symbol, timeframe)
        if data is None:
            self._logger.warning(f"Skipping backtest for {symbol} on {timeframe} due to missing data")
            return None

        self._logger.debug(f"Loaded data for {symbol} on {timeframe}: {data.head()}")

        # Calculate volatility for all symbols to set volatility threshold
        volatility_dict = {}
        for sym in self.symbols:
            sym_data = self.load_data(sym, timeframe)
            if sym_data is None:
                continue
            garch_strategy = GARCHVolatilityStrategy(sym_data)
            volatility, var, es = garch_strategy.get_garch_volatility_var_es(sym_data['Close'].values)
            if volatility is not None:
                volatility_dict[sym] = volatility

        # Calculate dynamic volatility threshold
        ranked_symbols = sorted(volatility_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        volatilities = [vol for _, vol in ranked_symbols]
        volatility_threshold = np.median(volatilities) * 0.75 if volatilities else 0.0001

        # Handle combined strategy
        if strategy_name.lower() == 'combined':
            strategy_names = self.config.get('combined_strategies', ['sma', 'lrf', 'zigzag_fibonacci', 'garch_volatility'])
            strategies = [self.get_strategy_instance(name) for name in strategy_names]
            sl_strategy, tp_strategy = self.get_sltp_instance(sl_strategy_name, tp_strategy_name)
            combined_strategy = CombinedStrategy(strategies, sl_strategy)
            combined_strategy.init(data, symbol, volatility_dict, volatility_threshold)
        else:
            strategy_class = self.get_strategy_instance(strategy_name)
            sl_strategy, tp_strategy = self.get_sltp_instance(sl_strategy_name, tp_strategy_name)

        if sl_params:
            sl_strategy.custom_sl = lambda x, y, z: x - sl_params['distance'] if z == 'buy' else x + sl_params['distance']
        if tp_params:
            tp_strategy.custom_tp = lambda x, y, z, w: x + tp_params['multiplier'] * (x - w) if z == 'buy' else x - tp_params['multiplier'] * (w - x)

        bt = Backtest(
            data,
            strategy_class if strategy_name.lower() != 'combined' else CombinedStrategy,
            cash=100000,
            commission=self.commission,
            margin=0.1,
        )
        try:
            if strategy_params is None:
                strategy_params = {'period': 20, 'volatility_factor': 1.5}
            stats = bt.run(**strategy_params)
            self._logger.info(f"Backtest completed for {symbol} on {timeframe} with {strategy_name}, SL: {sl_strategy_name}, TP: {tp_strategy_name}")

            result_entry = {
                'data': data,
                'stats': stats,
                'strategy_instance': bt._strategy,
                'strategy_params': strategy_params,
                'sl_params': sl_params,
                'tp_params': tp_params,
                'strategy_name': strategy_name,
                'sl_strategy': sl_strategy_name,
                'tp_strategy': tp_strategy_name
            }
            return symbol, timeframe, f"{strategy_name}_{sl_strategy_name}_{tp_strategy_name}_{str(strategy_params)}", result_entry
        except Exception as e:
            self._logger.error(f"Error running backtest for {symbol} on {timeframe} with {strategy_name}, SL: {sl_strategy_name}, TP: {tp_strategy_name}: {str(e)}")
            self._logger.error(traceback.format_exc())
            with open('terminal_log.txt', 'a') as f:
                f.write(f"Error running backtest for {symbol} on {timeframe} with {strategy_name}, SL: {sl_strategy_name}, TP: {tp_strategy_name}: {str(e)}\n")
                f.write(traceback.format_exc() + "\n")
            return None

    def optimize_parameters(self, symbol, timeframe):
        try:
            periods = [5, 10, 20, 50, 100]
            volatility_factors = [1.0, 1.5, 2.0, 3.0]
            sl_distances = [0.0003, 0.0005, 0.001, 0.005, 0.01, 0.02]
            tp_multipliers = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

            param_combinations = list(itertools.product(periods, volatility_factors, sl_distances, tp_multipliers))
            self._logger.info(f"Running grid search for {symbol} on {timeframe} with {len(param_combinations)} parameter combinations per strategy and SL/TP combination")

            for strategy_name in self.strategies:
                for sl_strategy_name in self.sl_strategies:
                    for tp_strategy_name in self.tp_strategies:
                        best_stats = None
                        best_params = None
                        best_return = float('-inf')

                        for period, volatility_factor, sl_distance, tp_multiplier in param_combinations:
                            strategy_params = {'period': period, 'volatility_factor': volatility_factor}
                            sl_params = {'distance': sl_distance}
                            tp_params = {'multiplier': tp_multiplier}

                            result = self.run_backtest(symbol, timeframe, strategy_name, sl_strategy_name, tp_strategy_name, strategy_params, sl_params, tp_params)
                            if result is None:
                                continue

                            sym, tf, key, result_entry = result
                            stats = result_entry['stats']
                            current_return = stats['Return [%]']
                            if current_return > best_return:
                                best_return = current_return
                                best_stats = stats
                                best_params = {
                                    'strategy_params': strategy_params,
                                    'sl_params': sl_params,
                                    'tp_params': tp_params
                                }

                            if self.parallel:
                                if sym not in self.results:
                                    self.results[sym] = self.manager.dict()
                                if tf not in self.results[sym]:
                                    self.results[sym][tf] = self.manager.dict()
                                self.results[sym][tf][key] = result_entry
                            else:
                                if sym not in self.results:
                                    self.results[sym] = {}
                                if tf not in self.results[sym]:
                                    self.results[sym][tf] = {}
                                self.results[sym][tf][key] = result_entry

                        if best_params:
                            key = f"{symbol}_{timeframe}_{strategy_name}_{sl_strategy_name}_{tp_strategy_name}"
                            best_entry = {
                                'best_params': best_params,
                                'return': best_return,
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'strategy_name': strategy_name,
                                'sl_strategy': sl_strategy_name,
                                'tp_strategy': tp_strategy_name
                            }
                            ranking_entry = {
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'strategy': strategy_name,
                                'sl_strategy': sl_strategy_name,
                                'tp_strategy': tp_strategy_name,
                                'return': best_return
                            }
                            if self.parallel:
                                self.best_params[key] = best_entry
                                self.strategy_rankings.append(ranking_entry)
                                self.sltp_rankings.append(ranking_entry)
                            else:
                                self.best_params[key] = best_entry
                                self.strategy_rankings.append(ranking_entry)
                                self.sltp_rankings.append(ranking_entry)
                            self._logger.info(f"Best parameters for {key}: {best_params} with return {best_return}%")
                        else:
                            self._logger.warning(f"No successful backtests for {symbol} on {timeframe} with {strategy_name}, SL: {sl_strategy_name}, TP: {tp_strategy_name}")
        except Exception as e:
            self._logger.error(f"Error in optimize_parameters for {symbol} on {timeframe}: {str(e)}")
            self._logger.error(traceback.format_exc())
            raise

    def create_dashboard(self):
        app = Dash(__name__)

        strategy_rankings = list(self.strategy_rankings)
        sltp_rankings = list(self.sltp_rankings)
        strategy_rankings = sorted(strategy_rankings, key=lambda x: x['return'], reverse=True)
        sltp_rankings = sorted(sltp_rankings, key=lambda x: x['return'], reverse=True)

        results = {}
        for symbol in self.results:
            results[symbol] = {}
            for timeframe in self.results[symbol]:
                results[symbol][timeframe] = {}
                for key, value in self.results[symbol][timeframe].items():
                    results[symbol][timeframe][key] = value

        best_params = dict(self.best_params)

        strategy_ranking_df = pd.DataFrame(strategy_rankings)
        sltp_ranking_df = pd.DataFrame(sltp_rankings)

        tabs = []
        rankings_tab = dcc.Tab(label='Rankings', children=[
            html.H2('Strategy Rankings'),
            dcc.Markdown(strategy_ranking_df.to_markdown() if not strategy_ranking_df.empty else "No rankings available"),
            html.H2('SL/TP Strategy Rankings'),
            dcc.Markdown(sltp_ranking_df.to_markdown() if not sltp_ranking_df.empty else "No rankings available")
        ])
        tabs.append(rankings_tab)

        for symbol in results:
            symbol_tabs = []
            for timeframe in results[symbol]:
                timeframe_tabs = []
                for key, result in results[symbol][timeframe].items():
                    stats = result['stats']
                    strategy_params = result['strategy_params']
                    sl_params = result['sl_params']
                    tp_params = result['tp_params']
                    strategy_name = result['strategy_name']
                    sl_strategy = result['sl_strategy']
                    tp_strategy = result['tp_strategy']

                    equity = stats['_equity_curve']['Equity']
                    equity_fig = px.line(
                        x=equity.index,
                        y=equity,
                        title=f'{symbol} - {timeframe} - {strategy_name} (SL: {sl_strategy}, TP: {tp_strategy}, Params: {key})',
                        labels={'x': 'Time', 'y': 'Equity'}
                    )

                    metrics = {
                        'Return (%)': stats['Return [%]'],
                        'Buy & Hold Return (%)': stats['Buy & Hold Return [%]'],
                        'Sharpe Ratio': stats['Sharpe Ratio'],
                        'Sortino Ratio': stats['Sortino Ratio'],
                        'Max Drawdown (%)': stats['Max. Drawdown [%]'],
                        'Win Rate (%)': stats['Win Rate [%]'],
                        'Number of Trades': stats['# Trades']
                    }
                    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])

                    best_params_key = f"{symbol}_{timeframe}_{strategy_name}_{sl_strategy}_{tp_strategy}"
                    best_params_text = "No best parameters found."
                    if best_params_key in best_params:
                        best_params_entry = best_params[best_params_key]
                        best_params_text = f"Best Parameters: Strategy: {best_params_entry['best_params']['strategy_params']}, SL: {best_params_entry['best_params']['sl_params']}, TP: {best_params_entry['best_params']['tp_params']}"

                    timeframe_tabs.append(
                        html.Div([
                            html.H3(f'{strategy_name} Results (SL: {sl_strategy}, TP: {tp_strategy}, Params: {key})'),
                            html.H4('Equity Curve'),
                            dcc.Graph(figure=equity_fig),
                            html.H4('Performance Metrics'),
                            dcc.Markdown(metrics_df.to_markdown()),
                            html.H4('Parameters'),
                            html.P(f"Strategy Params: {strategy_params}"),
                            html.P(f"SL Params: {sl_params}"),
                            html.P(f"TP Params: {tp_params}"),
                            html.H4('Best Parameters'),
                            html.P(best_params_text)
                        ])
                    )

                symbol_tabs.append(dcc.Tab(label=timeframe, children=timeframe_tabs))

            tabs.append(dcc.Tab(label=symbol, children=[
                dcc.Tabs(id=f'timeframe-tabs-{symbol}', children=symbol_tabs)
            ]))

        app.layout = html.Div([
            html.H1('FX Trading Bot Backtest Dashboard'),
            dcc.Tabs(id='symbol-tabs', children=tabs)
        ])

        port = self.config['dashboard']['port']
        self._logger.info(f"Starting dashboard at http://localhost:{port}")
        app.run(debug=False, port=port)

    def run(self):
        try:
            total_steps = len(self.symbols) * len(self.timeframes)
            with tqdm(total=total_steps, desc="Running Backtests") as pbar:
                if self.parallel:
                    pool = Pool()
                    results = []
                    for symbol in self.symbols:
                        for timeframe in self.timeframes:
                            self.config['backtest']['timeframe'] = timeframe
                            self.timeframe = timeframe
                            result = pool.apply_async(self.optimize_parameters, args=(symbol, timeframe))
                            results.append(result)
                            pbar.update(1)
                    for result in results:
                        try:
                            result.get()
                        except Exception as e:
                            self._logger.error(f"Error in parallel backtest: {str(e)}")
                            self._logger.error(traceback.format_exc())
                            with open('terminal_log.txt', 'a') as f:
                                f.write(f"Error in parallel backtest: {str(e)}\n")
                                f.write(traceback.format_exc() + "\n")
                    pool.close()
                    pool.join()
                    pool.terminate()
                else:
                    for symbol in self.symbols:
                        for timeframe in self.timeframes:
                            self.config['backtest']['timeframe'] = timeframe
                            self.timeframe = timeframe
                            self.optimize_parameters(symbol, timeframe)
                            pbar.update(1)
            if not self.results:
                self._logger.error("No backtests were successful due to missing data for all symbol-timeframe pairs")
                raise Exception("No data available for backtesting")
            self.create_dashboard()
        except Exception as e:
            self._logger.error(f"Error running backtest: {str(e)}")
            self._logger.error(traceback.format_exc())
            with open('terminal_log.txt', 'a') as f:
                f.write(f"Error running backtest: {str(e)}\n")
                f.write(traceback.format_exc() + "\n")
            raise
        finally:
            if self.backtest_module == 'mt5_backtest':
                mt5.shutdown()

if __name__ == "__main__":
    config = {
        'symbols': ['USDJPY', 'XAUUSD'],
        'timeframes': ['M1', 'M5', 'M15', 'H1', 'H4', 'D1'],
        'strategies': ['combined'],
        'combined_strategies': ['sma', 'lrf', 'zigzag_fibonacci', 'garch_volatility'],
        'backtest': {'timeframe': 'M1', 'cash': 100000, 'commission': 0.0002, 'margin': 0.1},
        'sl_strategies': ['fixed', 'atr', 'dynamic'],
        'tp_strategies': ['fixed', 'dynamic'],
        'backtest_module': 'mt5_backtest',
        'parallel': True,
        'signal': 'sma',
        'dashboard': {'port': 8050},
        'mt5': {'mode': 'demo', 'demo': {'login': 208711745, 'password': 'Brian@2025', 'server': 'Exness-MT5Trial9'}},
        'parameters': {
            'atr_period': 14,
            'profit_threshold': 5
        }
    }
    runner = BacktestRunner(config, {})
    runner.run()