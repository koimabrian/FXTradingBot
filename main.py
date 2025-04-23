# main.py
import argparse
from backtest_runner import BacktestRunner
import logging

def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    with open('backtest_log.txt', 'w') as f:
        pass
    logging.basicConfig(
        filename='backtest_log.txt',
        filemode='a',
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    file_handler = logging.FileHandler('backtest_log.txt')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info("Logging setup completed")
    return logger

def main():
    parser = argparse.ArgumentParser(description="FX Trading Bot")
    parser.add_argument('--mode', choices=['demo', 'live'], default='demo', help="Mode to run the bot in (demo or live)")
    parser.add_argument('--parallel', action='store_true', help="Run backtests in parallel")
    parser.add_argument('--debug', action='store_true', help="Enable debug logging")
    args = parser.parse_args()

    logger = setup_logging(args.debug)

    config = {
        'symbols': ['USDJPY', 'XAUUSD'],
        'timeframes': ['M1', 'M5', 'M15', 'H1', 'H4', 'D1'],
        'strategies': ['combined'],  # Use combined strategy
        'combined_strategies': ['sma', 'lrf', 'zigzag_fibonacci', 'garch_volatility'],  # Strategies to combine
        'backtest': {'timeframe': 'M1', 'cash': 100000, 'commission': 0.0002, 'margin': 0.1},
        'sl_strategies': ['fixed', 'atr', 'dynamic'],
        'tp_strategies': ['fixed', 'dynamic'],
        'backtest_module': 'mt5_backtest',
        'parallel': args.parallel,
        'signal': 'sma',
        'dashboard': {'port': 8050},
        'mt5': {'mode': args.mode, 'demo': {'login': 208711745, 'password': 'Brian@2025', 'server': 'Exness-MT5Trial9'}},
        'parameters': {
            'atr_period': 14,
            'profit_threshold': 5
        }
    }

    runner = BacktestRunner(config, {})
    runner.run()

if __name__ == "__main__":
    main()