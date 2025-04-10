# setup_data.py
import argparse
import yaml
import logging
import traceback
import os
from src.utils.data_manager import DataManager
from datetime import datetime
from tqdm import tqdm
import MetaTrader5 as mt5

def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    # Clear the log file by opening it in write mode
    with open('data_setup_log.txt', 'w') as f:
        pass
    # Configure logging to write only to the file, not the terminal
    logging.basicConfig(
        filename='data_setup_log.txt',
        filemode='a',
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    # Remove any handlers that might output to the terminal
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    file_handler = logging.FileHandler('data_setup_log.txt')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info("Logging setup completed")
    return logger

def load_config(config_path, config_type):
    logger = logging.getLogger(__name__)
    if os.path.isfile(config_path):
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Error loading {config_type} config file {config_path}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    absolute_path = os.path.join(os.path.dirname(__file__), config_path)
    if os.path.isfile(absolute_path):
        try:
            with open(absolute_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.warning(f"Using default {config_type} config file at {absolute_path} since {config_path} was not found")
            return config
        except Exception as e:
            logger.error(f"Error loading default {config_type} config file {absolute_path}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    logger.error(f"{config_type} config file not found at {config_path} or {absolute_path}")
    raise FileNotFoundError(f"{config_type} config file not found at {config_path} or {absolute_path}")

def initialize_mt5(config):
    logger = logging.getLogger(__name__)
    try:
        if not mt5.initialize():
            logger.error("Failed to initialize MT5")
            raise Exception("MT5 initialization failed")
        mode = config['mt5']['mode']
        credentials = config['mt5'][mode]
        if not mt5.login(credentials['login'], password=credentials['password'], server=credentials['server']):
            logger.error(f"Failed to login to MT5: {mt5.last_error()}")
            raise Exception("MT5 login failed")
        logger.info(f"MT5 initialized in {mode} mode with login {credentials['login']}")
        for symbol in config['data']['pairs']:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to add {symbol} to Market Watch")
                raise Exception(f"Failed to add {symbol} to Market Watch")
            logger.info(f"Added {symbol} to Market Watch")
    except Exception as e:
        logger.error(f"Error initializing MT5: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def main():
    parser = argparse.ArgumentParser(description="Setup data for FX Trading Bot backtesting")
    parser.add_argument('--mode', type=str, default='demo', choices=['demo', 'live'], help='Mode to run in')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--fetch-new', action='store_true', help='Fetch new data, overwriting existing data')
    parser.add_argument('--bars', type=int, default=10000, help='Number of bars to fetch')
    parser.add_argument('--data-config', type=str, default='config/data_config.yaml', help='Path to data configuration file')
    parser.add_argument('--demo-config', type=str, default='config/demo_config.yaml', help='Path to demo configuration file')
    parser.add_argument('--live-config', type=str, default='config/live_config.yaml', help='Path to live configuration file')
    args = parser.parse_args()

    # Clear terminal_log.txt
    with open('terminal_log.txt', 'w') as f:
        pass

    logger = setup_logging(args.debug)

    try:
        # Load configurations
        data_config = load_config(args.data_config, 'data')
        if args.mode == 'demo':
            mode_config = load_config(args.demo_config, 'demo')
        else:
            mode_config = load_config(args.live_config, 'live')

        # Merge configurations
        config = data_config.copy()
        config.update(mode_config)
        config['data']['bars'] = args.bars
        config['data']['fetch_new'] = args.fetch_new

        # Initialize MT5
        initialize_mt5(config)

        # Setup DataManager
        data_manager = DataManager(data_dir='data', logger=logger)

        # Calculate total steps for progress bar
        total_steps = len(config['data']['pairs']) * len(config['data']['timeframes'])

        # Fetch data for each pair and timeframe with progress bar
        with tqdm(total=total_steps, desc="Fetching Data") as pbar:
            for pair in config['data']['pairs']:
                for timeframe in config['data']['timeframes']:
                    try:
                        if args.fetch_new:
                            # Force fetch new data by removing existing CSV
                            file_path = os.path.join('data', f"{pair}_{timeframe}_data.csv")
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                logger.info(f"Removed existing CSV file: {file_path}")
                        data_manager.load_data(pair, timeframe, bars=args.bars, fetch_new=args.fetch_new)
                    except Exception as e:
                        logger.error(f"Failed to setup data for {pair} on {timeframe}: {str(e)}")
                        with open('terminal_log.txt', 'a') as f:
                            f.write(f"Failed to setup data for {pair} on {timeframe}: {str(e)}\n")
                            f.write(traceback.format_exc() + "\n")
                        continue
                    finally:
                        pbar.update(1)

    except Exception as e:
        logger.error(f"Error setting up data: {str(e)}")
        logger.error(traceback.format_exc())
        with open('terminal_log.txt', 'a') as f:
            f.write(f"Error setting up data: {str(e)}\n")
            f.write(traceback.format_exc() + "\n")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()