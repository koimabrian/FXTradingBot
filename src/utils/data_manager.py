# src/utils/data_manager.py
import logging
import traceback
import MetaTrader5 as mt5
import pandas as pd
import os
from datetime import datetime, timedelta

class DataManager:
    def __init__(self, data_dir='data', logger=None):
        self.data_dir = data_dir
        self.logger = logger or logging.getLogger(__name__)
        os.makedirs(self.data_dir, exist_ok=True)
        self.timeframe_map = {
            'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1
        }
        self.timeframe_intervals = {
            'M1': 60, 'M5': 300, 'M15': 900, 'H1': 3600, 'H4': 14400, 'D1': 86400
        }

    def fetch_data(self, symbol, timeframe, start_date=None, end_date=None, bars=10000):
        """Fetch data from MT5 using copy_rates_range or copy_rates_from_pos."""
        mt5_timeframe = self.timeframe_map.get(timeframe, mt5.TIMEFRAME_M1)

        if start_date and end_date:
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
            if rates is None or len(rates) == 0:
                self.logger.warning(f"No data fetched for {symbol} on {timeframe} from {start_date} to {end_date}")
                return None
            self.logger.info(f"Fetched {len(rates)} rows for {symbol} on {timeframe} using copy_rates_range")
        else:
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
            if rates is None or len(rates) == 0:
                self.logger.warning(f"No data fetched for {symbol} on {timeframe} using copy_rates_from_pos")
                return None
            self.logger.info(f"Fetched {len(rates)} rows for {symbol} on {timeframe} using copy_rates_from_pos")

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'tick_volume']].rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'
        })
        return df

    def validate_data(self, df, timeframe):
        """Validate the data in the DataFrame for consistency."""
        if df.empty:
            self.logger.warning("DataFrame is empty")
            return False

        if df[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().any().any():
            self.logger.warning("DataFrame contains missing values")
            return False

        if (df[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
            self.logger.warning("DataFrame contains negative or zero prices")
            return False

        if (df['High'] < df['Open']).any() or (df['High'] < df['Close']).any() or \
           (df['Low'] > df['Open']).any() or (df['Low'] > df['Close']).any():
            self.logger.warning("DataFrame contains inconsistent OHLC values")
            return False

        time_diff = df.index.to_series().diff().dt.total_seconds().dropna()
        expected_interval = self.timeframe_intervals[timeframe]
        if not (time_diff == expected_interval).all():
            self.logger.warning(f"DataFrame time intervals are inconsistent for {timeframe}")
            return False

        return True

    def load_data(self, symbol, timeframe, bars=10000, fetch_new=False):
        """Load data from CSV, append missing data, or fetch anew if invalid or fetch_new is True."""
        file_path = os.path.join(self.data_dir, f"{symbol}_{timeframe}_data.csv")
        current_date = datetime.now()

        existing_data = None
        if not fetch_new and os.path.isfile(file_path):
            try:
                existing_data = pd.read_csv(file_path)
                existing_data['time'] = pd.to_datetime(existing_data['time'])
                existing_data.set_index('time', inplace=True)
                self.logger.info(f"Loaded {len(existing_data)} rows from CSV for {symbol} on {timeframe}")
            except Exception as e:
                self.logger.error(f"Error loading CSV data for {symbol} on {timeframe}: {str(e)}")
                existing_data = None

        if existing_data is not None and not self.validate_data(existing_data, timeframe):
            self.logger.warning(f"Existing data for {symbol} on {timeframe} is invalid. Recreating CSV.")
            existing_data = None

        if existing_data is None:
            df = self.fetch_data(symbol, timeframe, None, None, bars)
            if df is None:
                end_date = current_date
                days_back = 30
                while days_back <= 365 and df is None:
                    start_date = end_date - timedelta(days=days_back)
                    df = self.fetch_data(symbol, timeframe, start_date, end_date)
                    if df is not None:
                        break
                    self.logger.warning(f"No data fetched for {symbol} on {timeframe} for {days_back} days back. Trying a longer range.")
                    days_back += 30

                if df is None:
                    start_date = datetime(2025, 2, 20)
                    df = self.fetch_data(symbol, timeframe, start_date, end_date)
                    if df is None:
                        self.logger.error(f"Failed to fetch data for {symbol} on {timeframe} even with known date range")
                        raise Exception(f"No data available for {symbol} on {timeframe}")

            df.to_csv(file_path)
            self.logger.info(f"Created new CSV with {len(df)} rows for {symbol} on {timeframe}")
            return df

        last_date = existing_data.index[-1]
        self.logger.info(f"Last date in CSV for {symbol} on {timeframe}: {last_date}")

        if last_date < current_date:
            start_date = last_date + timedelta(seconds=self.timeframe_intervals[timeframe])
            new_data = self.fetch_data(symbol, timeframe, start_date, current_date)
            if new_data is not None and not new_data.empty:
                combined_data = pd.concat([existing_data, new_data])
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                if self.validate_data(combined_data, timeframe):
                    combined_data.to_csv(file_path)
                    self.logger.info(f"Appended {len(new_data)} new rows to CSV for {symbol} on {timeframe}")
                else:
                    self.logger.warning(f"Combined data for {symbol} on {timeframe} is invalid. Recreating CSV.")
                    df = self.fetch_data(symbol, timeframe, None, None, bars)
                    if df is None:
                        end_date = current_date
                        days_back = 30
                        while days_back <= 365 and df is None:
                            start_date = end_date - timedelta(days=days_back)
                            df = self.fetch_data(symbol, timeframe, start_date, end_date)
                            if df is not None:
                                break
                            self.logger.warning(f"No data fetched for {symbol} on {timeframe} for {days_back} days back. Trying a longer range.")
                            days_back += 30

                        if df is None:
                            start_date = datetime(2025, 2, 20)
                            df = self.fetch_data(symbol, timeframe, start_date, end_date)
                            if df is None:
                                self.logger.error(f"Failed to fetch data for {symbol} on {timeframe} even with known date range")
                                raise Exception(f"No data available for {symbol} on {timeframe}")

                    df.to_csv(file_path)
                    self.logger.info(f"Recreated CSV with {len(df)} rows for {symbol} on {timeframe}")
                    return df
            else:
                self.logger.info(f"No new data to append for {symbol} on {timeframe}")
        else:
            self.logger.info(f"CSV data for {symbol} on {timeframe} is up to date")

        return existing_data

    def load_data_for_backtest(self, symbol, timeframe):
        """Load data from CSV for backtesting without validation."""
        file_path = os.path.join(self.data_dir, f"{symbol}_{timeframe}_data.csv")
        if not os.path.isfile(file_path):
            self.logger.warning(f"No data file found for {symbol} on {timeframe} at {file_path}")
            return None

        try:
            df = pd.read_csv(file_path)
            if df.empty:
                self.logger.warning(f"Data file for {symbol} on {timeframe} at {file_path} is empty")
                return None
            self.logger.debug(f"Columns in CSV for {symbol} on {timeframe}: {df.columns.tolist()}")
            if 'time' not in df.columns:
                self.logger.error(f"Data file for {symbol} on {timeframe} at {file_path} does not contain 'time' column")
                return None
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            self.logger.info(f"Loaded {len(df)} rows from CSV for {symbol} on {timeframe} for backtesting")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol} on {timeframe}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None