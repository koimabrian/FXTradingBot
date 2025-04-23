# run_trading.py
import logging
import time
from datetime import datetime
import pandas as pd
import MetaTrader5 as mt5
from src.strategies.sma import SMAStrategy
from src.strategies.lrf import LRFStrategy
from src.strategies.zigzag_fibonacci import ZigZagFibonacciStrategy
from src.strategies.garch_volatility import GARCHVolatilityStrategy
from src.stop_take.fixed_sltp import FixedSLTP

# Setup logging
logging.basicConfig(filename='trading_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize MT5 connection
if not mt5.initialize():
    logger.error("Failed to initialize MT5")
    quit()

# Strategy parameters
SYMBOLS = ["EURUSDm", "GBPUSDm", "USDJPYm", "AUDUSDm", "NZDUSDm", "USDCADm", "USDCHFm",
           "EURJPYm", "GBPJPYm", "AUDJPYm", "NZDJPYm", "CADJPYm", "CHFJPYm",
           "EURGBPm", "EURAUDm", "USDZARm", "XAUUSDm"]
TIMEFRAME = mt5.TIMEFRAME_M1
CHECK_INTERVAL = 60  # Check every 60 seconds (1 minute)

def get_data(symbol, timeframe, lookback=200):
    """Fetch historical data."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, lookback)
    if rates is None or len(rates) == 0:
        logger.error(f"{symbol} - Failed to fetch data")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
    return df

def count_open_positions():
    """Count the number of open positions."""
    positions = mt5.positions_get()
    return len(positions) if positions else 0

def close_position(position, sltp_strategy):
    """Close a position if profit threshold is met."""
    if sltp_strategy.check_close(position):
        symbol = position.symbol
        ticket = position.ticket
        volume = position.volume
        order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.error(f"{symbol} - Failed to get tick data for closing")
            return False
        price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": "Profit Threshold Closure",
            "type_time": mt5.ORDER_TIME_GTC,
        }

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"{symbol} - Symbol info not available for closing")
            return False
        filling_mode = symbol_info.filling_mode
        if filling_mode & mt5.ORDER_FILLING_FOK:
            request["type_filling"] = mt5.ORDER_FILLING_FOK
        elif filling_mode & mt5.ORDER_FILLING_IOC:
            request["type_filling"] = mt5.ORDER_FILLING_IOC
        else:
            request["type_filling"] = mt5.ORDER_FILLING_RETURN

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"{symbol} - Position {ticket} closed")
            return True
        else:
            logger.error(f"{symbol} - Failed to close position {ticket}, Retcode: {result.retcode if result else 'No result'}")
            return False
    return False

def send_market_order(symbol, order_type, lot_size):
    """Send a market order."""
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        logger.error(f"{symbol} - Failed to get tick data for order")
        return None
    
    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    if price == 0:
        logger.error(f"{symbol} - Invalid price: {price}")
        return None

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "deviation": 20,
        "magic": 123456,
        "comment": "Combined Strategy Trade",
        "type_time": mt5.ORDER_TIME_GTC,
    }

    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        logger.error(f"{symbol} - Symbol info not available")
        return None
    filling_mode = symbol_info.filling_mode
    if filling_mode & mt5.ORDER_FILLING_FOK:
        request["type_filling"] = mt5.ORDER_FILLING_FOK
    elif filling_mode & mt5.ORDER_FILLING_IOC:
        request["type_filling"] = mt5.ORDER_FILLING_IOC
    else:
        request["type_filling"] = mt5.ORDER_FILLING_RETURN

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"{symbol} - Order placed: {order_type}, Lot: {lot_size}")
        return result
    else:
        logger.error(f"{symbol} - Order failed, Retcode: {result.retcode if result else 'No result'}")
        return None

def get_dynamic_volatility_threshold(volatility_dict, top_n=10):
    """Calculate dynamic volatility threshold based on top volatile pairs."""
    ranked_symbols = sorted(volatility_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    volatilities = [vol for _, vol in ranked_symbols]
    if not volatilities:
        return 0.0001
    median_volatility = np.median(volatilities)
    dynamic_threshold = median_volatility * 0.75
    logger.info(f"Dynamic Volatility Threshold: {dynamic_threshold}")
    return dynamic_threshold

def run_trading():
    """Run the trading loop with CombinedStrategy."""
    logger.info("Starting CombinedStrategy trading loop")
    
    # Ensure all symbols are in Market Watch
    for symbol in SYMBOLS:
        if not mt5.symbol_select(symbol, True):
            logger.error(f"{symbol} - Failed to select symbol in Market Watch")
        else:
            logger.info(f"{symbol} - Successfully selected in Market Watch")

    # Initialize strategies
    strategies = [
        SMAStrategy,
        LRFStrategy,
        ZigZagFibonacciStrategy,
        GARCHVolatilityStrategy
    ]
    config = {'profit_threshold': 5}
    sltp_strategy = FixedSLTP(config)

    while True:
        try:
            # Monitor and close positions
            positions = mt5.positions_get()
            open_positions = count_open_positions()
            if positions:
                for pos in positions:
                    if close_position(pos, sltp_strategy):
                        open_positions -= 1
                        logger.info(f"Closed position for {pos.symbol}, open positions now: {open_positions}")

            # Calculate volatility for all symbols
            volatility_dict = {}
            for symbol in SYMBOLS:
                data = get_data(symbol, TIMEFRAME)
                if data is None:
                    continue
                garch_strategy = GARCHVolatilityStrategy(data)
                volatility, var, es = garch_strategy.get_garch_volatility_var_es(data['Close'].values)
                if volatility is not None:
                    volatility_dict[symbol] = volatility

            # Calculate dynamic volatility threshold
            volatility_threshold = get_dynamic_volatility_threshold(volatility_dict)

            # Rank symbols by volatility
            ranked_symbols = sorted(volatility_dict.items(), key=lambda x: x[1], reverse=True)
            top_symbols = [sym[0] for sym in ranked_symbols[:10]]
            logger.info(f"Top 10 volatile pairs: {[(sym, vol) for sym, vol in ranked_symbols[:10]]}")

            # Process trades for each symbol
            for symbol in SYMBOLS:
                if symbol not in top_symbols:
                    continue

                # Get current price
                tick = mt5.symbol_info_tick(symbol)
                if not tick or (tick.last == 0 and tick.bid == 0 and tick.ask == 0):
                    logger.warning(f"{symbol} - Failed to get valid tick data, using last close price")
                    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, 1)
                    current_price = rates[-1]['close'] if rates and len(rates) > 0 else None
                    if current_price is None:
                        logger.error(f"{symbol} - Failed to get last close price, skipping")
                        continue
                else:
                    current_price = tick.last if tick.last != 0 else (tick.bid + tick.ask) / 2

                # Fetch data and initialize combined strategy
                data = get_data(symbol, TIMEFRAME)
                if data is None:
                    continue
                combined_strategy = CombinedStrategy([strategy(data) for strategy in strategies], sltp_strategy)
                combined_strategy.init(data, symbol, volatility_dict, volatility_threshold)

                # Execute trade
                open_positions = combined_strategy.execute(current_price)
                if open_positions > combined_strategy.open_positions:
                    order_type = mt5.ORDER_TYPE_BUY if buy_strength > sell_strength else mt5.ORDER_TYPE_SELL
                    lot_size = 0.05 if symbol in ["USDZARm", "XAUUSDm"] else 0.1
                    send_market_order(symbol, order_type, lot_size)
                    combined_strategy.open_positions = open_positions

            # Sleep for the check interval
            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            logger.error(f"Error in trading loop: {str(e)}")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    run_trading()