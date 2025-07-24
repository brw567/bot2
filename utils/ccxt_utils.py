import ccxt
import logging
import asyncio
from config import BINANCE_API_KEY, BINANCE_API_SECRET

logging.basicConfig(level=logging.INFO, filename='bot.log', filemode='a', format='%(asctime)s - %(message)s')

def get_ccxt_client(exchange='binance'):
    """
    Initialize and return a CCXT client for the specified exchange.

    Args:
        exchange (str): Exchange name (default: 'binance').

    Returns:
        ccxt.Exchange: Configured client instance.

    Note: Uses rate limiting to prevent API bans; credentials from config.py.
    """
    try:
        if exchange == 'binance':
            client = ccxt.binance({
                'apiKey': BINANCE_API_KEY,
                'secret': BINANCE_API_SECRET,
                'enableRateLimit': True
            })
            return client
        else:
            raise ValueError(f"Unsupported exchange: {exchange}")
    except Exception as e:
        logging.error(f"CCXT client initialization failed for {exchange}: {e}")
        raise

def calculate_spread(symbol_spot, symbol_futures):
    """
    Calculate the spread between spot and futures prices.

    Args:
        symbol_spot (str): Spot pair (e.g., 'BTC/USDT').
        symbol_futures (str): Futures pair (e.g., 'BTC/USDT:USDT').

    Returns:
        float: Absolute spread as a fraction of spot price.

    Note: Used by arbitrage_strategy.py; handles API errors gracefully.
    """
    try:
        client = get_ccxt_client()
        spot_ticker = client.fetch_ticker(symbol_spot)
        futures_ticker = client.fetch_ticker(symbol_futures)
        spot_price = spot_ticker['last']
        futures_price = futures_ticker['last']
        spread = abs(spot_price - futures_price) / spot_price
        logging.info(f"Spread calculated: {symbol_spot}/{symbol_futures}, spread={spread:.4f}")
        return spread
    except Exception as e:
        logging.error(f"Spread calculation failed for {symbol_spot}/{symbol_futures}: {e}")
        return 0.0

def log_trade(trade_info):
    """
    Log trade details and send Telegram notification.

    Args:
        trade_info (str): Trade details (e.g., symbol, side, amount).

    Note: Dynamic import of telegram_utils to avoid circular dependency with
          telegram_utils.py (part of immediate task fix).
    """
    try:
        from utils.telegram_utils import send_notification  # Dynamic import
        asyncio.run(send_notification(f"Trade executed: {trade_info}"))
        logging.info(f"Trade logged: {trade_info}")
    except ImportError as e:
        logging.error(f"Dynamic import of telegram_utils failed: {e}")
    except Exception as e:
        logging.error(f"Log trade failed: {e}")
