import ccxt
import logging
from config import BINANCE_API_KEY, BINANCE_API_SECRET

logging.basicConfig(level=logging.INFO, filename='bot.log', filemode='a', format='%(asctime)s - %(message)s')

def get_binance_client():
    """
    Initialize and return a CCXT Binance client with API credentials.

    Returns:
        ccxt.binance: Configured Binance client.

    Note: Uses rate limiting to prevent API bans; credentials from config.py.
    """
    try:
        client = ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_API_SECRET,
            'enableRateLimit': True
        })
        return client
    except Exception as e:
        logging.error(f"Binance client initialization failed: {e}")
        raise

def execute_trade(symbol, side, amount, price=None):
    """
    Execute a market or limit order on Binance.

    Args:
        symbol (str): Trading pair (e.g., 'BTC/USDT').
        side (str): 'buy' or 'sell'.
        amount (float): Order size in base asset.
        price (float, optional): Limit price for limit orders; None for market.

    Returns:
        dict: Order details from CCXT.

    Note: Implements one retry on network errors to balance reliability and latency.
    Dynamic import of telegram_utils for notifications avoids circular dependencies.
    """
    try:
        client = get_binance_client()
        order_type = 'market' if price is None else 'limit'
        params = {} if price is None else {'price': price}
        order = client.create_order(symbol, order_type, side, amount, params=params)
        logging.info(f"Trade executed: {symbol}, {side}, amount={amount}, type={order_type}, price={price or 'market'}")
        
        # Notify via Telegram
        try:
            from utils.telegram_utils import send_notification  # Dynamic import
            asyncio.run(send_notification(
                f"Trade executed: {symbol}, {side}, amount={amount:.6f}, price={price or 'market'}"
            ))
        except Exception as e:
            logging.error(f"Trade notification failed: {e}")

        return order
    except ccxt.NetworkError as e:
        logging.warning(f"Network error on trade {symbol}: {e}. Retrying once...")
        try:
            # Retry once
            client = get_binance_client()
            order = client.create_order(symbol, order_type, side, amount, params=params)
            logging.info(f"Trade retry successful: {symbol}, {side}, amount={amount}")
            from utils.telegram_utils import send_notification
            asyncio.run(send_notification(
                f"Trade retry executed: {symbol}, {side}, amount={amount:.6f}, price={price or 'market'}"
            ))
            return order
        except Exception as retry_e:
            logging.error(f"Trade retry failed: {retry_e}")
            raise
    except Exception as e:
        logging.error(f"Trade execution failed for {symbol}: {e}")
        raise
