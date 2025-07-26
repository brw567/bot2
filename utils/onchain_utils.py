import logging
from logging.handlers import RotatingFileHandler
from dune_client.client import DuneClient
from config import DUNE_API_KEY, DUNE_QUERY_ID

handler = RotatingFileHandler('bot.log', maxBytes=1_000_000, backupCount=5)
logging.basicConfig(level=logging.INFO, handlers=[handler],
                    format='%(asctime)s - %(message)s')

def fetch_sth_rpl(symbol='BTC'):
    """
    Fetch Short-Term Holder Realized Price (STH RPL) for Bitcoin via Dune API.

    Args:
        symbol (str): Asset symbol (default 'BTC'; only BTC supported currently).

    Returns:
        float: Latest STH RPL value; 0.0 on failure.

    Note: Uses dune-client for simplified API calls, addressing immediate task.
    Requires valid DUNE_QUERY_ID from config.py (set to Dune query for STH RPL).
    Fallback to 0.0 ensures signal scoring robustness on API failure.
    """
    try:
        if symbol != 'BTC':
            logging.warning(f"STH RPL only supported for BTC, not {symbol}")
            return 0.0

        client = DuneClient(DUNE_API_KEY)
        # Execute query and fetch latest result
        client.execute_query(DUNE_QUERY_ID)
        data = client.get_query_results(DUNE_QUERY_ID)
        
        # Assume data structure: extract latest STH RPL
        if data and 'rows' in data and data['rows']:
            rpl = float(data['rows'][0].get('realized_price_sth', 0.0))
            logging.info(f"Fetched STH RPL for {symbol}: {rpl:.2f}")
            return rpl
        else:
            logging.warning(f"No STH RPL data returned for {symbol}")
            return 0.0
    except Exception as e:
        logging.error(f"Dune API fetch failed for {symbol}: {e}")
        return 0.0
