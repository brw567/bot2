import logging
from logging.handlers import RotatingFileHandler
import asyncio
import time
from strategies.base_strategy import BaseStrategy
from utils.binance_utils import get_binance_client
from utils.telegram_utils import send_notification
from config import DB_PATH
import sqlite3
import pandas as pd

handler = RotatingFileHandler('bot.log', maxBytes=1_000_000, backupCount=5)
logging.basicConfig(level=logging.INFO, handlers=[handler],
                    format='%(asctime)s - %(message)s')

class MEVStrategy(BaseStrategy):
    """
    MEV (Miner Extractable Value) detection strategy for identifying potential sandwich attacks.
    Monitors order book for rapid buy-sell patterns as a proxy for MEV on CEX (e.g., Binance).

    Inherits BaseStrategy for risk management (position sizing, SL/TP).
    Note: Uses two-snapshot order book analysis to detect rapid changes indicative of sandwich
    attacks, as CEX lacks mempool access; thresholds (20% imbalance, 15% delta) are heuristic.
    """
    def __init__(self, imbalance_threshold=0.2, delta_threshold=0.15, **kwargs):
        """
        Initialize MEV strategy with detection parameters.

        Args:
            imbalance_threshold (float): Min order book imbalance to flag MEV (default 20%).
            delta_threshold (float): Min change in depth between snapshots (default 15%).
            **kwargs: Passed to BaseStrategy (e.g., capital, risk_per_trade).
        """
        super().__init__(**kwargs)
        self.imbalance_threshold = imbalance_threshold
        self.delta_threshold = delta_threshold

    def detect_mev(self, symbol):
        """
        Detect potential MEV (e.g., sandwich attacks) by analyzing order book snapshots.

        Args:
            symbol (str): Trading pair (e.g., 'BTC/USDT').

        Returns:
            bool: True if MEV detected (trades should pause), False otherwise.

        Note: Takes two order book snapshots 1s apart to detect rapid buy-sell patterns.
        Logs to DB and sends Telegram notification if MEV is flagged.
        """
        try:
            client = get_binance_client()
            # First snapshot
            book = client.fetch_order_book(symbol, limit=10)
            bid_depth = sum([b[1] for b in book['bids']])
            ask_depth = sum([a[1] for a in book['asks']])
            imbalance = abs(bid_depth - ask_depth) / (bid_depth + ask_depth)

            # Second snapshot after 1s
            time.sleep(1)
            book2 = client.fetch_order_book(symbol, limit=10)
            bid_depth2 = sum([b[1] for b in book2['bids']])
            delta = abs(bid_depth2 - bid_depth) / bid_depth if bid_depth > 0 else 0

            # Check for MEV (rapid buy-sell pattern)
            if imbalance > self.imbalance_threshold or delta > self.delta_threshold:
                logging.info(f"Potential MEV detected for {symbol}: imbalance={imbalance:.2f}, delta={delta:.2f}")
                asyncio.run(send_notification(f"Potential sandwich/MEV detected for {symbol}: imbalance={imbalance:.2f}, delta={delta:.2f}"))
                # Log to DB for audit
                conn = sqlite3.connect(DB_PATH)
                conn.execute("INSERT INTO trades (symbol, profit, timestamp) VALUES (?, ?, ?)",
                             (symbol, 0.0, pd.Timestamp.now()))  # Mock profit for MEV event
                conn.commit()
                conn.close()
                return True
            return False
        except Exception as e:
            logging.error(f"MEV detection failed for {symbol}: {e}")
            return False
