import logging
import sqlite3
import pandas as pd
from config import DB_PATH
from strategies.base_strategy import BaseStrategy
from utils.ccxt_utils import calculate_spread  # No circular: dynamic imports in ccxt_utils
from utils.grok_utils import get_risk_assessment
from utils.binance_utils import get_binance_client

logging.basicConfig(level=logging.INFO, filename='bot.log', filemode='a', format='%(asctime)s - %(message)s')

class ArbitrageStrategy(BaseStrategy):
    """
    Arbitrage strategy for spot-futures price differences.
    Executes buy/sell if spread exceeds threshold after fees/slippage.

    Inherits BaseStrategy for risk management (position sizing, SL/TP).
    Note: Fixed threshold (0.002) balances profitability; configurable if needed.
    """
    def __init__(self, threshold=0.002, **kwargs):
        """
        Initialize arbitrage strategy.

        Args:
            threshold (float): Minimum spread to trigger trade (default 0.2%).
            **kwargs: Passed to BaseStrategy (e.g., capital, risk_per_trade).
        """
        super().__init__(**kwargs)
        self.threshold = threshold

    def run(self, symbol_spot, symbol_futures):
        """
        Execute arbitrage if spread exceeds threshold and risk is approved.

        Args:
            symbol_spot (str): Spot pair (e.g., 'BTC/USDT').
            symbol_futures (str): Futures pair (e.g., 'BTC/USDT:USDT').

        Note: Uses dynamic import of binance_utils to avoid circular dependencies.
        """
        try:
            spread = calculate_spread(symbol_spot, symbol_futures)
            client = get_binance_client()  # Dynamic import from binance_utils
            ticker = client.fetch_ticker(symbol_spot)
            price = ticker['last']
            vol = (ticker['high'] - ticker['low']) / ticker['low']  # Simple volatility

            # Risk check via Grok
            risk = get_risk_assessment(symbol_spot, price, vol, 0.65)  # Mock winrate
            if risk.trade != 'yes':
                logging.info(f"Arbitrage skipped for {symbol_spot}: Risk not approved")
                return

            # Position sizing
            sl, tp = self.get_dynamic_sl_tp(symbol_spot, price, vol)
            if not sl or not tp:
                logging.info(f"Arbitrage skipped for {symbol_spot}: Invalid SL/TP")
                return
            size = self.calculate_position_size(price, (price - sl) / price, vol=vol)

            # Execute if spread exceeds threshold
            if spread > self.threshold:
                from utils.binance_utils import execute_trade  # Dynamic import
                execute_trade(symbol_spot, 'buy', size)
                execute_trade(symbol_futures, 'sell', size)
                logging.info(f"Arbitrage executed: {symbol_spot}/{symbol_futures}, spread={spread:.4f}, size={size:.6f}")
                # Log trade to DB (mock profit for simplicity)
                conn = sqlite3.connect(DB_PATH)
                conn.execute("INSERT INTO trades (symbol, profit, timestamp) VALUES (?, ?, ?)",
                             (symbol_spot, spread * size * price, pd.Timestamp.now()))
                conn.commit()
                conn.close()
        except Exception as e:
            logging.error(f"Arbitrage run failed for {symbol_spot}: {e}")
