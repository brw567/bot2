import logging
from utils.grok_utils import get_risk_assessment
from config import DB_PATH
import sqlite3

logging.basicConfig(level=logging.INFO, filename='bot.log', filemode='a', format='%(asctime)s - %(message)s')

class BaseStrategy:
    """
    Base class for trading strategies, providing shared risk management logic.
    Includes position sizing (Kelly approximation) and dynamic SL/TP via Grok.

    Note: Used by arbitrage, grid, and MEV strategies to ensure consistent risk handling.
    """
    def __init__(self, capital=10000, risk_per_trade=0.01):
        """
        Initialize base strategy with trading parameters.

        Args:
            capital (float): Initial trading capital (default 10000 USDT).
            risk_per_trade (float): Risk per trade as fraction of capital (default 1%).

        Note: Parameters can be overridden via DB (config.DEFAULT_PARAMS).
        """
        self.capital = capital
        self.risk_per_trade = risk_per_trade

    def calculate_position_size(self, price, sl_distance, winrate=0.6):
        """
        Calculate position size using Kelly criterion approximation.

        Args:
            price (float): Current asset price.
            sl_distance (float): Stop-loss distance as fraction of price.
            winrate (float): Expected winrate (default 0.6 per Trader requirements).

        Returns:
            float: Position size (in units of asset).

        Note: Uses simplified R_ratio=1 for scalping; caps at 20% exposure.
        """
        try:
            if sl_distance <= 0 or price <= 0:
                logging.error(f"Invalid inputs for position size: price={price}, sl_distance={sl_distance}")
                return 0.0
            r_ratio = 1  # Simplified for scalping (risk/reward ratio)
            kelly = winrate - (1 - winrate) / r_ratio
            risk_amount = self.capital * self.risk_per_trade * kelly
            size = risk_amount / (price * sl_distance)
            max_size = self.capital / price * 0.2  # Max 20% exposure per Trader
            final_size = min(size, max_size)
            if final_size <= 0:
                logging.warning(f"Calculated position size non-positive: {final_size}")
                return 0.0
            logging.info(f"Position size calculated: {final_size:.6f} for price={price}, sl_distance={sl_distance}")
            return final_size
        except Exception as e:
            logging.error(f"Position size calculation failed: {e}")
            return 0.0

    def get_dynamic_sl_tp(self, symbol, price, vol):
        """
        Get dynamic stop-loss (SL) and take-profit (TP) using Grok risk assessment.

        Args:
            symbol (str): Trading pair (e.g., 'BTC/USDT').
            price (float): Current asset price.
            vol (float): Volatility (e.g., (high-low)/low).

        Returns:
            tuple: (sl, tp) as floats; None if trade not approved.

        Note: Uses structured Grok prompts (from grok_utils) for consistent outputs.
        """
        try:
            risk = get_risk_assessment(symbol, price, vol, 0.65)  # Mock winrate
            if risk.trade != 'yes':
                logging.info(f"Trade not approved by Grok for {symbol}")
                return None, None
            sl = price * (1 - vol * risk.sl_mult)
            tp = price * (1 + vol * risk.tp_mult)
            logging.info(f"SL/TP for {symbol}: SL={sl:.2f}, TP={tp:.2f}")
            return sl, tp
        except Exception as e:
            logging.error(f"SL/TP calculation failed for {symbol}: {e}")
            return None, None