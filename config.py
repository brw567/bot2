import os
from dotenv import load_dotenv
from db_utils import get_param

load_dotenv()

"""
Central configuration file for the Ultimate Crypto Scalping Bot.
Loads environment variables for secure API access and defines default parameters.
All secrets are stored in .env to prevent hardcoding.
"""

# Exchange API credentials (used by ccxt, backtrader for trading)
BINANCE_API_KEY = get_param('BINANCE_API_KEY', os.getenv('BINANCE_API_KEY'))
BINANCE_API_SECRET = get_param('BINANCE_API_SECRET', os.getenv('BINANCE_API_SECRET'))

# Telegram API credentials for notifications and sentiment analysis (telethon)
TELEGRAM_TOKEN = get_param('TELEGRAM_TOKEN', os.getenv('TELEGRAM_TOKEN'))
TELEGRAM_API_ID = int(get_param('TELEGRAM_API_ID', os.getenv('TELEGRAM_API_ID') or 0))
TELEGRAM_API_HASH = get_param('TELEGRAM_API_HASH', os.getenv('TELEGRAM_API_HASH'))
TELEGRAM_SESSION = get_param('TELEGRAM_SESSION', os.getenv('TELEGRAM_SESSION'))
NOTIFICATIONS_ENABLED = (
    get_param('NOTIFICATIONS_ENABLED', os.getenv('NOTIFICATIONS_ENABLED', 'True'))
    .lower() == 'true'
)

# Grok API for AI-driven risk assessment and parameter tuning (requests)
GROK_API_KEY = get_param('GROK_API_KEY', os.getenv('GROK_API_KEY'))
GROK_TIMEOUT = int(get_param('GROK_TIMEOUT', os.getenv("GROK_TIMEOUT", 10)))

# Dune API for on-chain metrics (e.g., STH RPL via dune-client)
DUNE_API_KEY = get_param('DUNE_API_KEY', os.getenv('DUNE_API_KEY'))
DUNE_QUERY_ID = get_param('DUNE_QUERY_ID', os.getenv('DUNE_QUERY_ID'))  # Must be set to a valid Dune query ID

# Redis configuration for pub/sub messaging (winrate, ML updates via redis)
REDIS_HOST = get_param('REDIS_HOST', os.getenv('REDIS_HOST', 'localhost'))
REDIS_PORT = int(get_param('REDIS_PORT', os.getenv('REDIS_PORT', 6379)))
REDIS_DB = int(get_param('REDIS_DB', os.getenv('REDIS_DB', 0)))

# Binance API rate limit (weights per minute, per 2025 docs)
# Used for quota monitoring in scalping_bot.py
BINANCE_WEIGHT_LIMIT = 6000

# Database and storage paths
DB_PATH = 'bot.db'  # SQLite database for parameters and trade logs
HDF5_DIR = 'data/hdf5/'  # Directory for historical data storage (used by h5py)

# Default trading parameters (stored in DB, can be overridden via GUI)
# Note: Chosen to balance risk/reward for scalping; adjustable in settings tab
DEFAULT_PARAMS = {
    'win_rate_threshold': 0.6,  # Minimum winrate to continue trading
    'max_consec_losses': 3,     # Pause after 3 consecutive losses
    'slippage_tolerance': 0.001,  # Max slippage (0.1%) for trade execution
    'risk_per_trade': 0.01,
    'grok_timeout': GROK_TIMEOUT,
    'auto_pair_limit': 10,  # Number of pairs to auto-trade (monitor 5x for analytics)
}

# Analytics settings for ContinuousAnalyzer
ANALYTICS_INTERVAL = int(get_param('ANALYTICS_INTERVAL', os.getenv('ANALYTICS_INTERVAL', 300)))  # seconds
VOL_THRESHOLD = float(get_param('VOL_THRESHOLD', os.getenv('VOL_THRESHOLD', 0.05)))
GARCH_FLAG = (get_param('GARCH_FLAG', os.getenv('GARCH_FLAG', 'False')).lower() == 'true')
