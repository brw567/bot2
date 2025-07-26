import sys
import types
import os

# Stub ccxt with fetch_balance
ccxt_stub = types.ModuleType('ccxt')
class DummyBinance:
    def fetch_balance(self):
        return {'total': {'BTC': 1, 'ETH': 20, 'USDT': 100}}
ccxt_stub.binance = lambda *a, **kw: DummyBinance()
sys.modules['ccxt'] = ccxt_stub

# Minimal dotenv stub
dotenv_stub = types.ModuleType('dotenv')
dotenv_stub.load_dotenv = lambda *a, **k: None
sys.modules['dotenv'] = dotenv_stub

os.environ.setdefault('TELEGRAM_API_ID', '1')
os.environ.setdefault('TELEGRAM_API_HASH', 'x')
os.environ.setdefault('TELEGRAM_SESSION', 'x')
os.environ.setdefault('BINANCE_API_KEY', 'x')
os.environ.setdefault('BINANCE_API_SECRET', 'x')

import importlib
import utils.binance_utils as bu
importlib.reload(bu)
from analyzer import pairs
get_balance = bu.get_balance


def test_get_balance_adds_pairs():
    pairs.clear()
    get_balance()
    assert 'ETH/USDT' in pairs
    assert 'BTC/USDT' not in pairs
