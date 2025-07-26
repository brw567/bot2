import sys
import types
import asyncio
import pandas as pd

# stub external modules
old_bu = sys.modules.get('utils.binance_utils')
bu_stub = types.ModuleType('utils.binance_utils')
class DummyClient:
    def fetch_ohlcv(self, pair, timeframe='1m', limit=100):
        return [[0,1,1,1,1,1] for _ in range(limit)]
async def fetch(symbol, timeframe='1m', limit=100):
    return DummyClient().fetch_ohlcv(symbol, timeframe, limit)
bu_stub.fetch_ohlcv_async = fetch
sys.modules['utils.binance_utils'] = bu_stub

old_ml = sys.modules.get('utils.ml_utils')
ml_stub = types.ModuleType('utils.ml_utils')
ml_stub.lstm_predict = lambda df: {'confidence': 0.8}
sys.modules['utils.ml_utils'] = ml_stub

old_ocu = sys.modules.get('utils.onchain_utils')
ocu_stub = types.ModuleType('utils.onchain_utils')
ocu_stub.get_oi_funding = lambda pair: ({'change':0.2}, 0.02)
sys.modules['utils.onchain_utils'] = ocu_stub

old_ta = sys.modules.get('ta')
ta_stub = types.ModuleType('ta')
def add_all_ta_features(df, **k):
    df['momentum_rsi'] = 60
    df['trend_macd_diff'] = 1
    df['volatility_atr'] = 0.1
    return df

ta_stub.add_all_ta_features = add_all_ta_features
sys.modules['ta'] = ta_stub

old_redis = sys.modules.get('redis')
redis_stub = types.ModuleType('redis')
class DummyRedis:
    def __init__(self, *a, **k):
        self.store = {}
    def ping(self):
        pass
    def set(self, k, v):
        self.store[k] = v
redis_stub.Redis = DummyRedis
sys.modules['redis'] = redis_stub

config_stub = types.ModuleType('config')
config_stub.ANALYTICS_TIMEFRAME = '1m'
config_stub.RSI_THRESHOLD = 50
config_stub.ATR_THRESHOLD = 0
config_stub.OI_THRESHOLD = 0.1
config_stub.FUNDING_THRESHOLD = 0.01
config_stub.REDIS_HOST = 'localhost'
config_stub.REDIS_PORT = 6379
config_stub.REDIS_DB = 0
config_stub.DB_PATH = ':memory:'
old_config = sys.modules.get('config')
sys.modules['config'] = config_stub

import importlib
from core.analytics_engine import AnalyticsEngine
importlib.reload(sys.modules['core.analytics_engine'])
if old_config is not None:
    sys.modules['config'] = old_config
else:
    del sys.modules['config']
if old_bu is not None:
    sys.modules['utils.binance_utils'] = old_bu
else:
    del sys.modules['utils.binance_utils']
if old_ml is not None:
    sys.modules['utils.ml_utils'] = old_ml
else:
    del sys.modules['utils.ml_utils']
if old_ocu is not None:
    sys.modules['utils.onchain_utils'] = old_ocu
else:
    del sys.modules['utils.onchain_utils']
if old_ta is not None:
    sys.modules['ta'] = old_ta
else:
    del sys.modules['ta']
if old_redis is not None:
    sys.modules['redis'] = old_redis
else:
    del sys.modules['redis']


def test_analyze_once():
    engine = AnalyticsEngine(['BTC/USDT'])
    asyncio.run(engine.analyze_once())
    assert engine.metrics['BTC/USDT']['strategy'] == 'momentum'
