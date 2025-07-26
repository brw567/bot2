import sys
import types
from types import SimpleNamespace
import os
import pandas as pd

# Stub external libraries used by backtest module
ccxt_stub = types.ModuleType('ccxt')
class DummyBinance:
    def fetch_ohlcv(self, pair, timeframe='1d', limit=10):
        return [[i,1,1,1,1+i,1] for i in range(limit)]
ccxt_stub.binance = lambda *a, **k: DummyBinance()
sys.modules['ccxt'] = ccxt_stub

talib_stub = types.ModuleType('talib')
talib_stub.EMA = lambda arr, timeperiod=12: pd.Series(arr)
talib_stub.RSI = lambda arr, timeperiod=14: pd.Series([50]*len(arr))
sys.modules['talib'] = talib_stub

vectorbt_stub = types.ModuleType('vectorbt')
sys.modules['vectorbt'] = vectorbt_stub
bt_stub = types.ModuleType('backtrader')
class DummyStrategy: pass
bt_stub.Strategy = DummyStrategy
bt_stub.indicators = SimpleNamespace(EMA=lambda *a, **k: None,
                                     RSI=lambda *a, **k: None)
bt_stub.Order = SimpleNamespace(Market=0)
bt_stub.feeds = SimpleNamespace(PandasData=object)
bt_stub.Cerebro = object
bt_stub.sizers = SimpleNamespace(FixedSize=object)
bt_stub.analyzers = SimpleNamespace(SharpeRatio=object,
                                   DrawDown=object,
                                   TradeAnalyzer=object)
sys.modules['backtrader'] = bt_stub

sklearn_stub = types.ModuleType('sklearn')
cluster_stub = types.ModuleType('sklearn.cluster')
class DummyKM:
    def __init__(self, *a, **k):
        pass
    def fit(self, data):
        return self
    def predict(self, data):
        return [0]
cluster_stub.KMeans = DummyKM
sklearn_stub.cluster = cluster_stub
sys.modules['sklearn'] = sklearn_stub
sys.modules['sklearn.cluster'] = cluster_stub

arch_stub = types.ModuleType('arch')
arch_stub.arch_model = lambda *a, **k: None
sys.modules['arch'] = arch_stub

# Stub ML utils
ml_utils_stub = types.ModuleType('utils.ml_utils')
ml_utils_stub.fetch_historical_data = lambda *a, **k: None
ml_utils_stub.train_model = lambda *a, **k: (None, 0.0, 0.0)
ml_utils_stub.predict_next_price = lambda *a, **k: 0.0
sys.modules['utils.ml_utils'] = ml_utils_stub

# dummy torch to satisfy any indirect imports
sys.modules['torch'] = types.ModuleType('torch')

# Environment variables for config
os.environ.setdefault('BINANCE_API_KEY', 'x')
os.environ.setdefault('BINANCE_API_SECRET', 'y')
os.environ.setdefault('GROK_API_KEY', 'x')
os.environ.setdefault('TELEGRAM_TOKEN', 'x')
import importlib
import config as cfg
importlib.reload(cfg)
import backtest
importlib.reload(backtest)


def test_multi_backtest_returns_metrics():
    metrics = backtest.multi_backtest(['BTC/USDT', 'ETH/USDT'], limit=5)
    keys = {'sharpe', 'var', 'cvar', 'max_dd', 'winrate'}
    assert keys.issubset(metrics)


def test_switching_backtest():
    class DummyAE:
        def __init__(self, pairs, timeframe='1m'):
            self.pairs = pairs
            self.metrics = {}
        async def analyze_once(self):
            self.metrics = {
                self.pairs[0]: {'pattern': 'trending'},
                self.pairs[1]: {'pattern': 'sideways'},
            }
    ae_mod = types.ModuleType('core.analytics_engine')
    ae_mod.AnalyticsEngine = DummyAE
    orig = sys.modules.get('core.analytics_engine')
    sys.modules['core.analytics_engine'] = ae_mod
    try:
        res = backtest.switching_backtest(['BTC/USDT','ETH/USDT'])
    finally:
        if orig is not None:
            sys.modules['core.analytics_engine'] = orig
    assert {'sharpe', 'winrate', 'max_dd'} <= set(res)
