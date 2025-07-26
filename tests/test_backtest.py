import sys
import types
from types import SimpleNamespace
import os
import pandas as pd

# Stub external libraries used by backtest module
ccxt_stub = types.ModuleType('ccxt')
class DummyBinance:
    def fetch_ohlcv(self, pair, timeframe='1d', since=None, limit=10):
        return [[i,1,1,1,1+i,1] for i in range(limit)]
ccxt_stub.binance = lambda *a, **k: DummyBinance()
sys.modules['ccxt'] = ccxt_stub

import utils.binance_utils as bu
bu.get_binance_client = lambda: DummyBinance()

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

ta_stub = types.ModuleType('ta')
ta_stub.add_all_ta_features = lambda df, **k: df.assign(momentum_rsi=55, trend_macd_diff=1, volatility_atr=0.5)
sys.modules['ta'] = ta_stub

requests_stub = types.ModuleType('requests')
class DummyReqEx(Exception):
    pass
requests_stub.RequestException = DummyReqEx
sys.modules['requests'] = requests_stub

tele_stub = types.ModuleType('utils.telegram_utils')
async def send_notification(msg):
    pass
async def send_alert(msg):
    pass
tele_stub.send_notification = send_notification
tele_stub.send_alert = send_alert
sys.modules['utils.telegram_utils'] = tele_stub

# onchain utils
onchain_stub = types.ModuleType('utils.onchain_utils')
onchain_stub.get_oi_funding = lambda pair: ({'change': 0}, 0)
onchain_stub.get_dune_data = lambda: {'volume': 0}
sys.modules['utils.onchain_utils'] = onchain_stub

# Stub ML utils
ml_utils_stub = types.ModuleType('utils.ml_utils')
ml_utils_stub.fetch_historical_data = lambda *a, **k: None
ml_utils_stub.train_model = lambda *a, **k: (None, 0.0, 0.0)
ml_utils_stub.predict_next_price = lambda *a, **k: 0.0
ml_utils_stub.lstm_predict = lambda df: {'confidence': 0.8}
sys.modules['utils.ml_utils'] = ml_utils_stub

# dummy torch to satisfy any indirect imports
sys.modules['torch'] = types.ModuleType('torch')

# Environment variables for config
os.environ.setdefault('BINANCE_API_KEY', 'x')
os.environ.setdefault('BINANCE_API_SECRET', 'y')
os.environ.setdefault('GROK_API_KEY', 'z')
os.environ.setdefault('TELEGRAM_TOKEN', 't')
os.environ.setdefault('TELEGRAM_API_ID', '1')
os.environ.setdefault('TELEGRAM_API_HASH', 'h')

import importlib
import config
importlib.reload(config)
import core.analytics_engine as ae
importlib.reload(ae)
import backtest
importlib.reload(backtest)
def test_multi_backtest_returns_metrics():
    metrics = backtest.multi_backtest(['BTC/USDT', 'ETH/USDT'], limit=5)
    keys = {'sharpe', 'var', 'cvar', 'max_dd', 'winrate'}
    assert keys.issubset(metrics)


def test_switching_backtest():
    async def _fake_fetch(self, pair, limit=100):
        return pd.DataFrame(
            [[i,1,1,1,1+i,1] for i in range(limit)],
            columns=['timestamp','open','high','low','close','volume']
        ).assign(momentum_rsi=55, trend_macd_diff=1, volatility_atr=0.5)

    async def _fake_analyze_once(self):
        for p in self.pairs:
            self.metrics[p] = {'pattern': 'trending'}

    ae.AnalyticsEngine.fetch_data = _fake_fetch
    ae.AnalyticsEngine.analyze_once = _fake_analyze_once
    sys.modules['core.analytics_engine'] = ae
    ml = types.ModuleType('utils.ml_utils')
    ml.fetch_historical_data = lambda *a, **k: None
    ml.train_model = lambda *a, **k: (None,0.0,0.0)
    ml.predict_next_price = lambda *a, **k: 0.0
    ml.lstm_predict = lambda df: {'confidence':0.8}
    sys.modules['utils.ml_utils'] = ml
    importlib.reload(backtest)
    backtest.ccxt = ccxt_stub
    backtest.ccxt.binance = lambda *a, **k: DummyBinance()
    res = backtest.switching_backtest(['BTC/USDT'])
    assert {'sharpe', 'winrate', 'max_dd'} <= set(res)
    assert res['winrate'] > 60
    assert res['sharpe'] > 1.5
    assert res['max_dd'] > -0.05
