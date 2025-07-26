import sys
import types
import os
import pandas as pd

# Stub external modules
ccxt_stub = types.ModuleType('ccxt')
class DummyBinance:
    def fetch_ohlcv(self, pair, timeframe='5m', limit=120):
        return [[0,1,1,1,1,1] for _ in range(limit)]
ccxt_stub.binance = lambda *a, **kw: DummyBinance()
sys.modules['ccxt'] = ccxt_stub

talib_stub = types.ModuleType('talib')
talib_stub.RSI = lambda arr, timeperiod=14: pd.Series([50]*len(arr))

def macd(arr, fastperiod=12, slowperiod=26, signalperiod=9):
    s = pd.Series([0]*len(arr))
    return s, s, s

talib_stub.MACD = macd
talib_stub.EMA = lambda arr, timeperiod=20: pd.Series(arr)
talib_stub.BBANDS = lambda arr, timeperiod=20: (pd.Series(arr), pd.Series(arr), pd.Series(arr))
talib_stub.STOCH = lambda h,l,c: (pd.Series(c), pd.Series(c))
sys.modules['talib'] = talib_stub

arch_stub = types.ModuleType('arch')
class DummyRes:
    def __init__(self):
        self.conditional_volatility = pd.Series([0.1,0.1,0.1])
class DummyModel:
    def fit(self, disp='off'):
        return DummyRes()
arch_stub.arch_model = lambda *a, **kw: DummyModel()
sys.modules['arch'] = arch_stub

sympy_stub = types.ModuleType('sympy')
sympy_stub.symbols = lambda x: (x+'a', x+'b') if x == 'a b' else None
sympy_stub.Eq = lambda l,r: (l,r)
sympy_stub.solve = lambda eqns, vars: {vars[0]:1, vars[1]:0}
sys.modules['sympy'] = sympy_stub
dotenv_stub = types.ModuleType('dotenv')
dotenv_stub.load_dotenv = lambda *a, **k: None
sys.modules['dotenv'] = dotenv_stub

os.environ.setdefault('TELEGRAM_API_ID', '1')
os.environ.setdefault('TELEGRAM_API_HASH', 'x')
os.environ.setdefault('TELEGRAM_SESSION', 'x')
os.environ.setdefault('BINANCE_API_KEY', 'x')
os.environ.setdefault('BINANCE_API_SECRET', 'x')

from analyzer import ContinuousAnalyzer


def test_compute_success():
    ca = ContinuousAnalyzer(None, [])
    score = ca.compute_success(sharpe=1.0, win_rate=0.6, slippage=0.01, vol=0.1, pattern="breakout")
    assert score > 0


def test_compute_metrics():
    ca = ContinuousAnalyzer(None, [])
    df = pd.DataFrame({'timestamp':[0,1,2], 'open':[1,1,1], 'high':[1,1,1], 'low':[1,1,1], 'close':[1,2,3], 'volume':[1,1,1]})
    metrics = ca.compute_metrics(df)
    assert 'sharpe' in metrics and 'rsi' in metrics


def test_compute_vol():
    ca = ContinuousAnalyzer(None, [])
    vol = ca.compute_vol([10, 10, 10])
    assert vol == 0.0

