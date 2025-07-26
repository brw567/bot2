import sys
import types
import importlib
import asyncio


def test_analyze_once():
    originals = {k: sys.modules.get(k) for k in ['utils.binance_utils','utils.ml_utils','utils.onchain_utils','redis','ta','dotenv']}
    try:
        binance_stub = types.ModuleType('utils.binance_utils')
        class DummyClient:
            def fetch_ohlcv(self, pair, tf, limit=100):
                return [[i,1,1,1,1+i,1] for i in range(limit)]
        binance_stub.get_binance_client = lambda: DummyClient()
        sys.modules['utils.binance_utils'] = binance_stub

        ml_stub = types.ModuleType('utils.ml_utils')
        ml_stub.lstm_predict = lambda df: {'confidence': 0.8}
        sys.modules['utils.ml_utils'] = ml_stub

        onchain_stub = types.ModuleType('utils.onchain_utils')
        onchain_stub.get_oi_funding = lambda pair: ({'change': 0}, 0)
        sys.modules['utils.onchain_utils'] = onchain_stub

        redis_stub = types.ModuleType('redis')
        class DummyRedis:
            def __init__(self, *a, **k):
                pass
            def set(self, *a, **k):
                pass
        redis_stub.Redis = DummyRedis
        sys.modules['redis'] = redis_stub

        ta_stub = types.ModuleType('ta')
        ta_stub.add_all_ta_features = lambda df, **k: df.assign(momentum_rsi=55, trend_macd_diff=1, volatility_atr=0.5)
        sys.modules['ta'] = ta_stub

        dotenv_stub = types.ModuleType('dotenv')
        dotenv_stub.load_dotenv = lambda *a, **k: None
        sys.modules['dotenv'] = dotenv_stub

        ae = importlib.import_module('core.analytics_engine')
        importlib.reload(ae)
        engine = ae.AnalyticsEngine(['BTC/USDT'])
        asyncio.run(engine.analyze_once())
        assert 'BTC/USDT' in engine.metrics
        assert engine.metrics['BTC/USDT']['strategy'] == 'momentum'
    finally:
        for k, v in originals.items():
            if v is not None:
                sys.modules[k] = v
            else:
                sys.modules.pop(k, None)
