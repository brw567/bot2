import sys
import types
import asyncio
import importlib
import os


def test_e2e_cycle():
    # stub modules
    originals = {k: sys.modules.get(k) for k in ['utils.binance_utils','utils.ml_utils','utils.onchain_utils','redis','ta','dotenv','utils.telegram_utils','ccxt']}
    try:
        os.environ.setdefault('BINANCE_API_KEY', 'x')
        os.environ.setdefault('BINANCE_API_SECRET', 'y')
        os.environ.setdefault('GROK_API_KEY', 'z')
        os.environ.setdefault('TELEGRAM_TOKEN', 't')
        os.environ.setdefault('TELEGRAM_API_ID', '1')
        os.environ.setdefault('TELEGRAM_API_HASH', 'h')
        binance_stub = types.ModuleType('utils.binance_utils')
        class DummyClient:
            def fetch_ohlcv(self, pair, tf, limit=100, since=None):
                return [[i,1,1,1,1+i,1] for i in range(limit)]
        binance_stub.get_binance_client = lambda: DummyClient()
        sys.modules['utils.binance_utils'] = binance_stub

        ml_stub = types.ModuleType('utils.ml_utils')
        ml_stub.lstm_predict = lambda df: {'confidence': 0.8}
        ml_stub.fetch_historical_data = lambda *a, **k: None
        ml_stub.train_model = lambda *a, **k: (None,0.0,0.0)
        ml_stub.predict_next_price = lambda *a, **k: 0.0
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

        tele_stub = types.ModuleType('utils.telegram_utils')
        notifications = []
        alerts = []
        async def send_notification(msg):
            notifications.append(msg)
        async def send_alert(msg):
            alerts.append(msg)
        tele_stub.send_notification = send_notification
        tele_stub.send_alert = send_alert
        sys.modules['utils.telegram_utils'] = tele_stub

        ccxt_stub = types.ModuleType('ccxt')
        class DummyBinance:
            def fetch_ohlcv(self, pair, timeframe='1d', since=None, limit=10):
                return [[i,1,1,1,1+i,1] for i in range(limit)]
        ccxt_stub.binance = lambda *a, **k: DummyBinance()
        sys.modules['ccxt'] = ccxt_stub

        ae = importlib.import_module('core.analytics_engine')
        importlib.reload(ae)
        bt = importlib.import_module('backtest')
        importlib.reload(bt)

        engine = ae.AnalyticsEngine(['BTC/USDT'])
        engine.metrics['BTC/USDT'] = {'strategy': 'grid'}
        asyncio.run(engine.analyze_once())
        res = bt.switching_backtest(['BTC/USDT'])
        assert res['winrate'] > 60
        assert res['sharpe'] > 1.5
        assert res['max_dd'] > -0.05
        assert notifications or alerts
    finally:
        for k, v in originals.items():
            if v is not None:
                sys.modules[k] = v
            else:
                sys.modules.pop(k, None)
