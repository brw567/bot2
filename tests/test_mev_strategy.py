import sys, types, os
sys.modules['streamlit'] = types.ModuleType('streamlit')
redis_stub = types.ModuleType('redis')
redis_stub.Redis = lambda *a, **k: None
sys.modules['redis'] = redis_stub
telethon_stub = types.ModuleType('telethon')
telethon_sessions_stub = types.ModuleType('telethon.sessions')
telethon_stub.TelegramClient = object
telethon_sessions_stub.StringSession = type('SS', (), {})
sys.modules['telethon'] = telethon_stub
sys.modules['telethon.sessions'] = telethon_sessions_stub
dotenv_stub = types.ModuleType('dotenv')
dotenv_stub.load_dotenv = lambda *a, **k: None
sys.modules['dotenv'] = dotenv_stub
telegram_stub = types.ModuleType('utils.telegram_utils')
telegram_stub.send_notification = lambda *a, **k: None
sys.modules['utils.telegram_utils'] = telegram_stub
os.environ.setdefault('TELEGRAM_API_ID', '1')
os.environ.setdefault('TELEGRAM_API_HASH', 'x')
os.environ.setdefault('TELEGRAM_SESSION', 'x')
os.environ.setdefault('BINANCE_API_KEY', 'x')
os.environ['BINANCE_API_SECRET'] = 'x'
os.environ.setdefault('GROK_API_KEY', 'x')
os.environ.setdefault('TELEGRAM_TOKEN', 'x')

from strategies.mev_strategy import MEVStrategy


def test_arbitrage_and_sandwich():
    strat = MEVStrategy()
    assert strat.is_arbitrage_opportunity(0.002, 'positive')
    assert not strat.is_arbitrage_opportunity(0.0, 'negative')
    assert strat.is_sandwich_risk(0.9)
    assert not strat.is_sandwich_risk(0.1)
