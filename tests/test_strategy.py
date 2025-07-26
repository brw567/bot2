import sys
import types
import os

sys.modules['streamlit'] = types.ModuleType('streamlit')
pydantic_stub = types.ModuleType('pydantic')
class BaseModel: pass
class ValidationError(Exception): pass
pydantic_stub.BaseModel = BaseModel
pydantic_stub.ValidationError = ValidationError
sys.modules['pydantic'] = pydantic_stub
sys.modules.setdefault('requests', types.ModuleType('requests'))
redis_stub = types.ModuleType('redis')
class DummyRedis:
    def __init__(self, *a, **kw):
        pass
redis_stub.Redis = DummyRedis
sys.modules.setdefault('redis', redis_stub)
telethon_stub = types.ModuleType('telethon')
telethon_sessions_stub = types.ModuleType('telethon.sessions')
telethon_stub.TelegramClient = object
class StringSession: pass
telethon_sessions_stub.StringSession = StringSession
sys.modules['telethon'] = telethon_stub
sys.modules['telethon.sessions'] = telethon_sessions_stub
dotenv_stub = types.ModuleType('dotenv')
dotenv_stub.load_dotenv = lambda *a, **k: None
sys.modules['dotenv'] = dotenv_stub
os.environ.setdefault('TELEGRAM_API_ID', '1')
os.environ.setdefault('TELEGRAM_API_HASH', 'x')
os.environ.setdefault('TELEGRAM_SESSION', 'x')
os.environ['BINANCE_API_KEY'] = 'x'
os.environ['BINANCE_API_SECRET'] = 'x'
os.environ.setdefault('GROK_API_KEY', 'x')
os.environ.setdefault('TELEGRAM_TOKEN', 'x')
from strategies.base_strategy import BaseStrategy

def test_volatility_scaling():
    strat = BaseStrategy(capital=10000, risk_per_trade=0.01)
    size_low = strat.calculate_position_size(price=100, sl_distance=0.01, vol=0.01)
    size_high = strat.calculate_position_size(price=100, sl_distance=0.01, vol=0.05)
    assert size_high < size_low


def test_position_size_cap():
    strat = BaseStrategy(capital=10000, risk_per_trade=1.0)
    size = strat.calculate_position_size(price=100, sl_distance=0.01, vol=0.0)
    assert size == 20.0
