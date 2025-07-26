import sys
import types
import importlib
import asyncio
import time

# Stub dependencies
requests_stub = types.ModuleType('requests')
class DummyResp:
    def __init__(self, txt):
        self.text = txt
    def raise_for_status(self):
        pass
    def json(self):
        return {'choices': [{'message': {'content': self.text}}]}
requests_stub.post = lambda *a, **k: DummyResp('{}')
sys.modules['requests'] = requests_stub

streamlit_stub = types.ModuleType('streamlit')
streamlit_stub.session_state = {}
sys.modules['streamlit'] = streamlit_stub

pydantic_stub = types.ModuleType('pydantic')
class BaseModel:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
class ValidationError(Exception):
    pass
pydantic_stub.BaseModel = BaseModel
pydantic_stub.ValidationError = ValidationError
sys.modules['pydantic'] = pydantic_stub

tele_stub = types.ModuleType('utils.telegram_utils')
async def fetch_channel_messages(channel, limit=100):
    return ['msg']
tele_stub.fetch_channel_messages = fetch_channel_messages
sys.modules['utils.telegram_utils'] = tele_stub

config_stub = types.ModuleType('config')
config_stub.GROK_API_KEY = 'x'
config_stub.GROK_TIMEOUT = 10
config_stub.GROK_PAIRS_INTERVAL = 4
config_stub.GROK_SENTIMENT_INTERVAL = 4
config_stub.VOL_THRESHOLD = 0.5
old_config = sys.modules.get('config')
sys.modules['config'] = config_stub

import utils.grok_utils as gu
importlib.reload(gu)

if old_config is not None:
    sys.modules['config'] = old_config
else:
    del sys.modules['config']


def test_pair_recs_cache():
    calls = {'n': 0}
    def dummy(count):
        calls['n'] += 1
        return [f'P{calls["n"]}']
    gu.get_recommended_pairs = dummy
    r1 = gu.get_grok_pair_recs(1)
    r2 = gu.get_grok_pair_recs(1)
    assert r1 == r2
    assert calls['n'] == 1
    gu._last_pairs_call -= 5
    r3 = gu.get_grok_pair_recs(1)
    assert calls['n'] == 2
    assert r3 != []


def test_sentiment_cache():
    calls = {'n': 0}
    async def dummy(symbol):
        calls['n'] += 1
        return gu.SentimentResponse(sentiment='pos', score=calls['n'], details='d')
    gu.get_sentiment_analysis = dummy
    r1 = asyncio.run(gu.get_grok_insights('BTC'))
    r2 = asyncio.run(gu.get_grok_insights('BTC'))
    assert calls['n'] == 1
    assert r1.score == r2.score
    gu._last_sentiment_call -= 3
    asyncio.run(gu.get_grok_insights('BTC', vol=1))
    assert calls['n'] == 2
