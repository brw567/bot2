import sys, types, os
dotenv_stub = types.ModuleType('dotenv')
dotenv_stub.load_dotenv = lambda *a, **k: None
sys.modules['dotenv'] = dotenv_stub

# stub dune_client
class DummyResults:
    def get_rows(self):
        return [{
            'sth_rpl': 1.0,
            'volume': 2.0,
            'whale_transfers': 3,
            'gas_fee': 4,
            'price_diff': 0.002,
            'mempool_density': 0.5,
            'gas_history': [1, 1, 1]
        }]

class DummyClient:
    def __init__(self, *a, **k):
        pass
    def run_query(self, query):
        return DummyResults()

dune_client_stub = types.ModuleType('dune_client')
client_stub = types.ModuleType('dune_client.client')
query_stub = types.ModuleType('dune_client.query')
client_stub.DuneClient = DummyClient
class QueryBase:
    def __init__(self, query_id, params=None):
        self.query_id = query_id
        self.params = params
class QueryParameter:
    @staticmethod
    def text_type(name, value):
        return (name, value)
query_stub.QueryBase = QueryBase
query_stub.QueryParameter = QueryParameter
dune_client_stub.client = client_stub
dune_client_stub.query = query_stub
sys.modules['dune_client'] = dune_client_stub
sys.modules['dune_client.client'] = client_stub
sys.modules['dune_client.query'] = query_stub

redis_stub = types.ModuleType('redis')
class DummyRedis:
    def __init__(self, *a, **k):
        self.store = {}
    def get(self, k):
        return self.store.get(k)
    def setex(self, k, t, v):
        self.store[k] = v
redis_stub.Redis = DummyRedis
sys.modules['redis'] = redis_stub

os.environ['DUNE_API_KEY'] = 'x'
os.environ['DUNE_QUERY_ID'] = '1'
os.environ.setdefault('TELEGRAM_API_ID', '1')
os.environ.setdefault('TELEGRAM_API_HASH', 'x')
os.environ.setdefault('TELEGRAM_SESSION', 'x')
os.environ.setdefault('BINANCE_API_KEY', 'x')
os.environ.setdefault('BINANCE_API_SECRET', 'x')

import importlib
import config
import utils.onchain_utils as ocu
importlib.reload(config)
importlib.reload(ocu)
from utils.onchain_utils import fetch_dune_metrics, fetch_sth_rpl


def test_fetch_dune_metrics():
    metrics = fetch_dune_metrics('BTC')
    assert metrics['volume'] == 2.0
    assert fetch_sth_rpl('BTC') == 1.0
