import logging
import json
from logging.handlers import RotatingFileHandler
import redis
from dune_client.client import DuneClient
from dune_client.query import QueryBase, QueryParameter
from config import (
    DUNE_API_KEY,
    DUNE_QUERY_ID,
    DUNE_QUERY_IDS,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
)

handler = RotatingFileHandler('bot.log', maxBytes=1_000_000, backupCount=5)
logging.basicConfig(level=logging.INFO, handlers=[handler],
                    format='%(asctime)s - %(message)s')

# Redis connection for caching on-chain metrics
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

# Metrics available from the Dune query
METRIC_KEYS = [
    "sth_rpl",
    "volume",
    "whale_transfers",
    "gas_fee",
    "price_diff",
    "mempool_density",
    "gas_history",
]

SUPPORTED_SYMBOLS = {"BTC", "ETH", "SOL"}


def fetch_dune_metrics(symbol: str = "BTC") -> dict:
    """Fetch on-chain metrics from Dune with Redis caching for the given symbol."""
    symbol = symbol.upper()
    if symbol not in SUPPORTED_SYMBOLS:
        logging.warning("Unsupported symbol %s", symbol)
        return {k: 0 for k in METRIC_KEYS}

    metrics: dict[str, object] = {}
    missing: list[str] = []
    for key in METRIC_KEYS:
        r_key = f"{symbol}:{key}"
        try:
            cached = redis_client.get(r_key)
            if cached is not None:
                if key == "gas_history":
                    metrics[key] = json.loads(cached)
                elif key in ("whale_transfers", "gas_fee"):
                    metrics[key] = int(cached)
                else:
                    metrics[key] = float(cached)
            else:
                missing.append(key)
        except Exception as e:  # pragma: no cover - Redis failures rare in tests
            logging.error(f"Redis fetch failed for {r_key}: {e}")
            missing.append(key)

    if missing:
        try:
            client = DuneClient(DUNE_API_KEY)
            query_id = int(DUNE_QUERY_IDS.get(symbol, DUNE_QUERY_ID))
            query = QueryBase(query_id, params=[QueryParameter.text_type("symbol", symbol)])
            data = client.run_query(query)
            rows = data.get_rows()
        except Exception as e:
            logging.error(f"Dune API fetch failed: {e}")
            rows = []

        if rows:
            row = rows[0]
            for key in missing:
                value = row.get(key)
                r_key = f"{symbol}:{key}"
                if key == "gas_history":
                    value = value or []
                    try:
                        redis_client.setex(r_key, 600, json.dumps(value))
                    except Exception as e:  # pragma: no cover
                        logging.error(f"Redis cache failed for {r_key}: {e}")
                    metrics[key] = value
                else:
                    value = value or 0
                    if key in ("whale_transfers", "gas_fee"):
                        value = int(value)
                    else:
                        value = float(value)
                    try:
                        redis_client.setex(r_key, 600, value)
                    except Exception as e:  # pragma: no cover
                        logging.error(f"Redis cache failed for {r_key}: {e}")
                    metrics[key] = value
        else:
            logging.warning("No data returned from Dune")
            for key in missing:
                metrics.setdefault(key, [] if key == "gas_history" else 0)

    return metrics

def fetch_sth_rpl(symbol: str = "BTC") -> float:
    """Return the latest STH RPL value using :func:`fetch_dune_metrics`."""
    metrics = fetch_dune_metrics(symbol)
    return float(metrics.get("sth_rpl", 0.0))


def get_dune_data() -> dict:
    """Return cached Dune metrics for BTC used in volatility checks."""
    return fetch_dune_metrics("BTC")


def get_oi_funding(pair: str) -> tuple[dict, float]:
    """Simplified open interest and funding rate fetch from Dune."""
    base = pair.split("/")[0]
    metrics = fetch_dune_metrics(base)
    oi_change = metrics.get("volume", 0)
    funding = metrics.get("gas_fee", 0)
    return {"change": oi_change}, float(funding)
