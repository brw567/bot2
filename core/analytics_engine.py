import asyncio
import json
import logging
from logging.handlers import RotatingFileHandler
from typing import Iterable, Dict

import pandas as pd

try:
    from ta import add_all_ta_features
except Exception:  # pragma: no cover - optional dep
    def add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume"):
        return df

from utils.binance_utils import get_binance_client
from utils.ml_utils import lstm_predict
from utils.onchain_utils import get_oi_funding
import redis
import config

handler = RotatingFileHandler('bot.log', maxBytes=1_000_000, backupCount=5)
logging.basicConfig(level=logging.INFO, handlers=[handler],
                    format='%(asctime)s - %(message)s')

STRATEGY_MAP = {
    'trending': 'momentum',
    'sideways': 'grid',
    'volatile': 'arbitrage'
}

class AnalyticsEngine:
    """Asynchronous engine for continuous multi-pair analytics."""

    def __init__(self, pairs: Iterable[str], timeframe: str = '1m'):
        self.pairs = list(pairs)
        self.timeframe = timeframe
        self.metrics: Dict[str, dict] = {}
        try:
            self.redis = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB)
        except Exception:
            self.redis = None

    async def fetch_data(self, pair: str, limit: int = 100) -> pd.DataFrame:
        """Return OHLCV dataframe for the pair with Grok fallback."""
        try:
            client = get_binance_client()
            ohlcv = await asyncio.to_thread(
                client.fetch_ohlcv, pair, self.timeframe, limit=limit
            )
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["source"] = "binance"
        except Exception as e:  # pragma: no cover - network issues
            logging.error(f"Binance fetch failed for {pair}: {e}")
            try:
                from utils.grok_utils import grok_fetch_ohlcv

                ohlcv = await grok_fetch_ohlcv(pair, self.timeframe, limit)
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["source"] = "grok"
            except Exception as ge:
                logging.error(f"Grok fallback failed for {pair}: {ge}")
                raise
        df = add_all_ta_features(
            df, open="open", high="high", low="low", close="close", volume="volume"
        )
        return df

    def compute_metrics(self, df: pd.DataFrame) -> dict:
        df = add_all_ta_features(
            df, open="open", high="high", low="low", close="close", volume="volume"
        )
        rsi = float(df.get("momentum_rsi", pd.Series([50])).iloc[-1])
        macd_diff = float(df.get("trend_macd_diff", pd.Series([0])).iloc[-1])
        atr = float(df.get("volatility_atr", pd.Series([0])).iloc[-1])
        returns = df["close"].pct_change().dropna()
        sharpe = float((returns.mean() / returns.std()) * (252 ** 0.5)) if returns.std() != 0 else 0.0
        return {
            "rsi": rsi,
            "macd_diff": macd_diff,
            "atr": atr,
            "sharpe": sharpe,
            "avg_atr": float(df.get("volatility_atr", pd.Series([atr])).mean()),
        }

    async def analyze_once(self):
        """Analyze all pairs once and update metrics."""
        for pair in self.pairs:
            try:
                df = await self.fetch_data(pair)
                metrics = self.compute_metrics(df)
                oi, funding = get_oi_funding(pair)
                oi_change = oi.get("change", 0) if isinstance(oi, dict) else 0
                metrics["funding_rate"] = funding
                if metrics['macd_diff'] > 0 and metrics['rsi'] > 50:
                    pattern = 'trending'
                elif metrics['atr'] < metrics['avg_atr']:
                    pattern = 'sideways'
                else:
                    pattern = 'volatile'
                strategy = STRATEGY_MAP.get(pattern, 'hold')
                prediction = lstm_predict(df)
                if prediction.get("confidence", 0) > config.CONFIDENCE_THRESHOLD:
                    metrics.update(
                        {
                            "pattern": pattern,
                            "strategy": strategy,
                            "oi_change": oi_change,
                            "source": str(df['source'].iloc[0]) if 'source' in df else 'binance',
                        }
                    )
                else:
                    metrics.update(
                        {
                            "pattern": "hold",
                            "strategy": "hold",
                            "oi_change": oi_change,
                            "source": str(df['source'].iloc[0]) if 'source' in df else 'binance',
                        }
                    )

                prev_strategy = self.metrics.get(pair, {}).get("strategy")
                self.metrics[pair] = metrics
                if self.redis:
                    try:
                        self.redis.set(f"metrics:{pair}", json.dumps(metrics))
                    except Exception as e:  # pragma: no cover - Redis optional
                        logging.error(f"Redis store failed: {e}")
                if prev_strategy and prev_strategy != metrics["strategy"]:
                    try:
                        from utils.telegram_utils import send_notification

                        await send_notification(
                            f"Switch to {metrics['strategy']} on {pair}"
                        )
                    except Exception as ne:  # pragma: no cover - notification errors
                        logging.error(f"Notification failed: {ne}")
            except Exception as e:  # pragma: no cover - network issues
                logging.error(f"Analytics error for {pair}: {e}")
                try:
                    from utils.telegram_utils import send_notification

                    await send_notification(f"Data failure for {pair}: {e}")
                except Exception:
                    pass

    async def continuous_analyze(self, interval: int = 60):
        """Run continuous analysis loop."""
        while True:
            await self.analyze_once()
            await asyncio.sleep(interval)
