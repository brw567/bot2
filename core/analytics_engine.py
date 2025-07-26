import asyncio
import json
import logging
from logging.handlers import RotatingFileHandler
from typing import Iterable, Dict
import os

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
        required = [
            'BINANCE_API_KEY',
            'BINANCE_API_SECRET',
            'GROK_API_KEY',
            'TELEGRAM_TOKEN',
            'TELEGRAM_API_ID',
            'TELEGRAM_API_HASH',
        ]
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            raise ValueError(f"Missing env key: {', '.join(missing)}")

    async def fetch_data(self, pair: str, limit: int = 100) -> pd.DataFrame:
        """Return OHLCV dataframe for the pair with Grok fallback."""
        try:
            client = get_binance_client()
            ohlcv = await asyncio.to_thread(client.fetch_ohlcv, pair, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['source'] = 'binance'
            df = add_all_ta_features(df, open='open', high='high', low='low', close='close', volume='volume')
        except Exception as e:
            logging.error(f"Binance fetch failed for {pair}: {e}")
            try:
                from utils.telegram_utils import send_alert
                await send_alert(f"Binance fetch failed for {pair}: {e}")
            except Exception as err:
                logging.error(f"Alert failed: {err}")
            from utils.grok_utils import grok_fetch_ohlcv  # local import to avoid overhead
            ohlcv = await grok_fetch_ohlcv(pair, self.timeframe, limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['source'] = 'grok'
            df = add_all_ta_features(df, open='open', high='high', low='low', close='close', volume='volume')
        return df

    def compute_metrics(self, df: pd.DataFrame) -> dict:
        try:
            rsi = float(df['momentum_rsi'].iloc[-1]) if 'momentum_rsi' in df else 50.0
            macd_diff = float(df['trend_macd_diff'].iloc[-1]) if 'trend_macd_diff' in df else 0.0
            atr = float(df['volatility_atr'].iloc[-1]) if 'volatility_atr' in df else 0.0
            sharpe = 0.0
            returns = df['close'].pct_change().dropna()
            if returns.std() != 0:
                sharpe = float((returns.mean() / returns.std()) * (252 ** 0.5))
            return {
                'rsi': rsi,
                'macd_diff': macd_diff,
                'atr': atr,
                'sharpe': sharpe,
                'avg_atr': float(df['volatility_atr'].mean()) if 'volatility_atr' in df else atr
            }
        except Exception as e:  # pragma: no cover - unexpected df issues
            logging.error(f"Metric calculation failed: {e}")
            return {'rsi': 50.0, 'macd_diff': 0.0, 'atr': 0.0, 'sharpe': 0.0, 'avg_atr': 0.0}

    async def analyze_once(self):
        """Analyze all pairs once and update metrics."""
        for pair in self.pairs:
            try:
                df = await self.fetch_data(pair)
                metrics = self.compute_metrics(df)
                oi, funding = get_oi_funding(pair)
                oi_change = oi.get('change', 0) if isinstance(oi, dict) else 0
                if metrics['macd_diff'] > 0 and metrics['rsi'] > 50:
                    pattern = 'trending'
                elif metrics['atr'] < metrics['avg_atr']:
                    pattern = 'sideways'
                else:
                    pattern = 'volatile'
                strategy = STRATEGY_MAP.get(pattern, 'hold')
                prediction = lstm_predict(df)
                if prediction.get('confidence', 0) > 0.7:
                    metrics.update({'pattern': pattern, 'strategy': strategy, 'oi_change': oi_change})
                else:
                    metrics.update({'pattern': 'hold', 'strategy': 'hold', 'oi_change': oi_change})
                metrics['funding_rate'] = funding
                metrics['data_source'] = df['source'].iloc[-1] if 'source' in df else 'binance'
                prev = self.metrics.get(pair, {})
                self.metrics[pair] = metrics
                if prev.get('strategy') and prev.get('strategy') != metrics['strategy']:
                    try:
                        from utils.telegram_utils import send_notification
                        await send_notification(f"Switch to {metrics['strategy']} on {pair}")
                    except Exception as e:
                        logging.error(f"Notification failed: {e}")
                if self.redis:
                    try:
                        self.redis.set(f"metrics:{pair}", json.dumps(metrics))
                    except Exception as e:  # pragma: no cover - Redis optional
                        logging.error(f"Redis store failed: {e}")
            except Exception as e:  # pragma: no cover - network issues
                logging.error(f"Analytics error for {pair}: {e}")
                try:
                    from utils.telegram_utils import send_alert
                    await send_alert(f"Analysis failed for {pair}: {e}")
                except Exception as err:
                    logging.error(f"Alert failed: {err}")

    async def continuous_analyze(self, interval: int = 60):
        """Run continuous analysis loop."""
        while True:
            try:
                await self.analyze_once()
            except Exception as e:  # pragma: no cover - top level errors
                logging.error(f"continuous_analyze failed: {e}")
                try:
                    from utils.telegram_utils import send_alert
                    await send_alert(f"Continuous analyze failure: {e}")
                except Exception as err:
                    logging.error(f"Alert failed: {err}")
            await asyncio.sleep(interval)
