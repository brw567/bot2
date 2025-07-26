import asyncio
import json
import logging
from logging.handlers import RotatingFileHandler
from typing import Iterable, Dict
import time
import os

import pandas as pd

try:
    from ta import add_all_ta_features
except Exception:  # pragma: no cover - optional dep
    def add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume"):
        return df

from utils.binance_utils import get_binance_client
from utils.ml_utils import lstm_predict
from utils.onchain_utils import get_oi_funding, get_dune_data
from utils.grok_utils import get_grok_pairs
from utils.config import load_config_from_db
from db_utils import store_var
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
        cfg = load_config_from_db()
        self.max_active = int(cfg.get('max_active_pairs', cfg.get('auto_pair_limit', 5)))
        self.swap_multiplier = int(cfg.get('swap_pair_multiplier', 10))
        self.grok_interval = int(cfg.get('grok_interval', 4 * 60 * 60))
        self.dune_interval = int(cfg.get('dune_interval', 600))
        self.analytics_interval = int(cfg.get('analytics_interval', 60))
        self.swap_threshold = float(cfg.get('swap_threshold', 1.5))
        self.cooldown = int(cfg.get('cooldown', 45 * 60))
        self.forecast_period = int(cfg.get('forecast_period', 4 * 60 * 60))
        self.history_period = int(cfg.get('history_period', 24 * 60 * 60))
        self.timeframe = timeframe
        self.metrics: Dict[str, dict] = {}
        self.active_pairs = list(pairs)[: self.max_active]
        self.swap_pairs = list(pairs)[self.max_active : self.max_active * self.swap_multiplier]
        self.pairs = self.active_pairs + self.swap_pairs
        self.cooldowns: Dict[str, float] = {}
        self.volatile = False
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
                metrics['rating'] = self.smart_rating(metrics, forecast=self.forecast_period)
                prev = self.metrics.get(pair, {})
                self.metrics[pair] = metrics
                self.volatile = self.volatile or self.detect_volatility(metrics, oi_change)
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

    def smart_rating(self, metrics: dict, forecast: int) -> float:
        """Return a simple combined rating for a pair."""
        return (
            0.4 * metrics.get('sharpe', 0) +
            0.3 * (metrics.get('rsi', 50) / 100) +
            0.3 * (metrics.get('oi_change', 0) / 100)
        )

    def detect_volatility(self, metrics: dict, oi_change: float) -> bool:
        return metrics.get('atr', 0) > metrics.get('avg_atr', 1) * 2 or oi_change > 10

    def handle_swapping(self):
        if not self.swap_pairs:
            return
        ratings = {p: self.metrics.get(p, {}).get('rating', 0) for p in self.pairs}
        if not ratings:
            return
        low = min(self.active_pairs, key=lambda p: ratings.get(p, 0)) if self.active_pairs else None
        high = max(self.swap_pairs, key=lambda p: ratings.get(p, 0)) if self.swap_pairs else None
        if not low or not high:
            return
        if ratings.get(high, 0) > ratings.get(low, 0) * self.swap_threshold:
            if self.cooldowns.get(low, 0) <= time.time():
                self.cooldowns[low] = time.time() + self.cooldown
                self.active_pairs.remove(low)
                self.swap_pairs.remove(high)
                self.swap_pairs.append(low)
                self.active_pairs.append(high)
                self.pairs = self.active_pairs + self.swap_pairs
                store_var('last_swap', f'{low}->{high}')

    async def update_pairs(self):
        while True:
            all_pairs = get_grok_pairs(self.max_active * self.swap_multiplier)
            self.active_pairs = all_pairs[: self.max_active]
            self.swap_pairs = all_pairs[self.max_active:]
            self.pairs = self.active_pairs + self.swap_pairs
            await asyncio.sleep(self.grok_interval / 2 if self.volatile else self.grok_interval)

    async def dune_cache(self):
        while True:
            data = get_dune_data()
            if isinstance(data, dict):
                self.volatile = self.volatile or data.get('oi_change', 0) > 10
            if self.redis:
                try:
                    self.redis.set('market_volatile', json.dumps({'volatile': self.volatile}))
                except Exception:
                    pass
            await asyncio.sleep(self.dune_interval)

    async def continuous_analyze(self, interval: int | None = None):
        """Run continuous analysis loop with dynamic intervals."""
        if not getattr(self, "_tasks_started", False):
            self._tasks_started = True
            asyncio.create_task(self.update_pairs())
            asyncio.create_task(self.dune_cache())
        interval = interval or self.analytics_interval
        while True:
            try:
                self.volatile = False
                await self.analyze_once()
                self.handle_swapping()
            except Exception as e:  # pragma: no cover - top level errors
                logging.error(f"continuous_analyze failed: {e}")
                try:
                    from utils.telegram_utils import send_alert
                    await send_alert(f"Continuous analyze failure: {e}")
                except Exception as err:
                    logging.error(f"Alert failed: {err}")
            await asyncio.sleep(30 if self.volatile else interval)

