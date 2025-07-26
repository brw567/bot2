import asyncio
import json
import logging
from logging.handlers import RotatingFileHandler
from typing import Iterable, Dict
import os
import time

import pandas as pd

try:
    from ta import add_all_ta_features
except Exception:  # pragma: no cover - optional dep
    def add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume"):
        return df

from utils.binance_utils import get_binance_client
from utils.ml_utils import lstm_predict
from utils.onchain_utils import get_oi_funding, get_dune_data
from utils.config import load_config_from_db
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
        self.cfg = load_config_from_db()
        self.max_active = int(self.cfg['max_active_pairs'])
        self.swap_multiplier = int(self.cfg['swap_multiplier'])
        self.grok_interval = int(self.cfg['grok_interval'])
        self.dune_interval = int(self.cfg['dune_interval'])
        self.analytics_interval = int(self.cfg['analytics_interval'])
        self.swap_threshold = float(self.cfg['swap_threshold'])
        self.cooldown_period = int(self.cfg['cooldown'])
        self.forecast_period = int(self.cfg['forecast_period'])
        self.history_period = int(self.cfg['history_period'])

        self.pairs = list(pairs)
        needed = self.max_active * self.swap_multiplier
        if len(self.pairs) < needed:
            from utils.grok_utils import get_grok_pairs
            extra = get_grok_pairs(needed)
            for p in extra:
                if p not in self.pairs:
                    self.pairs.append(p)
                if len(self.pairs) >= needed:
                    break
        self.timeframe = timeframe
        self.metrics: Dict[str, dict] = {}
        self.cooldowns: dict[str, float] = {}
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

    def smart_rating(self, metrics: dict, forecast: int) -> float:
        """Return a simplified profitability rating."""
        sharpe = metrics.get('sharpe', 0)
        rsi_comp = (metrics.get('rsi', 50) - 50) / 50
        oi = metrics.get('oi_change', 0) / 100
        return 0.4 * sharpe + 0.3 * rsi_comp + 0.3 * oi

    def detect_volatility(self, dune: dict) -> bool:
        """Determine if the market is volatile based on ATR and Dune data."""
        atr_flag = any(
            m.get('atr', 0) > m.get('avg_atr', 0) * 2 for m in self.metrics.values()
        )
        oi_flag = dune.get('volume', 0) > 10
        return atr_flag or oi_flag

    def handle_swapping(self) -> None:
        """Swap active pairs if a swap pair scores much higher."""
        active = self.pairs[: self.max_active]
        swaps = self.pairs[self.max_active :]
        if not swaps or not active:
            return
        ratings = {p: self.metrics.get(p, {}).get('rating', 0) for p in self.pairs}
        low = min(active, key=lambda p: ratings.get(p, 0))
        high = max(swaps, key=lambda p: ratings.get(p, 0))
        if ratings.get(high, 0) > ratings.get(low, 0) * self.swap_threshold:
            now = time.time()
            if now >= self.cooldowns.get(low, 0):
                li = self.pairs.index(low)
                hi = self.pairs.index(high)
                self.pairs[li], self.pairs[hi] = self.pairs[hi], self.pairs[li]
                self.cooldowns[low] = now + self.cooldown_period

    async def analyze_once(self):
        """Analyze all pairs once and update metrics."""
        dune_data = get_dune_data()
        self.volatile = self.detect_volatility(dune_data)
        if self.redis:
            try:
                self.redis.set("market_volatile", json.dumps({"volatile": self.volatile}))
            except Exception:
                pass
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
                metrics['oi_change'] = oi_change
                metrics['rating'] = self.smart_rating(metrics, self.forecast_period)
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
        self.handle_swapping()

    async def continuous_analyze(self, interval: int = 60):
        """Run continuous analysis loop with dynamic intervals and pair updates."""
        next_update = 0.0
        while True:
            now = time.time()
            if now >= next_update:
                from utils.grok_utils import get_grok_pairs
                self.pairs = get_grok_pairs(self.max_active * self.swap_multiplier)
                next_update = now + (self.grok_interval / 2 if self.volatile else self.grok_interval)
            try:
                await self.analyze_once()
            except Exception as e:  # pragma: no cover - top level errors
                logging.error(f"continuous_analyze failed: {e}")
                try:
                    from utils.telegram_utils import send_alert
                    await send_alert(f"Continuous analyze failure: {e}")
                except Exception as err:
                    logging.error(f"Alert failed: {err}")
            interval = 30 if self.volatile else self.analytics_interval
            await asyncio.sleep(interval)
