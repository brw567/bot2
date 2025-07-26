import asyncio
import json
import logging
from logging.handlers import RotatingFileHandler
from typing import Iterable, Dict, Any

import pandas as pd
from ta import add_all_ta_features

from config import (
    ANALYTICS_TIMEFRAME,
    RSI_THRESHOLD,
    ATR_THRESHOLD,
    OI_THRESHOLD,
    FUNDING_THRESHOLD,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    DB_PATH,
)
from utils.binance_utils import fetch_ohlcv_async
from utils.ml_utils import lstm_predict
from utils.onchain_utils import get_oi_funding
import redis
import sqlite3

handler = RotatingFileHandler('bot.log', maxBytes=1_000_000, backupCount=5)
logging.basicConfig(level=logging.INFO, handlers=[handler],
                    format='%(asctime)s - %(message)s')


class AnalyticsEngine:
    """Asynchronous analytics engine monitoring multiple pairs."""

    def __init__(self, pairs: Iterable[str], timeframe: str = ANALYTICS_TIMEFRAME):
        self.pairs = list(pairs)
        self.timeframe = timeframe
        self.metrics: Dict[str, Dict[str, Any]] = {}
        try:
            self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
            self.redis.ping()
        except Exception:
            self.redis = None
            self.db = sqlite3.connect(DB_PATH)
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS analytics_metrics (pair TEXT PRIMARY KEY, data TEXT)"
            )
            self.db.commit()

    def _store(self, pair: str, data: Dict[str, Any]) -> None:
        self.metrics[pair] = data
        payload = json.dumps(data)
        if self.redis:
            try:
                self.redis.set(f"metrics:{pair}", payload)
                return
            except Exception as e:
                logging.error(f"Redis store failed: {e}")
                self.redis = None
        if hasattr(self, "db"):
            self.db.execute(
                "INSERT OR REPLACE INTO analytics_metrics (pair, data) VALUES (?, ?)",
                (pair, payload),
            )
            self.db.commit()

    async def analyze_pair(self, pair: str) -> None:
        try:
            ohlcv = await fetch_ohlcv_async(pair, self.timeframe, limit=100)
        except Exception:
            from utils.grok_utils import get_backup_price

            price = get_backup_price(pair.split("/")[0])
            if price:
                self._store(pair, {"price": price, "error": "binance"})
            return
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
        rsi = df["momentum_rsi"].iloc[-1]
        macd_diff = df["trend_macd_diff"].iloc[-1]
        atr = df["volatility_atr"].iloc[-1]
        returns = df["close"].pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() else 0.0
        oi, funding = get_oi_funding(pair.split("/")[0])
        prediction = lstm_predict(df)
        pattern = "trending" if macd_diff > 0 and rsi > RSI_THRESHOLD else "sideways" if atr < df["volatility_atr"].mean() else "volatile"
        strategy = {"trending": "momentum", "sideways": "grid", "volatile": "arbitrage"}.get(pattern, "hold")
        if prediction.get("confidence", 0) < 0.7:
            strategy = "hold"
        metrics = {
            "pattern": pattern,
            "strategy": strategy,
            "rsi": float(rsi),
            "atr": float(atr),
            "sharpe": float(sharpe),
            "oi_change": float(oi.get("change", 0)),
            "funding": float(funding),
        }
        self._store(pair, metrics)

    async def analyze_once(self) -> None:
        for p in self.pairs:
            await self.analyze_pair(p)

    async def continuous_analyze(self, interval: int = 60) -> None:
        while True:
            await self.analyze_once()
            await asyncio.sleep(interval)
