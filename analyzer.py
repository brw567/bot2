import asyncio
import json
import logging
from logging.handlers import RotatingFileHandler
import time
from typing import Iterable

import pandas as pd
import numpy as np

try:
    import ccxt
except Exception:  # pragma: no cover - handled in tests with stub
    ccxt = None

try:
    import talib
except Exception:  # pragma: no cover - handled in tests with stub
    talib = None

try:
    from arch import arch_model
except Exception:  # pragma: no cover - handled in tests with stub
    arch_model = None

from config import (
    ANALYTICS_INTERVAL,
    VOL_THRESHOLD,
    GARCH_FLAG,
    VOLATILITY_THRESHOLD_PERCENT,
)
from utils.binance_utils import execute_trade, get_balance

handler = RotatingFileHandler('bot.log', maxBytes=1_000_000, backupCount=5)
logging.basicConfig(level=logging.INFO, handlers=[handler],
                    format='%(asctime)s - %(message)s')

# Global set of monitored pairs that other modules can extend
pairs: set[str] = set()


class ContinuousAnalyzer:
    """Continuously analyze trading pairs and publish metrics via Redis."""

    def __init__(self, redis_conn, pairs_iter: Iterable[str]):
        self.redis = redis_conn
        self.pairs = set(pairs_iter)
        pairs.update(self.pairs)
        self.prev_metrics: dict[str, dict] = {}
        try:
            get_balance()
        except Exception as e:
            logging.error(f"Balance check failed: {e}")

    async def fetch_data(self, pair: str) -> pd.DataFrame:
        """Fetch recent OHLCV data for a pair using ccxt."""
        if ccxt is None:
            raise RuntimeError("ccxt is required for fetch_data")
        client = ccxt.binance({"enableRateLimit": True})
        ohlcv = await asyncio.to_thread(client.fetch_ohlcv, pair, "5m", limit=120)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        return df

    def compute_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate trading metrics from OHLCV dataframe."""
        returns = np.log(df["close"]).diff().dropna()
        sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() else 0.0
        var = float(returns.quantile(0.05))
        cvar = float(returns[returns <= var].mean())
        rsi_val = float(talib.RSI(df["close"], timeperiod=14).iloc[-1]) if talib else 0.0
        macd, macd_signal, _ = talib.MACD(df["close"], 12, 26, 9) if talib else (pd.Series([0]), pd.Series([0]), None)
        macd_val = float(macd.iloc[-1] - macd_signal.iloc[-1]) if macd.size else 0.0
        if GARCH_FLAG and arch_model is not None:
            try:
                am = arch_model(returns.dropna() * 100, vol="GARCH", p=1, q=1)
                res = am.fit(disp="off")
                vol = float(res.conditional_volatility.iloc[-1]) / 100
            except Exception:
                vol = float(returns.std())
        else:
            vol = float(returns.std())
        return {
            "sharpe": sharpe,
            "var": var,
            "cvar": cvar,
            "rsi": rsi_val,
            "macd": macd_val,
            "volatility": vol,
        }

    def detect_pattern(self, df: pd.DataFrame) -> dict:
        """Detect technical patterns using TA-Lib and trend fitting via sympy."""
        ema = float(talib.EMA(df["close"], timeperiod=20).iloc[-1]) if talib else df["close"].iloc[-1]
        rsi_val = float(talib.RSI(df["close"], timeperiod=14).iloc[-1]) if talib else 50.0
        if talib:
            upper, middle, lower = talib.BBANDS(df["close"], timeperiod=20)
            bb_pos = float((df["close"].iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] or 1))
            fastk, fastd = talib.STOCH(df["high"], df["low"], df["close"])
            stoch = float(fastk.iloc[-1] - fastd.iloc[-1])
            macd, macd_signal, _ = talib.MACD(df["close"])
            macd_cross = float(macd.iloc[-1] - macd_signal.iloc[-1])
        else:
            bb_pos = stoch = macd_cross = 0.0
        try:
            import sympy as sp  # type: ignore
            a, b = sp.symbols("a b")
            x1, x2 = 0, len(df) - 1
            y1, y2 = df["close"].iloc[0], df["close"].iloc[-1]
            sol = sp.solve((sp.Eq(a * x1 + b, y1), sp.Eq(a * x2 + b, y2)), (a, b))
            trend = float(sol[a])
        except Exception:
            trend = float(np.polyfit(range(len(df)), df["close"], 1)[0])
        return {
            "ema": ema,
            "rsi_indicator": rsi_val,
            "bb_pos": bb_pos,
            "stoch": stoch,
            "macd_cross": macd_cross,
            "trend": trend,
        }

    def compute_vol(self, gas_history: list[float]) -> float:
        """Return standard deviation of gas history."""
        try:
            arr = np.array(gas_history, dtype=float)
            return float(arr.std()) if arr.size else 0.0
        except Exception:
            return 0.0

    def is_market_volatile(self) -> bool:
        """Return True if current metrics or cached data indicate high volatility."""
        try:
            for data in self.prev_metrics.values():
                if data.get("volatility", 0) > VOL_THRESHOLD:
                    return True
            if self.redis:
                cached = self.redis.get("BTC:gas_fee")
                if cached and float(cached) > 50:
                    return True
        except Exception as e:
            logging.error(f"Volatility check failed: {e}")
        return False

    def compute_success(self, sharpe: float, win_rate: float, slippage: float, vol: float, pattern: str | None = None) -> float:
        """Return weighted success score based on performance metrics and pattern."""
        score = 0.5 * sharpe + 0.4 * win_rate - 0.05 * slippage - 0.05 * vol
        if pattern == "breakout":
            score += 0.1
        elif pattern == "reversal":
            score += 0.05
        return score

    def should_switch(self, metrics: dict) -> bool:
        """Decide if strategy should switch based on Sharpe delta and VaR."""
        pair = metrics.get("pair")
        prev = self.prev_metrics.get(pair)
        self.prev_metrics[pair] = metrics
        if not prev:
            return False
        delta = metrics["sharpe"] - prev.get("sharpe", 0)
        return delta < -0.5 or metrics["var"] < -0.05

    async def monitor_exit(self, pair: str):
        """Monitor open trades and exit on RSI>70 or timeout."""
        start = time.time()
        while True:
            try:
                df = await self.fetch_data(pair)
                rsi_val = float(talib.RSI(df["close"], timeperiod=14).iloc[-1]) if talib else 0.0
                if rsi_val > 70 or time.time() - start > 3600:
                    execute_trade(pair, "sell", 1)
                    return
            except Exception as e:  # pragma: no cover
                logging.error(f"monitor_exit error for {pair}: {e}")
                return
            await asyncio.sleep(30)

    async def run(self):
        """Main loop fetching data and publishing metrics."""
        while True:
            try:
                get_balance()
            except Exception as e:
                logging.error(f"Balance check failed: {e}")
            for pair in list(self.pairs):
                try:
                    df = await self.fetch_data(pair)
                    metrics = self.compute_metrics(df)
                    pattern = self.detect_pattern(df)
                    metrics.update(pattern)
                    metrics["pair"] = pair
                    if self.redis:
                        self.redis.set(f"metrics:{pair}", json.dumps(metrics))
                        self.redis.publish("analytics", json.dumps({pair: metrics}))
                    if self.should_switch(metrics):
                        logging.info(f"Strategy switch suggested for {pair}")
                        asyncio.create_task(self.monitor_exit(pair))
                except Exception as e:  # pragma: no cover
                    logging.error(f"Analyzer error for {pair}: {e}")
            volatile = self.is_market_volatile()
            if self.redis:
                msg = json.dumps({"volatile": volatile})
                self.redis.set("market_volatile", msg)
                self.redis.publish("market_volatile", msg)
            sleep_int = 30 if volatile else ANALYTICS_INTERVAL
            await asyncio.sleep(sleep_int)

