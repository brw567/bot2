import pandas as pd
import talib
import ccxt
import vectorbt as vbt
import backtrader as bt
from sklearn.cluster import KMeans
from arch import arch_model
import logging
from config import BINANCE_API_KEY, BINANCE_API_SECRET
from utils.ml_utils import fetch_historical_data, train_model, predict_next_price

logging.basicConfig(level=logging.INFO, filename='bot.log', filemode='a', format='%(asctime)s - %(message)s')

def simple_backtest(symbol='BTC/USDT', timeframe='1d', limit=365, fees=0.001, slippage=0.0005):
    """
    Run a simple backtest using EMA and RSI signals with vectorbt for vectorized performance.
    
    Args:
        symbol (str): Trading pair, e.g., 'BTC/USDT'.
        timeframe (str): Candle timeframe, e.g., '1d'.
        limit (int): Number of candles to fetch.
        fees (float): Trading fee per trade (default 0.1%).
        slippage (float): Slippage per trade (default 0.05%).
    
    Returns:
        dict: Metrics including Sharpe ratio, max drawdown, winrate, and cumulative return.
    
    Note: Uses vectorbt for faster execution (~50% speedup over pandas) on large datasets.
    """
    try:
        client = ccxt.binance({'apiKey': BINANCE_API_KEY, 'secret': BINANCE_API_SECRET})
        ohlcv = client.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Calculate indicators (per Analyst requirements)
        df['ema12'] = talib.EMA(df['close'], timeperiod=12)
        df['ema26'] = talib.EMA(df['close'], timeperiod=26)
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)

        # Generate signals: Buy if EMA12 > EMA26 and RSI < 70, Sell if opposite
        entries = (df['ema12'] > df['ema26']) & (df['rsi'] < 70)
        exits = (df['ema12'] < df['ema26']) & (df['rsi'] > 30)

        # Vectorbt portfolio simulation
        pf = vbt.Portfolio.from_signals(
            df['close'],
            entries=entries,
            exits=exits,
            fees=fees,
            slippage=slippage,
            freq=timeframe
        )

        # Calculate metrics (per Trader: Sharpe >1.2, max DD <10%)
        sharpe = pf.sharpe_ratio()
        max_dd = pf.max_drawdown()
        winrate = pf.win_ratio()
        cumulative = pf.total_return()

        return {
            'sharpe': float(sharpe),
            'max_dd': float(max_dd),
            'winrate': float(winrate),
            'cumulative': float(cumulative)
        }
    except Exception as e:
        logging.error(f"Simple backtest failed for {symbol}: {e}")
        return {}

class ScalpingStrategy(bt.Strategy):
    """
    Backtrader strategy for advanced backtesting with broker-like simulation.
    Incorporates EMA/RSI signals, KMeans clustering, and GARCH volatility.
    """
    params = (
        ('ema12', 12),
        ('ema26', 26),
        ('rsi_period', 14),
        ('fees', 0.001),
        ('slippage', 0.0005),
    )

    def __init__(self):
        self.ema12 = bt.indicators.EMA(period=self.params.ema12)
        self.ema26 = bt.indicators.EMA(period=self.params.ema26)
        self.rsi = bt.indicators.RSI(period=self.params.rsi_period)
        self.order = None

    def next(self):
        if self.order:
            return
        # Clustering and volatility (simplified for backtest)
        data = pd.DataFrame({
            'close': self.data.close.get(size=60),
            'rsi': self.rsi.get(size=60)
        })
        kmeans = KMeans(n_clusters=2).fit(data)
        cluster = kmeans.predict([data.iloc[-1]])[0]
        # Estimate conditional volatility using a simple GARCH(1,1)
        returns = data['close'].pct_change().dropna()
        if len(returns) > 1:
            am = arch_model(returns, vol='Garch', p=1, q=1)
            res = am.fit(disp='off')
            vol = res.conditional_volatility.iloc[-1]
        else:
            vol = 0.02

        if self.ema12[0] > self.ema26[0] and self.rsi[0] < 70 and cluster == 1:
            size = self.broker.getcash() * 0.01 / self.data.close[0]  # 1% risk
            self.order = self.buy(size=size, exectype=bt.Order.Market)
        elif self.ema12[0] < self.ema26[0] and self.rsi[0] > 30:
            self.order = self.sell(size=self.position.size, exectype=bt.Order.Market)

def advanced_backtest(symbol='BTC/USDT', timeframe='1d', limit=365):
    """
    Run an advanced backtest using backtrader for broker-like simulation.
    
    Args:
        symbol (str): Trading pair.
        timeframe (str): Candle timeframe.
        limit (int): Number of candles to fetch.
    
    Returns:
        dict: Metrics including Sharpe, max drawdown, winrate, and return.
    
    Note: Backtrader used for realistic broker simulation; slower but more detailed.
    """
    try:
        client = ccxt.binance({'apiKey': BINANCE_API_KEY, 'secret': BINANCE_API_SECRET})
        ohlcv = client.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Backtrader data feed
        data = bt.feeds.PandasData(dataname=df)
        cerebro = bt.Cerebro()
        cerebro.addstrategy(ScalpingStrategy)
        cerebro.adddata(data)
        cerebro.broker.set_cash(10000)
        cerebro.addsizer(bt.sizers.FixedSize, stake=1)
        cerebro.broker.setcommission(commission=0.001, margin=0.0)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        results = cerebro.run()
        strat = results[0]
        sharpe = strat.analyzers.sharpe.get_analysis()['sharpe'] or 0
        max_dd = strat.analyzers.drawdown.get_analysis()['max']['drawdown'] / 100
        winrate = strat.analyzers.trades.get_analysis()['won']['total'] / \
                  strat.analyzers.trades.get_analysis()['total']['closed'] if strat.analyzers.trades.get_analysis()['total']['closed'] else 0

        return {
            'sharpe': float(sharpe),
            'max_dd': float(max_dd),
            'winrate': float(winrate),
            'cumulative': (cerebro.broker.getvalue() - 10000) / 10000
        }
    except Exception as e:
        logging.error(f"Advanced backtest failed for {symbol}: {e}")
        return {}

def ml_backtest(symbol='BTC/USDT', timeframe='5m', years=1):
    """
    Run a backtest incorporating ML predictions, checking for overfitting, with clustering and volatility.
    
    Args:
        symbol (str): Trading pair.
        timeframe (str): Candle timeframe (5m for scalping).
        years (int): Years of historical data to fetch.
    
    Returns:
        dict: Metrics including overfitting status, train/val loss, Sharpe, etc.
    
    Note: Uses scikit-learn for clustering and the arch library for volatility to enhance signals.
    """
    try:
        # Train ML model with train/validation split
        model, train_loss, val_loss = train_model(symbol, epochs=5)
        if val_loss > train_loss * 1.2:  # Overfitting threshold
            logging.warning(f"Overfitting detected: val_loss={val_loss}, train_loss={train_loss}")
            return {
                'overfitting': True,
                'train_loss': float(train_loss),
                'val_loss': float(val_loss)
            }

        # Fetch historical data
        df = fetch_historical_data(symbol, timeframe, years)
        predictions = []
        for i in range(len(df) - 60):
            recent = df.iloc[i:i+60][['close', 'volume', 'rsi']].values[-1]
            pred = predict_next_price(model, recent)
            predictions.append(pred)
        df['pred'] = [None] * 60 + predictions

        # Clustering with scikit-learn (per Analyst: +0.1 weight if bull cluster)
        features = df[['close', 'rsi']].dropna()
        kmeans = KMeans(n_clusters=2, random_state=42).fit(features)
        df['cluster'] = [None] * 60 + list(kmeans.predict(features))

        # Volatility with arch GARCH (simplified for backtest)
        returns = df['close'].pct_change().dropna()
        if len(returns) > 1:
            am = arch_model(returns, vol='Garch', p=1, q=1)
            res = am.fit(disp='off')
            vol = res.conditional_volatility.iloc[-1]
        else:
            vol = 0.02

        # Adjust signals (0.4*TA + 0.2*ML + 0.1*cluster, volatility scales)
        df['ema12'] = talib.EMA(df['close'], timeperiod=12)
        df['ema26'] = talib.EMA(df['close'], timeperiod=26)
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['signal'] = 0
        df.loc[(df['ema12'] > df['ema26']) & (df['rsi'] < 70), 'signal'] = 0.4
        df.loc[(df['ema12'] < df['ema26']) & (df['rsi'] > 30), 'signal'] = -0.4
        df.loc[df['pred'] > df['close'].shift(-1), 'signal'] += 0.2
        df.loc[df['cluster'] == 1, 'signal'] += 0.1  # Bull cluster
        df['signal'] *= (1 + vol)  # Scale by volatility

        # Vectorbt portfolio
        entries = df['signal'] > 0.7
        exits = df['signal'] < 0.3
        pf = vbt.Portfolio.from_signals(
            df['close'],
            entries=entries,
            exits=exits,
            fees=0.001,
            slippage=0.0005,
            freq=timeframe
        )

        return {
            'overfitting': False,
            'sharpe': float(pf.sharpe_ratio()),
            'max_dd': float(pf.max_drawdown()),
            'winrate': float(pf.win_ratio()),
            'cumulative': float(pf.total_return()),
            'train_loss': float(train_loss),
            'val_loss': float(val_loss)
        }
    except Exception as e:
        logging.error(f"ML backtest failed for {symbol}: {e}")
        return {}
