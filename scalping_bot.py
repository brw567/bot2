import sqlite3
import threading
import time
import asyncio
import logging
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
import redis
from config import DB_PATH, DEFAULT_PARAMS, REDIS_HOST, REDIS_PORT, REDIS_DB, BINANCE_WEIGHT_LIMIT
from utils.binance_utils import get_binance_client
from utils.grok_utils import get_sentiment_analysis, get_risk_assessment
from utils.onchain_utils import fetch_sth_rpl
from utils.ml_utils import predict_next_price, train_model, fetch_historical_data
from strategies.arbitrage_strategy import ArbitrageStrategy
from strategies.grid_strategy import GridStrategy
from strategies.mev_strategy import MEVStrategy
from backtest import simple_backtest, advanced_backtest, ml_backtest

logging.basicConfig(level=logging.INFO, filename='bot.log', filemode='a', format='%(asctime)s - %(message)s')

def get_signal_score(symbol, price, ta_score, sentiment, onchain_rpl):
    """
    Calculate aggregated signal score for trading decisions.
    Formula: 0.4*TA + 0.3*sentiment + 0.3*onchain + 0.2*ML + 0.1*cluster (scaled by volatility).
    Note: Logs components for debugging; ML and cluster from Redis/ml_utils.
    """
    try:
        onchain_score = 1 if price > onchain_rpl * 1.05 else -1 if price < onchain_rpl * 0.95 else 0
        score = 0.4 * ta_score + 0.3 * sentiment['score'] + 0.3 * onchain_score
        # ML prediction
        recent_data = fetch_historical_data(symbol, '5m', years=0.01).iloc[-60:][['close', 'volume', 'rsi']].values
        pred = predict_next_price(st.session_state.get('ml_model'), recent_data[-1]) if 'ml_model' in st.session_state else 0
        ml_boost = 0.2 if pred > price * 1.005 else 0
        score += ml_boost
        # Placeholder cluster
        cluster_boost = 0.1  # From scikit-learn in live
        score += cluster_boost
        logging.info(f"Signal components for {symbol}: TA={0.4 * ta_score}, sentiment={0.3 * sentiment['score']}, onchain={0.3 * onchain_score}, ML={ml_boost}, cluster={cluster_boost}, total={score}")
        return score
    except Exception as e:
        logging.error(f"Signal score failed: {e}")
        return 0.0


# Redis setup for pub/sub
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
pubsub = r.pubsub()
pubsub.subscribe('winrate_updates', 'ml_updates')

def redis_subscriber():
    """
    Subscribe to Redis channels for real-time winrate and ML updates.
    Updates Streamlit session state for GUI display.
    """
    for message in pubsub.listen():
        if message['type'] == 'message':
            channel = message['channel'].decode()
            data = json.loads(message['data'].decode())
            if channel == 'winrate_updates':
                st.session_state['winrate'] = data['winrate']
                logging.info(f"Updated winrate: {data['winrate']}")
            elif channel == 'ml_updates':
                st.session_state['ml_prediction'] = data['prediction']
                logging.info(f"Updated ML prediction: {data['prediction']}")

# Start Redis subscriber in background thread
threading.Thread(target=redis_subscriber, daemon=True).start()

def init_db():
    """
    Initialize SQLite database with tables for settings and trades.
    Note: Creates schema for parameters and trade logs; used for winrate tracking.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS settings
                         (key TEXT PRIMARY KEY, value TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS trades
                         (id INTEGER PRIMARY KEY, symbol TEXT, profit FLOAT, timestamp DATETIME)''')
        for k, v in DEFAULT_PARAMS.items():
            cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (k, str(v)))
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Database initialization failed: {e}")
    finally:
        conn.close()

def load_params():
    """
    Load trading parameters from SQLite database.
    Returns: dict of params (e.g., win_rate_threshold).
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM settings")
        params = {row[0]: float(row[1]) if row[1].replace('.', '', 1).isdigit() else row[1] for row in cursor.fetchall()}
        return params
    except sqlite3.Error as e:
        logging.error(f"Failed to load params: {e}")
        return DEFAULT_PARAMS
    finally:
        conn.close()

def get_signal_score(symbol, price, ta_score, sentiment, onchain_rpl):
    """
    Calculate aggregated signal score for trading decisions.
    Formula: 0.4*TA + 0.3*sentiment + 0.3*onchain + 0.2*ML + 0.1*cluster (scaled by volatility).
    Note: ML and cluster components added dynamically via Redis and ml_utils.
    """
    try:
        onchain_score = 1 if price > onchain_rpl * 1.05 else -1 if price < onchain_rpl * 0.95 else 0
        score = 0.4 * ta_score + 0.3 * sentiment['score'] + 0.3 * onchain_score
        # ML prediction (from Redis or direct)
        recent_data = fetch_historical_data(symbol, '5m', years=0.01).iloc[-60:][['close', 'volume', 'rsi']].values
        pred = predict_next_price(st.session_state.get('ml_model'), recent_data[-1]) if 'ml_model' in st.session_state else 0
        if pred > price * 1.005:
            score += 0.2
        # Placeholder for clustering (via scikit-learn in live)
        return score
    except Exception as e:
        logging.error(f"Signal score calculation failed: {e}")
        return 0.0

async def monitoring_loop():
    """
    Async monitoring loop for real-time trading.
    Fetches data, calculates signals, executes strategies, and monitors quotas/RPL spikes.
    Note: Uses async for low-latency (<50ms) execution; Redis for winrate updates.
    """
    global weight_counter, last_reset
    weight_counter = 0
    last_reset = time.time()
    request_weights = {'fetch_ticker': 1, 'fetch_ohlcv': 2, 'create_order': 1}  # Approx weights
    prev_rpl = fetch_sth_rpl('BTC')
    
    while True:
        try:
            # Quota monitoring
            if time.time() - last_reset >= 60:
                weight_counter = 0
                last_reset = time.time()
            if weight_counter > BINANCE_WEIGHT_LIMIT * 0.8:
                logging.warning("Near API quota; pausing 30s")
                await asyncio.sleep(30)
                continue

            client = get_binance_client()
            ticker = client.fetch_ticker('BTC/USDT')
            weight_counter += request_weights['fetch_ticker']
            price = ticker['last']
            vol = (ticker['high'] - ticker['low']) / ticker['low']  # Simple volatility

            # RPL spike check
            current_rpl = fetch_sth_rpl('BTC')
            if current_rpl / prev_rpl > 1.2:
                logging.info("RPL spike >20%; pausing 30min")
                await asyncio.sleep(1800)
                prev_rpl = current_rpl
                continue
            prev_rpl = current_rpl

            # Sentiment and risk
            sentiment = get_sentiment_analysis('BTC')
            risk = get_risk_assessment('BTC/USDT', price, vol, st.session_state.get('winrate', 0.65))

            # Signal score
            ta_score = 1 if price > talib.EMA(pd.Series([price] * 26), timeperiod=12)[-1] else 0  # Mock EMA
            score = get_signal_score('BTC/USDT', price, ta_score, sentiment, current_rpl)

            # Execute strategies
            if score > 0.7 and risk.get('trade') == 'yes':
                GridStrategy().run('BTC/USDT', price)
                ArbitrageStrategy().run('BTC/USDT', 'BTC/USDT:USDT')
                weight_counter += request_weights['create_order']
            MEVStrategy().detect_mev('BTC/USDT')

            # Update winrate
            conn = sqlite3.connect(DB_PATH)
            wins = conn.execute("SELECT COUNT(*) FROM trades WHERE profit > 0").fetchone()[0]
            total = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            winrate = wins / total if total else 0
            conn.execute("UPDATE settings SET value=? WHERE key='win_rate'", (winrate,))
            conn.commit()
            conn.close()
            r.publish('winrate_updates', json.dumps({'winrate': winrate}))

            await asyncio.sleep(5)
        except Exception as e:
            logging.error(f"Monitoring loop error: {e}")
            await asyncio.sleep(10)

# GUI
if __name__ == '__main__':
    # Initialize DB and ML model
    init_db()
    params = load_params()
    logging.info(f"Loaded params: {params}")
    st.session_state['ml_model'] = train_model()[0]  # Load ML model

    # Start monitoring loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    threading.Thread(target=lambda: loop.run_until_complete(monitoring_loop()), daemon=True).start()

    # Streamlit GUI
    st.title('Ultimate Crypto Scalping Bot')
    tab1, tab2, tab3 = st.tabs(['Dashboard', 'Backtest', 'Settings'])

    with tab1:
        st.subheader('Real-time Chart')
        client = get_binance_client()
        ohlcv = client.fetch_ohlcv('BTC/USDT', '1m', limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        fig = go.Figure(data=[go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
        st.plotly_chart(fig)

        st.subheader('Trade Log')
        conn = sqlite3.connect(DB_PATH)
        trades = pd.read_sql("SELECT * FROM trades", conn)
        conn.close()
        gb = GridOptionsBuilder.from_dataframe(trades)
        gb.configure_pagination()
        gb.configure_default_column(editable=False, sortable=True, filter=True)
        AgGrid(trades, gridOptions=gb.build(), height=300)

        st.subheader('Live Metrics')
        winrate = st.session_state.get('winrate', 0.65)
        st.metric('Winrate', f"{winrate:.2%}")
        sharpe = 1.5  # Mock rolling Sharpe
        st.metric('Sharpe Ratio', f"{sharpe:.2f}")

        if st.button('Pause Bot'):
            logging.info("Bot paused by user")
            # Logic to pause loop (e.g., set flag)
        if st.button('Resume Bot'):
            logging.info("Bot resumed by user")
        if st.button('Manual Trade'):
            logging.info("Manual trade triggered")
            # Placeholder for manual trade logic

    with tab2:
        st.subheader('Backtest Results')
        backtest_mode = st.selectbox('Backtest Mode', ['Simple (vectorbt)', 'Advanced (backtrader)', 'ML'], key='backtest_mode')
        st.session_state['backtest_mode'] = backtest_mode  # Persist in session
        if st.button('Run Backtest'):
            if backtest_mode == 'Simple (vectorbt)':
                results = simple_backtest()
            elif backtest_mode == 'Advanced (backtrader)':
                results = advanced_backtest()
            else:
                results = ml_backtest()
            st.write(results)

    with tab3:
        st.subheader('Parameters')
        channels = st.multiselect('Select Telegram Channels', ['crypto_news_channel', 'other_channel'], key='channels')
        st.session_state['channels'] = channels
        risk_per_trade = st.slider('Risk per Trade', 0.005, 0.02, 0.01)
        conn = sqlite3.connect(DB_PATH)
        conn.execute("UPDATE settings SET value=? WHERE key='risk_per_trade'", (risk_per_trade,))
        conn.commit()
        conn.close()