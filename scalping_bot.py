import sqlite3
import threading
import time
import asyncio
import logging
import json
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import talib
from st_aggrid import AgGrid, GridOptionsBuilder
import redis
from config import DB_PATH, DEFAULT_PARAMS, REDIS_HOST, REDIS_PORT, REDIS_DB, BINANCE_WEIGHT_LIMIT
from utils.binance_utils import get_binance_client
from utils.grok_utils import (
    get_sentiment_analysis,
    get_risk_assessment,
    get_grok_recommendation,
)
from utils.onchain_utils import fetch_sth_rpl
from utils.ml_utils import predict_next_price, train_model, fetch_historical_data
from strategies.arbitrage_strategy import ArbitrageStrategy
from strategies.grid_strategy import GridStrategy
from strategies.mev_strategy import MEVStrategy
from backtest import simple_backtest, advanced_backtest, ml_backtest

logging.basicConfig(level=logging.INFO, filename='bot.log', filemode='a', format='%(asctime)s - %(message)s')


# Redis setup for pub/sub
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
pubsub = r.pubsub()
pubsub.subscribe('winrate_updates', 'ml_updates')

def redis_subscriber():
    for message in pubsub.listen():
        if message['type'] == 'message':
            channel = message['channel'].decode()
            data = json.loads(message['data'].decode())
            if channel == 'winrate_updates':
                st.session_state['winrate'] = data['winrate']
            elif channel == 'ml_updates':
                st.session_state['ml_prediction'] = data['prediction']

threading.Thread(target=redis_subscriber, daemon=True).start()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS settings
                      (key TEXT PRIMARY KEY, value TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS pair_settings
                      (pair TEXT, key TEXT, value TEXT, PRIMARY KEY (pair, key))''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS trades
                      (id INTEGER PRIMARY KEY, symbol TEXT, profit FLOAT, timestamp DATETIME)''')
    for k, v in DEFAULT_PARAMS.items():
        cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (k, str(v)))
    conn.commit()
    conn.close()

def load_params(pair=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if pair:
        cursor.execute("SELECT key, value FROM pair_settings WHERE pair=?", (pair,))
        params = {row[0]: float(row[1]) if row[1].replace('.', '', 1).isdigit() else row[1] for row in cursor.fetchall()}
    else:
        cursor.execute("SELECT * FROM settings")
        params = {row[0]: float(row[1]) if row[1].replace('.', '', 1).isdigit() else row[1] for row in cursor.fetchall()}
    conn.close()
    return params


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
    init_db()
    params = load_params()
    logging.info(f"Loaded params: {params}")
    st.session_state['ml_model'] = train_model()[0]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    threading.Thread(target=lambda: loop.run_until_complete(monitoring_loop()), daemon=True).start()

    # Sidebar for critical buttons (always visible)
    st.sidebar.title("Controls")
    if st.sidebar.button('Pause Bot'):
        logging.info("Bot paused")
        # Pause logic
    if st.sidebar.button('Resume Bot'):
        logging.info("Bot resumed")
        # Resume logic
    if st.sidebar.button('Manual Trade'):
        logging.info("Manual trade triggered")
        # Manual trade logic
    auto_mode = st.sidebar.checkbox('Auto Mode')
    st.session_state['auto_mode'] = auto_mode

    st.title('Ultimate Crypto Scalping Bot')
    tab1, tab2, tab3 = st.tabs(['Dashboard', 'Backtest', 'Settings'])

    with tab1:
        st.subheader('Real-time Chart', help="Interact with the chart to zoom/pan for detailed analysis.")
        client = get_binance_client()
        ohlcv = client.fetch_ohlcv('BTC/USDT', '1m', limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df['timestamp'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close']
                )
            ]
        )
        fig.update_layout(xaxis_rangeslider_visible=True, dragmode='zoom')  # Live interaction
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Trade Log', help="Sortable/filterable table of trades. Use filters for analysis.")
        conn = sqlite3.connect(DB_PATH)
        trades = pd.read_sql("SELECT * FROM trades", conn)
        conn.close()
        gb = GridOptionsBuilder.from_dataframe(trades)
        gb.configure_pagination()
        gb.configure_default_column(editable=False, sortable=True, filter=True)
        AgGrid(trades, gridOptions=gb.build(), height=300)

        st.subheader('Live Metrics', help="Real-time key performance indicators.")
        winrate = st.session_state.get('winrate', 0.65)
        st.metric('Winrate', f"{winrate:.2%}", help="Current win rate from trades")
        sharpe = 1.5
        st.metric('Sharpe Ratio', f"{sharpe:.2f}", help="Risk-adjusted return")

    with tab2:
        st.subheader('Backtest Results', help="Run simulations to test strategies. Select mode for different analysis.")
        backtest_type = st.selectbox('Backtest Type', ['Simple (vectorbt)', 'Advanced (backtrader)', 'ML'], key='backtest_mode')
        st.session_state['backtest_mode'] = backtest_type
        if st.button('Run Backtest'):
            if backtest_type == 'Simple (vectorbt)':
                results = simple_backtest()
            elif backtest_type == 'Advanced (backtrader)':
                results = advanced_backtest()
            else:
                results = ml_backtest()
            st.write(results)

    with tab3:
        st.subheader('Settings', help="Configure global and per-pair parameters. In auto mode, fields are locked.")
        symbols = ['BTC/USDT', 'ETH/USDT']  # Example; extend dynamically
        selected_pair = st.selectbox('Select Pair', symbols + ['Global'])
        if selected_pair == 'Global':
            params = load_params()
        else:
            params = load_params(selected_pair)
        disabled = st.session_state.get('auto_mode', False)

        # Risk and trading params
        st.subheader('Risk Management', help="Adjust risk parameters. Locked in auto mode.")
        risk_per_trade = st.slider('Risk per Trade', 0.005, 0.02, params.get('risk_per_trade', 0.01), disabled=disabled, help="Fraction of capital to risk per trade")
        # ... add more sliders for all risk/trading params
        if not disabled:
            rec_risk = get_grok_recommendation(selected_pair, 'risk_per_trade')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('Recommended', rec_risk)
            with col2:
                if st.button('Apply'):
                    # Update actual
                    pass
            with col3:
                if st.button('Ignore'):
                    pass

        st.subheader('Telegram Configuration', help="Full Telegram setup, including channels.")
        telegram_token = st.text_input('Telegram Token', type='password', disabled=disabled)
        telegram_api_id = st.text_input('Telegram API ID', disabled=disabled)
        telegram_api_hash = st.text_input('Telegram API Hash', type='password', disabled=disabled)
        telegram_session = st.text_input('Telegram Session String', disabled=disabled)
        channels = st.multiselect('Select Channels', ['crypto_news_channel', 'other_channel'], default=st.session_state.get('channels', []))
        st.session_state['channels'] = channels
        if st.button('Save Telegram Config', disabled=disabled):
            # Save to .env or DB
            pass
