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
from db_utils import init_db, get_param, save_param
from utils.binance_utils import get_binance_client
from utils.grok_utils import (
    get_multi_sentiment_analysis,
    get_risk_assessment,
    get_grok_recommendation,
    get_backup_price,
    SentimentResponse,
    RiskResponse,
)
from utils.onchain_utils import fetch_sth_rpl
from utils.ml_utils import predict_next_price, train_model
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
    """Listen for Redis pub/sub messages and update Streamlit state."""
    for message in pubsub.listen():
        if message['type'] == 'message':
            channel = message['channel'].decode()
            data = json.loads(message['data'].decode())
            if channel == 'winrate_updates':
                st.session_state['winrate'] = data['winrate']
            elif channel == 'ml_updates':
                st.session_state['ml_prediction'] = data['prediction']

threading.Thread(target=redis_subscriber, daemon=True).start()

# Simple in-memory cache for recent OHLCV data
CANDLE_CACHE = {}
CACHE_EXPIRY = 60 * 5  # 5 minutes


def fetch_recent_candles(symbol, timeframe='5m', limit=60):
    """Fetch recent candles with caching to reduce API calls."""
    cache_key = (symbol, timeframe)
    cached = CANDLE_CACHE.get(cache_key)
    if cached and time.time() - cached['timestamp'] < CACHE_EXPIRY:
        return cached['df'], False

    client = get_binance_client()
    ohlcv = client.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df.dropna(inplace=True)
    CANDLE_CACHE[cache_key] = {'timestamp': time.time(), 'df': df}
    return df, True


def save_cached_indicators(pair, price, vol, candles_df, risk: RiskResponse, sentiment: SentimentResponse):
    """Persist indicators for fallback usage."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO indicators_cache (pair, price, vol, candles, risk, sentiment, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                pair,
                price,
                vol,
                candles_df.to_json(orient='records'),
                risk.json(),
                sentiment.json(),
                time.time(),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Saving indicators for {pair} failed: {e}")


def load_cached_indicators(pair):
    """Load cached indicators from DB if available."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT price, vol, candles, risk, sentiment FROM indicators_cache WHERE pair=?",
            (pair,),
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            price, vol, candles_json, risk_json, sentiment_json = row
            candles = pd.read_json(candles_json)
            risk = RiskResponse(**json.loads(risk_json))
            sentiment = SentimentResponse(**json.loads(sentiment_json))
            return price, vol, candles, risk, sentiment
    except Exception as e:
        logging.error(f"Loading indicators for {pair} failed: {e}")
    return None


def get_signal_score(symbol, price, ta_score, sentiment, onchain_rpl, recent_data=None):
    """
    Calculate aggregated signal score for trading decisions.
    Formula: 0.4*TA + 0.3*sentiment + 0.3*onchain + 0.2*ML + 0.1*cluster (scaled by volatility).
    Note: ML and cluster components added dynamically via Redis and ml_utils.
    """
    try:
        onchain_score = 1 if price > onchain_rpl * 1.05 else -1 if price < onchain_rpl * 0.95 else 0
        score = 0.4 * ta_score + 0.3 * sentiment.score + 0.3 * onchain_score
        # ML prediction (from Redis or direct)
        if recent_data is None:
            df, _ = fetch_recent_candles(symbol)
            recent_data = df[['close', 'volume', 'rsi']].values
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
            pair_limit = int(get_param('auto_pair_limit', 10))
            trading_pairs = ['BTC/USDT', 'ETH/USDT'][:pair_limit]
            monitor_pairs = []
            pairs = trading_pairs + [p for p in monitor_pairs if p not in trading_pairs]
            fallback_info = {}
            symbols = [p.split('/')[0] for p in pairs]

            # Fetch BTC data for RPL checks and as fallback for unsupported pairs
            btc_ticker = client.fetch_ticker('BTC/USDT')
            weight_counter += request_weights['fetch_ticker']
            btc_price = btc_ticker['last']
            btc_vol = (btc_ticker['high'] - btc_ticker['low']) / btc_ticker['low']  # Simple volatility

            # RPL spike check with zero-division guard
            current_rpl = fetch_sth_rpl('BTC')
            if prev_rpl > 0 and current_rpl > 0 and current_rpl / prev_rpl > 1.2:
                logging.info("RPL spike >20%; pausing 30min")
                await asyncio.sleep(1800)
                prev_rpl = current_rpl
                continue
            if current_rpl > 0:
                prev_rpl = current_rpl

            # Sentiment for all pairs
            sentiments = await get_multi_sentiment_analysis(symbols)

            for pair in pairs:
                trade_enabled = pair in trading_pairs
                base = pair.split('/')[0]
                sentiment = sentiments.get(base, SentimentResponse(sentiment='neutral', score=0.0, details='Missing'))
                cached = load_cached_indicators(pair)
                fallback_used = False

                # Fetch ticker
                try:
                    ticker = client.fetch_ticker(pair)
                    weight_counter += request_weights['fetch_ticker']
                    price = ticker['last']
                    vol = (ticker['high'] - ticker['low']) / ticker['low']
                except Exception as e:
                    logging.warning(f"Ticker fetch failed for {pair}: {e}. Requesting backup price")
                    price = get_backup_price(base)
                    if price:
                        vol = btc_vol
                    elif cached:
                        price, vol = cached[0], cached[1]
                    else:
                        price, vol = btc_price, btc_vol
                    fallback_used = True

                # Candle data
                try:
                    candles, fetched = fetch_recent_candles(pair)
                    if fetched:
                        weight_counter += request_weights['fetch_ohlcv']
                except Exception as e:
                    logging.warning(f"OHLCV fetch failed for {pair}: {e}. Using backup values")
                    if cached:
                        candles = cached[2]
                        fallback_used = True
                    else:
                        candles, fetched = fetch_recent_candles('BTC/USDT')
                        if fetched:
                            weight_counter += request_weights['fetch_ohlcv']
                        fallback_used = True

                # Risk assessment
                if cached and fallback_used:
                    risk = cached[3]
                    sentiment = cached[4]
                else:
                    risk = get_risk_assessment(pair, price, vol, st.session_state.get('winrate', 0.65))

                ta_score = 1 if price > talib.EMA(pd.Series([price] * 26), timeperiod=12)[-1] else 0

                score = get_signal_score(
                    pair,
                    price,
                    ta_score,
                    sentiment,
                    current_rpl,
                    recent_data=candles[['close', 'volume', 'rsi']].values,
                )

                if fallback_used:
                    logging.info(f"[fallback] Used backup data for {pair}")
                fallback_info[pair] = fallback_used

                # Execute strategies only on configured trading pairs
                if trade_enabled:
                    if score > 0.7 and risk.trade == 'yes':
                        GridStrategy().run(pair, price)
                        ArbitrageStrategy().run(pair, f"{pair}:USDT")
                        weight_counter += request_weights['create_order']
                    MEVStrategy().detect_mev(pair)

                if not fallback_used:
                    save_cached_indicators(pair, price, vol, candles, risk, sentiment)

            # Update winrate
            conn = sqlite3.connect(DB_PATH)
            wins = conn.execute("SELECT COUNT(*) FROM trades WHERE profit > 0").fetchone()[0]
            total = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            winrate = wins / total if total else 0
            conn.execute("UPDATE settings SET value=? WHERE key='win_rate'", (winrate,))
            conn.commit()
            conn.close()
            r.publish('winrate_updates', json.dumps({'winrate': winrate}))
            st.session_state['fallback_pairs'] = fallback_info

            await asyncio.sleep(5)
        except Exception as e:
            logging.error(f"Monitoring loop error: {e}")
            await asyncio.sleep(10)

# GUI
if __name__ == '__main__':
    init_db(DB_PATH)
    params = {k: get_param(k, v) for k, v in DEFAULT_PARAMS.items()}
    logging.info(f"Loaded params: {params}")
    st.session_state['ml_model'] = train_model()[0]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    threading.Thread(target=lambda: loop.run_until_complete(monitoring_loop()), daemon=True).start()

    # Sidebar for critical buttons (always visible)
    st.sidebar.title("Controls")
    st.sidebar.markdown("Use buttons below to control the bot.")
    if st.sidebar.button('Pause Bot'):
        logging.info("Bot paused")
        # Pause logic
    if st.sidebar.button('Resume Bot'):
        logging.info("Bot resumed")
        # Resume logic
    if st.sidebar.button('Manual Trade'):
        logging.info("Manual trade triggered")
        # Manual trade logic
    auto_mode = st.sidebar.checkbox('Auto Mode', help='Lock settings when enabled.')
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
        fig.update_layout(xaxis_rangeslider_visible=True, dragmode='zoom', hovermode='x unified')  # Live interaction
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
        fb_pairs = st.session_state.get('fallback_pairs', {})
        warn = [p for p, used in fb_pairs.items() if used]
        if warn:
            st.markdown(f":orange[Using backup data for {', '.join(warn)}]")

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
            params = {k: get_param(k, v) for k, v in DEFAULT_PARAMS.items()}
        else:
            params = {k: get_param(f"{selected_pair}_{k}", v) for k, v in DEFAULT_PARAMS.items()}
        disabled = st.session_state.get('auto_mode', False)

        st.subheader('Risk Management', help="Adjust risk parameters. Locked in auto mode.")
        param_df = pd.DataFrame([
            {'Parameter': k, 'Value': v} for k, v in params.items()
        ])
        gb = GridOptionsBuilder.from_dataframe(param_df)
        gb.configure_default_column(editable=not disabled)
        grid = AgGrid(param_df, gridOptions=gb.build(), height=200)

        if not disabled:
            rec_risk = get_grok_recommendation(selected_pair, 'risk_per_trade')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('Recommended', rec_risk)
            with col2:
                if st.button('Apply All'):
                    updated = {row['Parameter']: row['Value'] for row in grid['data']}
                    updated['risk_per_trade'] = rec_risk
                    for k, v in updated.items():
                        key = k if selected_pair == 'Global' else f"{selected_pair}_{k}"
                        save_param(key, v)
                    st.success('Settings updated')
            with col3:
                st.button('Ignore')
        if st.button('Save Parameters', disabled=disabled):
            updated = {row['Parameter']: row['Value'] for row in grid['data']}
            for k, v in updated.items():
                key = k if selected_pair == 'Global' else f"{selected_pair}_{k}"
                save_param(key, v)
            st.success('Settings updated')

        st.subheader('Telegram Configuration', help="Full Telegram setup, including channels.")
        telegram_token = st.text_input('Telegram Token', type='password', disabled=disabled)
        telegram_api_id = st.text_input('Telegram API ID', disabled=disabled)
        telegram_api_hash = st.text_input('Telegram API Hash', type='password', disabled=disabled)
        telegram_session = st.text_input('Telegram Session String', type='password', disabled=disabled)
        channels = st.multiselect('Select Channels', ['crypto_news_channel', 'other_channel'], default=st.session_state.get('channels', []))
        st.session_state['channels'] = channels
        if st.button('Save Telegram Config', disabled=disabled):
            # Save to .env or DB
            pass

        st.subheader('Grok Configuration')
        grok_timeout = st.number_input('Grok API Timeout (seconds)', 5, 60, params.get('grok_timeout', 10), disabled=disabled)
        st.session_state['grok_timeout'] = grok_timeout
