import asyncio
import logging
from logging.handlers import RotatingFileHandler
import sqlite3
from datetime import datetime
import redis
try:
    from telethon import TelegramClient, events
except Exception:  # pragma: no cover - handled in tests with stub
    from telethon import TelegramClient  # type: ignore
    events = None  # type: ignore
from telethon.sessions import StringSession
from config import (
    TELEGRAM_API_ID,
    TELEGRAM_API_HASH,
    TELEGRAM_SESSION,
    DB_PATH,
    NOTIFICATIONS_ENABLED,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
)

handler = RotatingFileHandler('bot.log', maxBytes=1_000_000, backupCount=5)
logging.basicConfig(level=logging.INFO, handlers=[handler],
                    format='%(asctime)s - %(message)s')

# Redis connection for caching incoming Telegram messages
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

# Global Telegram client used for the background listener
listener_client = None


def cache_message(text: str) -> None:
    """Store a Telegram message in Redis for later analysis."""
    try:
        redis_client.lpush("telegram:messages", text)
        redis_client.ltrim("telegram:messages", 0, 999)
    except Exception as e:
        logging.error(f"Failed to cache message: {e}")

async def get_client():
    """
    Initialize and return a Telethon client for Telegram interactions.

    Returns:
        TelegramClient: Configured client instance.

    Note: Uses credentials from config.py; StringSession for persistent sessions.
    """
    try:
        client = TelegramClient(StringSession(TELEGRAM_SESSION), TELEGRAM_API_ID, TELEGRAM_API_HASH)
        return client
    except Exception as e:
        logging.error(f"Telegram client initialization failed: {e}")
        raise

async def send_notification(message):
    """
    Send a notification to the configured Telegram user or channel.

    Args:
        message (str): Message to send.

    Note: Sends to 'me' (self) for simplicity; adjust for bot or group in prod.
    """
    if not NOTIFICATIONS_ENABLED:
        logging.info("Notifications disabled; message suppressed")
        return
    client = None
    try:
        client = await get_client()
        await client.start()
        await client.send_message('me', message)
        logging.info(f"Notification sent: {message}")
    except Exception as e:
        logging.error(f"Telegram notification failed: {e}")
    finally:
        if client:
            await client.disconnect()

async def fetch_channel_messages(channel, limit=100):
    """
    Fetch recent messages from a Telegram channel for sentiment analysis.

    Args:
        channel (str): Telegram channel name (e.g., 'crypto_news_channel').
        limit (int): Max number of messages to fetch (default 100, per immediate task).

    Returns:
        list: List of message texts (filtered for non-empty).

    Note: Uses timestamp filtering via DB to fetch only new messages since last pull,
    addressing immediate task for extended sentiment fetch. Stores last pull time in DB
    as an ISO-formatted timestamp.
    """
    client = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key='last_telegram_pull'")
        last_pull = cursor.fetchone()
        last_pull_ts = (
            datetime.fromisoformat(last_pull[0])
            if last_pull
            else datetime.fromisoformat("1970-01-01T00:00:00")
        )
        conn.close()

        client = await get_client()
        await client.start()
        messages = await client.get_messages(channel, limit=limit, min_id=0, offset_date=last_pull_ts)
        texts = [msg.text for msg in messages if msg.text and msg.date > last_pull_ts]
        for t in texts:
            cache_message(t)
        logging.info(f"Fetched {len(texts)} messages from {channel}")

        # Update last pull timestamp
        if messages:
            new_ts = messages[0].date.isoformat()
            conn = sqlite3.connect(DB_PATH)
            conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('last_telegram_pull', ?)", (new_ts,))
            conn.commit()
            conn.close()

        return texts
    except Exception as e:
        logging.error(f"Fetch messages failed for {channel}: {e}")
        return []
    finally:
        if client:
            await client.disconnect()


async def _process_event(event):
    """Process new Telegram messages through Grok for insights."""
    try:
        text = event.message.message
        cache_message(text)
        from utils.grok_utils import get_grok_insights  # type: ignore
        await get_grok_insights(text)
    except Exception as e:
        logging.error(f"get_grok_insights failed: {e}")


async def setup_listener():
    """Start background Telegram listener for incoming messages."""
    global listener_client
    listener_client = await get_client()
    if events is not None:
        listener_client.add_event_handler(_process_event, events.NewMessage())
    await listener_client.start()
    asyncio.create_task(listener_client.run_until_disconnected())
