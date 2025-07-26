import sqlite3
from typing import Optional

_DB_PATH = None


def init_db(db_path: str) -> None:
    """Initialize the SQLite database and required tables."""
    global _DB_PATH
    _DB_PATH = db_path
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS bot_state (
            id INTEGER PRIMARY KEY CHECK (id=1),
            state TEXT
        )"""
    )
    cursor.execute(
        "INSERT OR IGNORE INTO bot_state (id, state) VALUES (1, 'stopped')"
    )
    from config import DEFAULT_PARAMS
    for k, v in DEFAULT_PARAMS.items():
        cursor.execute(
            "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
            (k, str(v)),
        )
    conn.commit()
    conn.close()


def _get_conn() -> Optional[sqlite3.Connection]:
    if not _DB_PATH:
        return None
    return sqlite3.connect(_DB_PATH)


def save_param(key: str, value: str) -> None:
    conn = _get_conn()
    if conn is None:
        return
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, str(value)),
        )


def get_param(key: str, default=None):
    conn = _get_conn()
    if conn is None:
        return default
    cursor = conn.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else default


def set_state(value: str) -> None:
    conn = _get_conn()
    if conn is None:
        return
    with conn:
        conn.execute("UPDATE bot_state SET state=? WHERE id=1", (value,))


def get_state() -> Optional[str]:
    conn = _get_conn()
    if conn is None:
        return None
    cursor = conn.execute("SELECT state FROM bot_state WHERE id=1")
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def check_user(username: str, password: str) -> bool:
    conn = _get_conn()
    if conn is None:
        return False
    cursor = conn.execute(
        "SELECT password FROM users WHERE username=?", (username,)
    )
    row = cursor.fetchone()
    conn.close()
    return row is not None and row[0] == password
