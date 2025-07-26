-- Initialization script for the trading bot database

-- Settings table stores configurable parameters
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Users table with plain text passwords for simplicity
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
);

-- Bot state table keeps a single row tracking current state
CREATE TABLE IF NOT EXISTS bot_state (
    id INTEGER PRIMARY KEY CHECK (id=1),
    state TEXT
);

-- Insert default bot state if not present
INSERT OR IGNORE INTO bot_state (id, state) VALUES (1, 'stopped');

-- Default admin credentials
INSERT OR IGNORE INTO users (username, password) VALUES ('admin', 'admin');

-- Default settings values
INSERT OR REPLACE INTO settings (key, value) VALUES
    ('win_rate_threshold', '0.6'),
    ('max_consec_losses', '3'),
    ('slippage_tolerance', '0.001'),
    ('risk_per_trade', '0.01'),
    ('grok_timeout', '10'),
    ('auto_pair_limit', '10'),
    ('swap_pair_multiplier', '10'),
    ('volatility_check_interval', '14400'),
    ('volatility_threshold_percent', '50.0');
