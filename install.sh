#!/bin/bash

# Install script for Ultimate Crypto Scalping Bot
# Sets up virtual environment, dependencies, secures .env, initializes DB,
# creates necessary directories, sets permissions, and creates Telethon session
# Note: Run this after cloning repo and creating .env from .env.example

# Ensure virtual environment is activated
# Note: Ensures isolated dependencies to prevent global package conflicts
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Virtual environment not activated. Setting up venv..."
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv || {
            echo "Error: Failed to create virtual environment. Ensure Python3 is installed."
            exit 1
        }
    fi
    source venv/bin/activate || {
        echo "Error: Failed to activate virtual environment."
        exit 1
    }
    echo "Virtual environment activated"
fi

# Install dependencies from requirements.txt
# Note: Includes ccxt, ta-lib, torch, vectorbt, etc. for trading and analysis
echo "Installing dependencies..."
pip install -r requirements.txt || {
    echo "Failed to install dependencies. Check requirements.txt or pip version."
    exit 1
}

# Secure .env file permissions (600: owner read/write only)
# Note: .env contains sensitive API keys; strict permissions prevent unauthorized access
echo "Securing .env file..."
if [ -f .env ]; then
    chmod 600 .env
    echo ".env permissions set to 600"
else
    echo "Warning: .env file not found. Create it from .env.example before proceeding."
    exit 1
fi

# Create necessary directories (e.g., for HDF5 data storage)
# Note: Uses -p to create recursively without errors if already exists
echo "Creating data directories..."
mkdir -p data/hdf5 || {
    echo "Failed to create data directories."
    exit 1
}
# Set permissions for directories (755: owner rwx, others rx for readability)
chmod 755 data data/hdf5
echo "Data directories created and permissions set to 755"

# Initialize SQLite database
# Note: Creates tables for settings and trades as defined in scalping_bot.py
echo "Initializing SQLite database..."
python -c "from scalping_bot import init_db; init_db()" || {
    echo "Failed to initialize database. Check scalping_bot.py or Python environment."
    exit 1
}

# Initialize Telethon session for Telegram integration
# Note: Requires TELEGRAM_API_ID and TELEGRAM_API_HASH in .env; prompts user for phone/code
# Saves session string to .env as TELEGRAM_SESSION
echo "Setting up Telethon session..."
python -c "from telethon.sync import TelegramClient; from telethon.sessions import StringSession; from config import TELEGRAM_API_ID, TELEGRAM_API_HASH; client = TelegramClient(StringSession(), TELEGRAM_API_ID, TELEGRAM_API_HASH); client.start(); print('Session string: ' + client.session.save()); print('Add to .env as TELEGRAM_SESSION')" || {
    echo "Failed to initialize Telethon session. Ensure TELEGRAM_API_ID/HASH are set in .env."
    exit 1
}

echo "Setup complete! Fill TELEGRAM_SESSION in .env, then run 'streamlit run scalping_bot.py' to start the bot."