# Install Guide for Ultimate Crypto Scalping Bot on Ubuntu Server 22.04 LTS

This guide details the full deployment process for the Ultimate Crypto Scalping Bot on a fresh Ubuntu Server 22.04 LTS instance (e.g., AWS EC2, DigitalOcean, or bare metal). It covers OS configuration, user creation, SSH setup, folder structure, Linux packages, Git configuration, Redis deployment, and bot deployment. Run commands as root or with sudo where noted. For security, a non-root user is used for bot operations.

## 1. OS Configuration

Update and upgrade system packages to ensure a secure base:
```sudo apt update && sudo apt upgrade -y```

Set timezone to UTC for trading consistency:
```sudo timedatectl set-timezone UTC```

Install essential tools:
```sudo apt install -y curl wget unzip vim net-tools```


## 2. Creating a User

Create a dedicated user `botuser` to isolate bot operations:
```sudo adduser botuser```

Add user to sudo group for admin tasks:
```sudo usermod -aG sudo botuser```

Switch to the new user:
```su - botuser```


## 3. Configuring SSH

Generate an SSH key pair on your local machine:
```ssh-keygen -t ed25519 -C "your_email@example.com"```

Copy the public key to the server (`replace server_ip`):
```ssh-copy-id botuser@server_ip```

On the server, disable password authentication for security:
```sudo vim /etc/ssh/sshd_config```

Set PasswordAuthentication no and PubkeyAuthentication yes. Restart SSH:
```sudo systemctl restart ssh```

Test SSH from local machine:
```ssh botuser@server_ip```


## 4. Initial Fixing Folder Structure

Create the project directory:
```
sudo mkdir -p /opt/bot
sudo chown botuser:botuser /opt/bot
cd /opt/bot
```

Create data and log directories:
```
mkdir -p data/hdf5 logs
chmod 755 data data/hdf5 logs
```


## 5. Deployment of Required Linux Packages

Install Python 3.12, Git, Redis, and dependencies:
```
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev build-essential libssl-dev libffi-dev libpq-dev git redis-server
```

**Note:** `ta-lib` may require additional binaries; see TA-Lib install guide if errors occur.


## 6. Git Configuration

Configure Git:
```
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
```

Clone the repository:
```
git clone https://github.com/brw567/bot1.git .
```

Secure Git by disabling credential storage:
```
git config --global credential.helper ""
```


## 7. Redis DB Deployment and Configuration

Start and enable Redis:
```
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

Configure Redis for low-latency pub/sub (edit /etc/redis/redis.conf):
```
sudo vim /etc/redis/redis.conf
```
Set:
````
bind 127.0.0.1 (local only for security)
maxmemory 256mb (limit for bot scale)
maxmemory-policy allkeys-lru (evict least used)
````

Restart Redis:
```sudo systemctl restart redis-server```

Test Redis:
```redis-cli ping  # Should return "PONG"```


## 8. Bot Deployment

Create and activate virtual environment:

python3.12 -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Secure .env file:

cp .env.example .env
vim .env  # Fill API keys (Binance, Telegram, Grok, Dune, Redis)
chmod 600 .env

Run install script to initialize DB and Telethon session:

./install.sh

Note: install.sh prompts for Telegram phone/code; copy the session string to .env as TELEGRAM_SESSION.

Run bot as a systemd service for continuous operation:

sudo vim /etc/systemd/system/bot.service

Add:

[Unit]
Description=Ultimate Crypto Scalping Bot
After=network.target redis-server.service

[Service]
User=botuser
WorkingDirectory=/opt/bot
ExecStart=/opt/bot/venv/bin/streamlit run scalping_bot.py
Restart=always

[Install]
WantedBy=multi-user.target

Enable and start the service:

sudo systemctl daemon-reload
sudo systemctl enable bot
sudo systemctl start bot

Monitor the service:

sudo systemctl status bot
tail -f /opt/bot/bot.log

Troubleshooting





SSH Issues: Check /var/log/auth.log.



Redis: Run redis-cli monitor for debugging.



Bot Errors: Check bot.log or journalctl -u bot.



Dependency Issues: Re-run pip install -r requirements.txt.



Updates: Pull and restart:

git pull
pip install -r requirements.txt
sudo systemctl restart bot

Note: Ensure DUNE_QUERY_ID in .env matches a valid Dune query for STH RPL. Run in paper mode (mock CCXT trades) before live trading.