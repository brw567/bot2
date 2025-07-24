# Install Guide for Ultimate Crypto Scalping Bot on Ubuntu Server 22.04 LTS

This guide extends the basic install.sh script to include full OS-level setup for a production-ready deployment on Ubuntu Server 22.04 LTS. It assumes a fresh install (e.g., from ISO or cloud provider like DigitalOcean/Vultr). Prerequisites: Basic Linux knowledge, access to root via console/SSH.

## 1. OS Configuration
- Log in as root (default on fresh install).
- Update system: `apt update && apt upgrade -y`.
- Set hostname: `hostnamectl set-hostname bot-server`.
- Set timezone: `timedatectl set-timezone UTC` (adjust as needed).
- Install essential packages: `apt install -y sudo curl net-tools ufw`.

Verification: `hostname` shows new name; `timedatectl` shows timezone.

## 2. Creating a Non-Root User
- Create user: `adduser botuser` (set strong password when prompted).
- Add sudo privileges: `usermod -aG sudo botuser`.
- Switch to user: `su - botuser`.

Verification: `sudo whoami` outputs 'root' after password.

## 3. Configuring SSH
- Install OpenSSH: `sudo apt install -y openssh-server`.
- Edit config: `sudo nano /etc/ssh/sshd_config` – set `PermitRootLogin no`, `PasswordAuthentication no` (for key auth), `PubkeyAuthentication yes`.
- Restart SSH: `sudo systemctl restart ssh`.
- Generate SSH key (on local machine): `ssh-keygen -t ed25519`.
- Copy key to server: `ssh-copy-id botuser@server-ip`.
- Test SSH: `ssh botuser@server-ip` (passwordless).
- Secure firewall: `sudo ufw allow OpenSSH && sudo ufw enable`.

Verification: `sudo systemctl status ssh` shows active; SSH from local connects without password.

## 4. Initial Fixing Folder Structure
- Create app dir: `sudo mkdir -p /opt/bot/data/hdf5`.
- Set ownership: `sudo chown -R botuser:botuser /opt/bot`.
- Set permissions: `sudo chmod -R 755 /opt/bot`; `sudo chmod 700 /opt/bot/data/hdf5` (secure data).

Verification: `ls -la /opt/bot` shows correct owner/perms.

## 5. Deployment of Required Linux Packages
- Install build essentials (for ta-lib, etc.): `sudo apt install -y build-essential libta-lib-dev`.
- Install Python3/venv: `sudo apt install -y python3 python3-venv python3-pip`.
- Install Git: `sudo apt install -y git`.

Verification: `python3 --version` (3.10+); `git --version`; `pip3 --version`.

## 6. Git Configuration
- Clone repo: `git clone https://github.com/brw567/bot1 /opt/bot`.
- Set global config (optional): `git config --global user.name "botuser"`; `git config --global user.email "bot@example.com"`.

Verification: `cd /opt/bot && git status` shows clean repo.

## 7. Redis DB Deployment and Configuration
- Install Redis: `sudo apt install -y redis-server`.
- Edit config: `sudo nano /etc/redis/redis.conf` – set `bind 127.0.0.1`, `requirepass strongpassword` (replace with secure pass), `supervised systemd`.
- Restart Redis: `sudo systemctl restart redis-server`.
- Enable on boot: `sudo systemctl enable redis-server`.

Verification: `redis-cli -a strongpassword ping` outputs 'PONG'; `sudo systemctl status redis-server` shows active.

## 8. Bot Deployment
- Navigate: `cd /opt/bot`.
- Create venv: `python3 -m venv venv && source venv/bin/activate`.
- Run install: `./install.sh` (installs deps, inits DB, sets Telethon).
- Fill .env: `nano .env` (add keys, Redis host/port, password).
- Start bot: `streamlit run scalping_bot.py` (or as service: create /etc/systemd/system/bot.service with [Unit]/[Service]/[Install] sections, `ExecStart=/opt/bot/venv/bin/streamlit run /opt/bot/scalping_bot.py`, `sudo systemctl enable --now bot`).

Verification: Bot GUI at http://server-ip:8501; logs in bot.log.

## Troubleshooting
- SSH fail: Check ufw status, sshd_config.
- Redis auth: Verify requirepass in config.
- Deps fail: `sudo apt update`; check ta-lib build logs.
- Bot crash: Check bot.log; ensure venv activated.

Note: Use strong passwords, backup .env, monitor with `htop`/`systemctl`.