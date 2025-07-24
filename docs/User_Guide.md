# User Guide

## Installation
1. Clone repo: `git clone https://github.com/brw567/bot1`.
2. Create venv: `python -m venv venv && source venv/bin/activate`.
3. Run `./install.sh` (installs deps, secures .env, inits DB, creates data dirs, sets Telethon session).
4. Fill `.env` from `.env.example` (API keys, session string).

## Usage
- Run `streamlit run scalping_bot.py`.
- GUI Tabs:
  - Dashboard: Real-time charts, trade log (aggrid sortable), live metrics (winrate, Sharpe).
  - Backtest: Select mode (simple/vectorbt, advanced/backtrader, ML), run sims.
  - Settings: Select Telegram channels, adjust params, and manage per-pair settings via AgGrid.
  - Sidebar controls remain visible for quick pause or manual trades.
  - Telegram: Notifications for trades/MEV; sentiment from selected channels.

## Features
- Strategies: Arbitrage (spread >0.2%), grid (VWAP-centered), MEV detection (order book snapshots).
- Risk: Dynamic SL/TP, Kelly sizing (<1% risk/trade).
- Pauses: RPL spikes (>20%), quotas (>80% limit), drawdowns.
- Fallback prices from Grok shown in yellow if Binance data fails.
- Grok-recommended pairs (5x limit) monitored for analytics.

## Troubleshooting
- API errors: Check keys in `.env`.
- No messages: Verify channels in GUI.
- Run cleanup: `python cleanup_hdf5.py` for HDF5 bloat.
