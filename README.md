# Ultimate Crypto Scalping Bot

## Project Summary
The Ultimate Crypto Scalping Bot is an advanced trading tool designed for high-frequency scalping in cryptocurrency markets, focusing on small price movements for quick profits. It integrates real-time data analysis (technical indicators, sentiment, on-chain metrics), AI-driven decisions (Grok for risk/tuning), machine learning predictions (LSTM), and automated executions on Binance (spot/futures). Key features include modular strategies (arbitrage, grid, MEV detection), risk management (dynamic SL/TP, Kelly sizing), backtesting (vectorbt/backtrader), and a Streamlit GUI with Telegram notifications.

### Pros
- High-frequency scalping with >60% winrate target.
- AI/ML enhanced signals for accuracy (~15% boost in sims).
- Risk controls (drawdown pauses, quotas) for capital preservation.
- User-friendly GUI and mobile alerts.
- Backup price fetch from Grok when Binance fails (UI shows yellow highlight).
- Grok recommends additional pairs, monitoring 5x the configured amount for analytics.
- Per-pair settings with DB persistence and AgGrid editing.
- Interactive charts and sidebar controls remain visible on all pages.
- Unit tests via `pytest` ensure core logic like volatility scaling works.

### Cons
- Dependency on APIs (Binance, Grok, Dune, Telegram)—downtime risks.
- Requires setup (API keys, Telethon session).
- Crypto volatility/fees may erode profits.

### High-Level Architecture (HLA)
- **Data Layer**: WebSocket (Binance), on-chain (Dune), Telegram channels.
- **Analysis Layer**: TA-Lib indicators, LSTM ML, Grok AI, signal aggregation.
- **Strategy Layer**: BaseStrategy with arbitrage, grid, MEV.
- **Execution Layer**: CCXT trades, async monitoring.
- **UI/Control Layer**: Streamlit GUI, Telegram bots.
- **Persistence**: SQLite (params/trades), HDF5 (historical), Redis (pub/sub).

### Low-Level Architecture (LLA)
- Utils: binance/ccxt/grok/ml/onchain/telegram_utils.py.
- Strategies: base/arbitrage/grid/mev_strategy.py.
- Core: scalping_bot.py (async loop, GUI), backtest.py, cleanup_hdf5.py.
- Config: config.py (.env loading).

### Documentation Structure
- [User_Guide.md](docs/User_Guide.md): Setup, usage, GUI.
- [Developer_Guide.md](docs/Developer_Guide.md): Code structure, contribution, testing.
- [Ops Guide.md](docs/Ops%20Guide.md): Deployment, monitoring, maintenance.
- [Install Guide.md](docs/Install%20Guide.md): Revised setup with Telethon init.

### Setup
Run `./install.sh` to initialize. See Install Guide.md for details.

### Contribution
Fork repo, create feature branch, PR with tests.
Run `pytest` before submitting a pull request.

### License
MIT
