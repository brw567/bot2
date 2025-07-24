# File Structure.md
This document outlines the file structure of the Ultimate Crypto Scalping Bot project. It follows a modular design with directories for utilities, strategies, and data persistence. Runtime-generated files (e.g., in data/hdf5/) are noted but not included in the static structure.

```
.
├── .env.example              # Example environment file for configuration
├── backtest.py               # Backtesting scripts with vectorbt, backtrader, and ML integration
├── cleanup_hdf5.py           # Script to clean up old HDF5 files for storage management
├── config.py                 # Central configuration loading from .env
├── data                      # Directory for runtime data
│   └── hdf5                  # Subdirectory for historical HDF5 files (generated at runtime)
├── docs                      # Directory for project documentation
│   ├── Developer_Guide.md    # Developer guide with code structure and contribution
│   ├── Install Guide.md      # Extended installation guide for Ubuntu
│   ├── Ops Guide.md          # Operations guide for deployment and maintenance
│   ├── README.md             # Project overview and summary
│   ├── User_Guide.md         # User guide for setup and usage
│   └── File Structure.md     # Structure of the files
├── install.sh                # Setup script for dependencies, DB init, and Telethon session
├── requirements.txt          # List of Python dependencies with versions
├── scalping_bot.py           # Main bot script with async loop and Streamlit GUI
├── strategies                # Directory for trading strategies
│   ├── arbitrage_strategy.py # Arbitrage strategy implementation
│   ├── base_strategy.py      # Base class for strategies with risk management
│   ├── grid_strategy.py      # Grid trading strategy with VWAP centering
│   └── mev_strategy.py       # MEV detection strategy
└── utils                     # Directory for utility modules
    ├── binance_utils.py      # Binance-specific utilities (client, trades)
    ├── ccxt_utils.py         # CCXT general utilities (spread, logging)
    ├── grok_utils.py         # Grok API utilities for sentiment and risk
    ├── ml_utils.py           # ML utilities for LSTM predictions and data fetch
    ├── onchain_utils.py      # On-chain utilities (Dune API for RPL)
    └── telegram_utils.py     # Telegram utilities for notifications and message fetch
```

## Notes
 - **Root Files:** Entry points (scalping_bot.py), configs (config.py, .env.example), and scripts (install.sh, cleanup_hdf5.py) for quick access.
 - **utils/:** Reusable helpers for integrations (e.g., ccxt_utils.py for exchange ops).
 - **strategies/:** Modular trading logic, inheriting base_strategy.py.
 - **data/hdf5/:** Runtime-generated for historical data; managed by cleanup_hdf5.py.
 - **docs/:** Added for the full documentation package (README.md, guides).
 - No `__init__.py` files needed unless packaging; dynamic imports handle circulars.


