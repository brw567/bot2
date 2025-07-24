# Developer Guide

## Code Structure
- Root: scalping_bot.py (main), backtest.py, cleanup_hdf5.py, config.py, install.sh, requirements.txt.
- utils/: binance_utils.py, ccxt_utils.py, grok_utils.py, ml_utils.py, onchain_utils.py, telegram_utils.py.
- strategies/: base_strategy.py, arbitrage_strategy.py, grid_strategy.py, mev_strategy.py.

## Contribution
1. Fork repo, create branch (`git checkout -b feature/x`).
2. Develop: Use venv, test with `backtest.py`.
3. Test: Unit (e.g., spread calc), functional (async loop), acceptance (GUI signals).
4. PR: Include changes, tests.

## Key Components
- Signals: `get_signal_score` in scalping_bot.py (weighted formula).
- ML: LSTM in ml_utils.py (train/val split, overfitting check).
- Backtest: Modes in backtest.py (vectorbt simple, backtrader advanced, ML).
- Strategies: Inherit BaseStrategy for risk (Kelly, Grok SL/TP).
- GUI: Streamlit tabs, aggrid tables, backtest toggle.

## Debugging
- Logs: bot.log (signals, errors).
- Circular Fix: Dynamic imports (e.g., in ccxt_utils.py).
- Cleanup: HDF5 script with h5py validation.
