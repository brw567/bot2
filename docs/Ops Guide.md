# Ops Guide

## Deployment
- Docker: Build with Dockerfile (add: FROM python:3.12, COPY ., RUN pip install -r requirements.txt, CMD ["streamlit", "run", "scalping_bot.py"]).
- Prod: Set .env securely, run as service (systemd/Supervisor).

## Monitoring
- Logs: bot.log (trades, errors, signals).
- Quotas: BINANCE_WEIGHT_LIMIT monitoring in scalping_bot.py (pause >80%).
- Winrate/ML: Redis pub/sub updates in scalping_bot.py.
- MEV/RPL: Pauses logged, Telegram alerts.

## Maintenance
- Cleanup: Schedule `python cleanup_hdf5.py` daily (cron: 0 0 * * * python /path/cleanup_hdf5.py).
- Retrain ML: Periodic in scalping_bot.py or manual via ml_utils.train_model.
- Updates: Pip install -r requirements.txt; check CCXT/Dune API changes.
- Scaling: Redis for multi-instance; Docker for cloud deployment.

## Troubleshooting
- API Rate Limits: Increase pauses in monitoring_loop.
- DB Issues: Re-init with install.sh.
- Telegram: Verify session in .env.
