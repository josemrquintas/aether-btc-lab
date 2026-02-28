# Deployment Guide

## Prerequisites

- VPS with Python 3.12+
- PostgreSQL 15+
- Binance Futures account (with Futures trading enabled)
- Telegram bot token (for alerts)

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/josemrquintas/aether-btc.git
cd aether-btc
```

### 2. Create Virtual Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

Required environment variables:

| Variable | Description |
|----------|-------------|
| `BINANCE_API_KEY` | Binance API key |
| `BINANCE_API_SECRET` | Binance API secret |
| `AETHER_BTC_DB_URL` | PostgreSQL connection string |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token |
| `TELEGRAM_CHAT_ID` | Telegram chat ID for alerts |

### 4. Initialize Database

```bash
python -c "from aether_btc.data.database import init_db; init_db()"
```

## Data Pipeline

### Fetch Historical Data

```bash
# Full historical fetch (takes several hours)
python scripts/fetch_historical.py --pair BTCUSDT --start 2019-09-01

# Compute indicators
python scripts/compute_indicators.py --pair BTCUSDT
```

### GA Training

```bash
# Train with default parameters
python scripts/train_ga.py --population 80 --generations 50 --workers 4

# Best chromosome saved to models/ directory
```

## Live Trading

### Paper Trading (Testnet)

```bash
# Run single cycle (no execution)
python scripts/live_trade.py --chromosome models/best_chromosome.json --paper --once

# Run continuously on testnet
python scripts/live_trade.py --chromosome models/best_chromosome.json --paper
```

### Production

```bash
# Live trading
python scripts/live_trade.py --chromosome models/best_chromosome.json
```

### Systemd Service

Create `/etc/systemd/system/aether-btc.service`:

```ini
[Unit]
Description=Aether BTC Live Trading Bot
After=network.target postgresql.service

[Service]
Type=simple
User=josequintas
WorkingDirectory=/home/josequintas/projects/aether-btc
Environment="PATH=/home/josequintas/projects/aether-btc/.venv/bin"
EnvironmentFile=/home/josequintas/projects/aether-btc/.env
ExecStart=/home/josequintas/projects/aether-btc/.venv/bin/python scripts/live_trade.py --chromosome models/best_chromosome.json
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable aether-btc
sudo systemctl start aether-btc
sudo systemctl status aether-btc
```

### Monitoring

```bash
# Check service status
sudo systemctl status aether-btc

# View logs
sudo journalctl -u aether-btc -f

# Check recent trades in DB
psql aether_btc -c "SELECT * FROM trades ORDER BY created_at DESC LIMIT 10;"
```

## Emergency Procedures

### Stop Trading Immediately

```bash
sudo systemctl stop aether-btc
```

### Close All Positions

Via Binance web interface or API:
1. Log into Binance Futures
2. Close all open positions
3. Cancel all open orders

### Database Backup

```bash
pg_dump aether_btc > backup_$(date +%Y%m%d).sql
```

## Security

- Never commit `.env` to git
- Use read-only API keys where possible
- Enable IP whitelist on Binance API keys
- Set withdrawal disabled on API keys
- Monitor Telegram alerts for unexpected behavior
