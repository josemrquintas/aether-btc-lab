# Aether BTC

GA-optimized crypto daytrading system for BTC/USDT on Binance Futures.

## Overview

Aether BTC uses a **genetic algorithm** to evolve trading signal weights, leverage selection, and position sizing parameters. It trades BTC/USDT perpetual futures on 15-minute candles, 24/7.

### Key Features

- **GA-Evolved Signals**: 26 technical indicator weights + bias evolved via genetic algorithm
- **Dynamic Leverage**: 1x-20x leverage per trade, optimized by GA based on confidence and volatility
- **Full Cost Simulation**: Commission (0.04%), slippage, and funding rate costs
- **Liquidation Modeling**: Accurate liquidation price calculation with configurable safety buffer
- **Walk-Forward Validation**: 6-month train / 2-month test rolling windows
- **Risk Management**: Position sizing, circuit breaker, max drawdown limits

## Quick Start

```bash
# Clone and install
git clone https://github.com/josemrquintas/aether-btc.git
cd aether-btc
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your database and API credentials

# Run tests
pytest tests/

# Fetch historical data
python scripts/fetch_historical.py --pair BTCUSDT --start 2019-09-01

# Compute indicators
python scripts/compute_indicators.py --pair BTCUSDT

# Train GA
python scripts/train_ga.py --population 80 --generations 50

# Run backtest with trained chromosome
python scripts/run_backtest.py --chromosome models/best_chromosome.json
```

## Architecture

```
src/aether_btc/
  core/           → Enums, models, config
  signals/        → 40+ technical indicators
  ga/             → Genetic algorithm (chromosome, fitness, engine)
  backtest/       → 15min bar simulation with leverage + liquidation
  risk/           → Position sizing, leverage calc, circuit breaker
  portfolio/      → Isolated margin position management
  data/           → Binance data fetcher + PostgreSQL storage
  execution/      → Binance Futures API client
  alerts/         → Telegram notifications
```

See [docs/architecture.md](docs/architecture.md) for the full system design.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Initial Capital | $1,000 | Starting account balance |
| Max Leverage | 20x | Hard cap on leverage |
| Commission | 0.04% | Binance Futures taker fee |
| Max Positions | 3 | Concurrent open positions |
| Daily Loss Limit | 10% | Circuit breaker threshold |
| Risk Per Trade | 2% | Max account risk per position |

## Documentation

- [Architecture](docs/architecture.md) — System design and data flow
- [Chromosome](docs/chromosome.md) — 37-gene specification and GA operators
- [Backtest Guide](docs/backtest-guide.md) — Running and interpreting backtests
- [Data Pipeline](docs/data-pipeline.md) — Fetching data and computing indicators
- [Deployment](docs/deployment.md) — VPS setup and live trading

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/unit/test_core_models.py -v
pytest tests/unit/test_backtest_engine.py -v
pytest tests/unit/test_ga_fitness.py -v
```

83 unit tests covering: core models, backtest engine, GA chromosome/fitness/engine, risk manager, portfolio manager, indicators, and data pipeline.

## Tech Stack

- **Python 3.12** with pandas, numpy, ta
- **Binance API** via python-binance
- **PostgreSQL** via SQLAlchemy
- **Pydantic** for configuration
- **structlog** for structured logging
- **pytest** for testing

## License

Private repository — all rights reserved.
