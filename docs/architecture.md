# Architecture

## Overview

Aether BTC is a GA-optimized crypto daytrading system for BTC/USDT on Binance Futures. It evolves signal weights, leverage selection, and position sizing using a genetic algorithm, then executes trades on 15-minute candles 24/7.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Aether BTC System                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌────────────┐    ┌──────────┐               │
│  │  Binance  │───▶│    Data    │───▶│Indicators│               │
│  │   API     │    │  Pipeline  │    │  Engine  │               │
│  └──────────┘    └────────────┘    └──────────┘               │
│                        │                  │                     │
│                        ▼                  ▼                     │
│                  ┌──────────┐    ┌──────────────┐             │
│                  │PostgreSQL│    │  GA Engine    │             │
│                  │ Database │    │ (Evolution)   │             │
│                  └──────────┘    └──────────────┘             │
│                        │                  │                     │
│                        ▼                  ▼                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐             │
│  │ Telegram  │◀──│ Live Bot  │◀──│  Chromosome   │             │
│  │  Alerts   │    │(Executor)│    │   (Trained)  │             │
│  └──────────┘    └──────────┘    └──────────────┘             │
│                        │                                       │
│                        ▼                                       │
│                  ┌──────────┐                                  │
│                  │  Binance  │                                  │
│                  │ Futures   │                                  │
│                  └──────────┘                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Component Diagram

```
src/aether_btc/
├── core/                    # Foundation layer
│   ├── enums.py             # SignalType, PositionSide, OrderIntent, ExitReason
│   ├── models.py            # Signal, Order, Position, ClosedPosition, EquitySnapshot
│   └── config.py            # Pydantic settings (DB, Binance, trading, backtest, alerts)
│
├── data/                    # Data layer
│   ├── database.py          # SQLAlchemy ORM models, engine setup
│   ├── binance_client.py    # Binance API wrapper (klines, funding rates)
│   └── pipeline.py          # Fetch → store → load orchestration
│
├── signals/                 # Feature engineering
│   └── indicators.py        # 40+ technical indicators (trend, momentum, volatility, volume)
│
├── ga/                      # Genetic algorithm
│   ├── chromosome.py        # 37-gene specification (leverage, allocation, feature weights)
│   ├── fitness.py           # Signal scoring, fitness evaluation
│   └── engine.py            # Population evolution (selection, crossover, mutation)
│
├── backtest/                # Simulation
│   ├── engine.py            # 9-step bar simulation loop (15min bars)
│   └── walk_forward.py      # Rolling train/test window validation
│
├── risk/                    # Risk management
│   └── manager.py           # Leverage calc, position sizing, liquidation buffer, circuit breaker
│
├── portfolio/               # Position management
│   └── manager.py           # Isolated margin, open/close positions, equity tracking
│
├── execution/               # Live trading
│   └── binance_futures.py   # Binance Futures API client (orders, positions, leverage)
│
└── alerts/                  # Notifications
    └── telegram.py          # Trade entry/exit/system alerts via Telegram
```

## Data Flow

### Training Pipeline

```
1. Fetch Historical Data
   Binance API → 15min candles (2019-present) → PostgreSQL
   Binance API → funding rates (every 8h) → PostgreSQL

2. Compute Indicators
   Raw candles → Indicators.add_all() → 40+ features per bar

3. GA Evolution
   Population of 80 chromosomes
   → Evaluate each via backtest on training window
   → Tournament selection → crossover → mutation
   → Repeat for 50 generations
   → Best chromosome saved to DB

4. Walk-Forward Validation
   Rolling 6-month train / 2-month test windows
   → Independent evolution per window
   → Aggregate OOS consistency metrics
```

### Live Trading Cycle (every 15 minutes)

```
1. Fetch latest 15min candle from Binance
2. Compute indicators on rolling window
3. Apply trained chromosome → signal score (sigmoid)
4. If score > cutoff → BUY signal; if score < 1-cutoff → SHORT signal
5. Risk manager: calculate leverage, position size, stop levels
6. Portfolio manager: check margin, max positions, circuit breaker
7. Execute order on Binance Futures (next candle open)
8. Set SL/TP orders
9. Send Telegram alert
```

## Backtest Engine — 9-Step Bar Loop

Each 15-minute bar is processed in this exact order:

1. **Update prices to OPEN** — set current_price on all positions
2. **Check liquidation** — longs checked at bar LOW, shorts at bar HIGH
3. **Check SL/TP intrabar** — trigger stops at exact price levels
4. **Execute pending orders** — orders from previous bar execute at OPEN + slippage
5. **Apply funding rates** — every 32 bars (8 hours)
6. **Update prices to CLOSE** — final price update
7. **Generate signals** — run signal_fn(candles, bar_idx, chromosome)
8. **Convert to orders** — queued for NEXT bar (no look-ahead)
9. **Record equity snapshot** — cash + unrealized PnL

## Key Design Decisions

### No Look-Ahead Bias
Signals generated on bar T produce orders that execute at bar T+1 OPEN. This is enforced by the pending order queue in the backtest engine.

### NET P&L Only
All P&L calculations deduct commissions (0.04% taker on notional), slippage, and funding costs. There is no concept of "gross P&L" in reporting.

### Isolated Margin
Each position has its own margin. Liquidation of one position does not affect others. This matches Binance Futures isolated margin mode.

### GA-Evolved Leverage
Leverage is not a fixed parameter — it's computed per trade by the risk manager using GA-evolved genes (base leverage * confidence scaling * volatility dampening), capped at the chromosome's max_leverage_cap.

## Database Schema

Five main tables:

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `candles_15m` | Historical price data | timestamp, OHLCV, quote_volume |
| `funding_rates` | Binance funding rates | timestamp, pair, rate |
| `ga_training_progress` | Evolution tracking | run_id, generation, fitness metrics |
| `models` | Trained chromosomes | chromosome JSONB, fitness, Sharpe, Calmar |
| `trades` | Live trade log | pair, side, leverage, PnL, costs |

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| Data | pandas, numpy |
| Technical Analysis | ta |
| Exchange API | python-binance |
| Database | PostgreSQL + SQLAlchemy |
| Config | Pydantic + python-dotenv |
| Logging | structlog |
| Alerts | python-telegram-bot |
| Testing | pytest + hypothesis |
