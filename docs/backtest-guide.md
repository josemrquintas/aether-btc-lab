# Backtest Guide

## Running a Backtest

### Prerequisites
1. Historical candles loaded in PostgreSQL (`candles_15m` table)
2. Funding rates loaded (`funding_rates` table)
3. Indicators computed on the candle data
4. A chromosome (trained model or default)

### Quick Start

```bash
# With a trained chromosome from JSON
python scripts/run_backtest.py --chromosome models/best_chromosome.json

# With a model from the database
python scripts/run_backtest.py --model-id 1

# Custom date range and capital
python scripts/run_backtest.py --chromosome models/best.json --start 2024-01-01 --end 2024-06-30 --capital 5000
```

### Using the Backtest Engine Programmatically

```python
from aether_btc.backtest.engine import BacktestEngine, BacktestConfig
from aether_btc.ga.chromosome import Chromosome
from aether_btc.ga.fitness import generate_signal_with_exits

# Load data
candles = ...  # DataFrame with OHLCV + indicators
funding_rates = ...  # Series indexed by timestamp

# Load chromosome
chromosome = Chromosome.from_json("models/best_chromosome.json")

# Configure
config = BacktestConfig(
    initial_capital=1000.0,
    commission_rate=0.0004,  # 0.04% taker
    slippage_bps=2.0,
    max_leverage=20.0,
    max_positions=3,
    funding_interval_bars=32,  # 8h / 15min
    daily_loss_limit=0.10,
)

# Run
engine = BacktestEngine(config)
result = engine.run(
    candles=candles,
    funding_rates=funding_rates,
    signal_fn=generate_signal_with_exits,
    chromosome=chromosome,
)

# Inspect results
print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Calmar Ratio: {result.calmar_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")
print(f"Total Trades: {result.total_trades}")
print(f"Win Rate: {result.win_rate:.2%}")
```

## Interpreting Results

### Key Metrics

| Metric | Description | Good Target |
|--------|-------------|-------------|
| **Sharpe Ratio** | Risk-adjusted return (annualized) | > 1.5 |
| **Calmar Ratio** | Return / max drawdown | > 2.0 |
| **Max Drawdown** | Largest peak-to-trough decline | < 20% |
| **Win Rate** | % of profitable trades | > 45% |
| **Profit Factor** | Gross profit / gross loss | > 1.5 |
| **Avg Leverage** | Mean leverage across trades | 3x-8x |
| **Total Trades** | Number of round-trip trades | > 50 (for statistical significance) |

### Annualization

15-minute bars have 35,040 bars per year (96 bars/day * 365 days). The annualization factor is `sqrt(35040)` for Sharpe ratio calculations.

### Cost Breakdown

Every trade incurs:
- **Commission**: 0.04% of notional value (entry + exit)
- **Slippage**: configurable (default 2 bps per side)
- **Funding**: varies, applied every 8 hours to open positions

Example for a $1,000 account, 10x leverage, $10,000 notional:
- Commission per side: $10,000 * 0.04% = $4.00
- Round-trip commission: $8.00
- That's 0.8% of margin — significant with leverage!

## Common Issues

### No Trades Generated
- Check `min_confidence_cutoff` — if too high (>0.8), few signals pass
- Check that indicators are computed (no NaN in feature columns)
- Ensure warmup period is sufficient (first 200 bars are skipped)

### Excessive Drawdown
- Lower `max_leverage_cap` gene
- Increase `liquidation_buffer_pct`
- Tighten `atr_stop_multiplier` (smaller SL)
- The circuit breaker triggers at 10% drawdown from peak

### Liquidation Events
- Liquidation means 100% loss of position margin
- Check `liquidation_buffer_pct` — should be > 0.10
- Lower leverage reduces liquidation risk exponentially

### Look-Ahead Bias Check
- Orders ALWAYS execute at next bar OPEN
- Signals generated from bar T's CLOSE price
- If results seem too good, verify with `validate_no_lookahead()`

## Walk-Forward Validation

```bash
# Run walk-forward with default settings (6mo train, 2mo test)
python scripts/train_ga.py --population 80 --generations 50
```

The walk-forward runner:
1. Splits data into rolling windows (180-day train, 60-day test)
2. Trains a GA independently on each training window
3. Tests the best chromosome on the subsequent test window
4. Reports OOS (out-of-sample) consistency

**Target**: >50% of test windows should have positive returns.
