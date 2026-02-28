# Chromosome Specification

## Overview

The chromosome encodes all parameters that the GA evolves. Each "gene" has a name, min/max bounds, default value, and type. The GA explores the gene space to find parameter combinations that maximize trading fitness (Sharpe + Calmar ratio).

## Gene Categories

### Leverage Genes (4 genes)

Control how much leverage is applied per trade. The risk manager computes effective leverage as:

```
leverage = base * (1 + confidence_scale * (confidence - 0.5)) * (1 - volatility_dampen * normalized_atr)
leverage = clamp(leverage, 1.0, min(max_leverage_cap, config.max_leverage))
```

| Gene | Min | Max | Default | Description |
|------|-----|-----|---------|-------------|
| `leverage_base` | 1.0 | 10.0 | 3.0 | Base leverage before adjustments |
| `leverage_confidence_scale` | 0.0 | 2.0 | 1.0 | How much signal confidence increases leverage |
| `leverage_volatility_dampen` | 0.0 | 1.0 | 0.5 | How much high volatility reduces leverage |
| `max_leverage_cap` | 5.0 | 20.0 | 10.0 | Hard cap on leverage per trade |

### Allocation Genes (5 genes)

Control position sizing, trade filtering, and hold duration.

| Gene | Min | Max | Default | Description |
|------|-----|-----|---------|-------------|
| `funding_rate_threshold` | -0.001 | 0.001 | 0.0 | Skip trade if funding rate exceeds this (adverse direction) |
| `liquidation_buffer_pct` | 0.05 | 0.30 | 0.15 | Min distance from entry to liquidation price |
| `min_confidence_cutoff` | 0.30 | 0.80 | 0.55 | Minimum signal score to open a position |
| `position_size_multiplier` | 0.5 | 3.0 | 1.0 | Scale the ATR-based position size |
| `min_hold_bars` | 1 | 32 | 4 | Minimum bars to hold before allowing exit signal |

### Stop / Target Genes (2 genes)

| Gene | Min | Max | Default | Description |
|------|-----|-----|---------|-------------|
| `atr_stop_multiplier` | 1.0 | 5.0 | 2.0 | SL distance = ATR * multiplier |
| `risk_reward_ratio` | 1.0 | 4.0 | 2.0 | TP distance = SL distance * ratio |

### Feature Weight Genes (26 genes)

Each gene is a weight in the signal scoring function. The signal score is computed as:

```
score = sigmoid(sum(feature_i * weight_i) + bias)
```

Where `sigmoid(x) = 1 / (1 + exp(-x))`. Score > cutoff → BUY, score < (1 - cutoff) → SHORT.

| Gene | Min | Max | Default | Feature Column |
|------|-----|-----|---------|----------------|
| `fw_return_1bar` | -2.0 | 2.0 | 0.0 | `return_1bar` |
| `fw_return_4bar` | -2.0 | 2.0 | 0.0 | `return_4bar` |
| `fw_return_16bar` | -2.0 | 2.0 | 0.0 | `return_16bar` |
| `fw_return_96bar` | -2.0 | 2.0 | 0.0 | `return_96bar` |
| `fw_momentum_accel` | -2.0 | 2.0 | 0.0 | `momentum_acceleration` |
| `fw_bb_position` | -2.0 | 2.0 | 0.0 | `bb_position` |
| `fw_price_vs_sma50` | -2.0 | 2.0 | 0.0 | `price_vs_sma50` |
| `fw_price_vs_sma200` | -2.0 | 2.0 | 0.0 | `price_vs_sma200` |
| `fw_zscore` | -2.0 | 2.0 | 0.0 | `zscore_20d` |
| `fw_atr_norm` | -2.0 | 2.0 | 0.0 | `atr_normalized` |
| `fw_volatility_ratio` | -2.0 | 2.0 | 0.0 | `volatility_ratio` |
| `fw_bb_width` | -2.0 | 2.0 | 0.0 | `bb_width` |
| `fw_intraday_range` | -2.0 | 2.0 | 0.0 | `intraday_range` |
| `fw_volume_ratio` | -2.0 | 2.0 | 0.0 | `volume_ratio` |
| `fw_cmf` | -2.0 | 2.0 | 0.0 | `cmf` |
| `fw_mfi` | -2.0 | 2.0 | 0.0 | `mfi` |
| `fw_obv_slope` | -2.0 | 2.0 | 0.0 | `obv_slope` |
| `fw_rsi` | -2.0 | 2.0 | 0.0 | `rsi` |
| `fw_macd_hist` | -2.0 | 2.0 | 0.0 | `macd_histogram` |
| `fw_adx` | -2.0 | 2.0 | 0.0 | `adx` |
| `fw_roc` | -2.0 | 2.0 | 0.0 | `roc` |
| `fw_stoch_k` | -2.0 | 2.0 | 0.0 | `stoch_k` |
| `fw_supertrend` | -2.0 | 2.0 | 0.0 | `supertrend_direction` |
| `fw_cci` | -2.0 | 2.0 | 0.0 | `cci` |
| `fw_funding_rate` | -2.0 | 2.0 | 0.0 | `funding_rate` |
| `fw_quote_vol_ratio` | -2.0 | 2.0 | 0.0 | `quote_volume_ratio` |

### Bias Gene (1 gene)

| Gene | Min | Max | Default | Description |
|------|-----|-----|---------|-------------|
| `fw_bias` | -5.0 | 5.0 | 0.0 | Constant offset in signal scoring (positive = bullish bias) |

## Total: 37 Genes

## GA Operators

### Initialization
- `Chromosome.random()` — each gene drawn uniformly from [min, max]
- `Chromosome.from_defaults()` — all genes set to default values

### Mutation
- Gaussian mutation: `gene += N(0, std) * (max - min)`
- Mutation rate: 10% of genes per mutation
- Result clamped to [min, max]

### Crossover
- Uniform crossover: each gene randomly taken from parent A or B (50/50)
- Produces two children

### Selection
- Tournament selection (size 3): pick 3 random, best wins
- Elitism: top 5 chromosomes survive unchanged

## Fitness Function

```
fitness = 0.6 * min(sharpe, 5.0) + 0.4 * min(calmar, 10.0) - penalties
```

Penalties:
- No trades: fitness = 0
- Liquidation: -0.5 per liquidation event
- L2 regularization: 0.01 * sum(weight^2) on feature weights

Maximum possible fitness = 0.6 * 5.0 + 0.4 * 10.0 = 7.0

## Example: Default Chromosome Behavior

With all weights at 0 and bias at 0:
- Signal score = sigmoid(0) = 0.5
- With default cutoff of 0.55: score (0.5) < cutoff → no trade
- This is the "null hypothesis" — the GA must evolve weights away from 0 to generate any trades
