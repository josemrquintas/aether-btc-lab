# Signal Strategies

## Overview

Aether BTC uses 5 signal strategies that encode domain knowledge about different market regimes. Each strategy independently analyzes indicators and produces a directional signal (BUY/SHORT/HOLD) with a confidence score. The GA learns to weight these strategy outputs alongside raw indicator features.

## Architecture

```
Raw indicators (26 features) ────────────────────────────┐
5 Strategies (14 features: presence + confidence) ───────┤→ sigmoid(40 weights) → BUY/SHORT
                                                         │
  The GA learns to combine BOTH raw features AND         │
  strategy signals. It can ignore strategies entirely     │
  (set weights to 0) or rely on them heavily.            │
```

### ABC Contract

Every strategy implements `SignalStrategy`:
- `name` — unique identifier (e.g., "momentum")
- `version` — semver string
- `min_confirmations` — minimum sub-signals needed to fire
- `generate_signal(df, bar_idx) -> StrategySignal` — evaluate a single bar
- `get_required_columns() -> list[str]` — indicator columns needed

### StrategySignal Output

```python
@dataclass
class StrategySignal:
    signal_type: SignalType   # BUY, SHORT, or HOLD
    confidence: float         # [0, 1] — sub-indicator agreement
    confirmations: int        # sub-indicators that fired
    total_checks: int         # total sub-indicators evaluated
    metadata: dict            # per-indicator details for debugging
```

### StrategyRunner

`StrategyRunner.precompute_all_bars(df)` runs all 5 strategies on every bar **once**, before GA evolution. This adds 14 columns:
- 10 per-strategy columns: `sig_{name}` (-1/0/1) + `sig_{name}_conf` (0-1)
- 4 aggregate columns: `n_buy_signals`, `n_sell_signals`, `signal_consensus`, `max_confidence`

Zero performance cost during GA evolution — strategies are precomputed.

## The 5 Strategies

### 1. MomentumStrategy

Detects when multiple momentum indicators simultaneously confirm directional conviction.

| Sub-signal | Bullish | Bearish | Weight |
|-----------|---------|---------|--------|
| RSI reversal | RSI < 30 and rising | RSI > 70 and falling | 0.8 |
| MACD crossover | Histogram flips positive | Histogram flips negative | 0.7 |
| Stochastic crossover | %K crosses above %D when %K < 20 | %K crosses below %D when %K > 80 | 0.6 |
| ROC direction | ROC > 0 and increasing | ROC < 0 and decreasing | 0.5 |
| CCI reversal | CCI crosses above -100 | CCI crosses below +100 | 0.6 |

Min confirmations: 2. Confidence = sum(firing_weights) / sum(all_weights).

### 2. MeanReversionStrategy

Detects price extremes likely to revert to the mean.

| Sub-signal | Bullish | Bearish | Weight |
|-----------|---------|---------|--------|
| BB extreme | bb_position <= 0.0 | bb_position >= 1.0 | 0.8 |
| Z-score extreme | zscore <= -2.0 | zscore >= +2.0 | 0.9 (scaled) |
| RSI extreme | RSI < 25 | RSI > 75 | 0.7 |

Min confirmations: 1. Z-score weight is proportional to extremity: `0.9 * min(abs(z)/3, 1)`.

### 3. TrendFollowingStrategy

Only trades in the direction of the dominant trend. **ADX gate**: no signals when ADX < 20.

| Sub-signal | Bullish | Bearish | Weight |
|-----------|---------|---------|--------|
| EMA crossover | EMA(9) crosses above EMA(21) | EMA(9) crosses below EMA(21) | 0.8 |
| SMA200 trend | Close > SMA(200) | Close < SMA(200) | 0.6 |
| Supertrend flip | Direction -1 → +1 | Direction +1 → -1 | 0.9 |
| MACD position | MACD > 0 | MACD < 0 | 0.5 |

Min confirmations: 2. ADX bonus: `min((adx - 20) / 30, 0.3)` for strong trends.

### 4. VolatilityBreakoutStrategy

Detects volatility expansion after compression.

| Sub-signal | Bullish | Bearish | Weight |
|-----------|---------|---------|--------|
| Keltner break | Close > kc_upper | Close < kc_lower | 0.8 |
| Donchian break | Close >= dc_upper | Close <= dc_lower | 0.7 |
| BB squeeze release | Width expanding + close > middle | Width expanding + close < middle | 0.9 |
| Strong bar | Range > 1.5*ATR, close in top 25% | Range > 1.5*ATR, close in bottom 25% | 0.6 |

Min confirmations: 2. Confidence = min(firing_weight / 2.0, 1.0).

### 5. FundingVolumeStrategy (crypto-specific)

Combines funding rate (contrarian) with volume analysis. No equities equivalent.

| Sub-signal | Bullish | Bearish | Weight |
|-----------|---------|---------|--------|
| Funding contrarian | rate < -0.05% (shorts crowded) | rate > +0.05% (longs crowded) | 0.8 |
| Volume spike | vol_ratio > 2 + bullish close | vol_ratio > 2 + bearish close | 0.7 |
| OBV trend | New 20-bar OBV high | New 20-bar OBV low | 0.7 |
| MFI reversal | MFI crosses above 20 | MFI crosses below 80 | 0.6 |
| CMF direction | CMF > +0.05 | CMF < -0.05 | 0.5 |

Min confirmations: 2. Funding is CONTRARIAN: high positive funding = bearish.

## Signal Flow

```
1. Load candles from DB
2. Indicators.add_all(candles)           → 40+ indicator columns
3. StrategyRunner.precompute_all_bars()  → 14 strategy feature columns
4. GA evolves chromosomes:
   For each chromosome, for each bar:
     score = sigmoid(dot(40 features, 40 weights) + bias)
     if score > cutoff → BUY; if score < 1-cutoff → SHORT
5. Backtest processes orders at next-bar OPEN
```

## Adding a New Strategy

1. Create `src/aether_btc/signals/new_strategy.py` implementing `SignalStrategy`
2. Add to `STRATEGIES` list in `signals/__init__.py`
3. Add 2 genes to `chromosome.py`: `fw_sig_{name}` and `fw_sig_{name}_conf`
4. Add both to `FEATURE_COLUMN_MAP`
5. Update tests in `test_strategies.py`
