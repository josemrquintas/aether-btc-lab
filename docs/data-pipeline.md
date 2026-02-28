# Data Pipeline

## Overview

The data pipeline fetches historical BTC/USDT data from Binance, stores it in PostgreSQL, and computes technical indicators for use by the GA and backtest engine.

## Data Sources

### 15-Minute Candles
- **Source**: Binance Spot/Futures API (`get_historical_klines`)
- **Pair**: BTCUSDT
- **Interval**: 15 minutes
- **Range**: 2019-09-01 to present (~6.5 years)
- **Volume**: ~630,000 candles (96 per day * 365 * 6.5)
- **Endpoint**: Public (no API key required for historical data)

### Funding Rates
- **Source**: Binance Futures API (`futures_funding_rate`)
- **Frequency**: Every 8 hours (00:00, 08:00, 16:00 UTC)
- **Range**: 2019-09-01 to present
- **Volume**: ~7,100 records (3 per day * 365 * 6.5)
- **Endpoint**: Public

## Database Schema

### candles_15m

```sql
CREATE TABLE candles_15m (
    id SERIAL PRIMARY KEY,
    pair VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(20,8),
    high DECIMAL(20,8),
    low DECIMAL(20,8),
    close DECIMAL(20,8),
    volume DECIMAL(20,8),
    quote_volume DECIMAL(20,8),
    trades_count INTEGER,
    UNIQUE(pair, timestamp)
);
```

### funding_rates

```sql
CREATE TABLE funding_rates (
    id SERIAL PRIMARY KEY,
    pair VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    funding_rate DECIMAL(12,8),
    UNIQUE(pair, timestamp)
);
```

## Fetching Data

### CLI Usage

```bash
# Fetch everything from 2019-09-01 to now
python scripts/fetch_historical.py --pair BTCUSDT --interval 15m --start 2019-09-01

# Skip candles if already fetched, only get funding rates
python scripts/fetch_historical.py --skip-candles

# Specify end date
python scripts/fetch_historical.py --end 2024-12-31
```

### Pagination

Binance limits responses to 1,000 records per request. The pipeline handles this automatically:

- **Candles**: Uses `python-binance`'s built-in pagination in `get_historical_klines`
- **Funding rates**: Manual pagination using `startTime` parameter, fetching 1,000 records at a time and advancing the start timestamp

### Rate Limiting

The Binance API has rate limits. The pipeline uses `tenacity` for retry with exponential backoff:
- Max 3 retries per request
- 1-second base wait between retries

## Computing Indicators

### CLI Usage

```bash
python scripts/compute_indicators.py --pair BTCUSDT
```

### Indicator Categories

| Category | Indicators | Count |
|----------|-----------|-------|
| **Trend** | SMA(20,50,200), EMA(9,21), MACD(12,26,9), Supertrend, ADX | 5 |
| **Momentum** | RSI(14), Stochastic(14,3,3), ROC(12), CCI(20) | 4 |
| **Volatility** | Bollinger Bands(20,2), ATR(14), Z-score(20) | 3 |
| **Volume** | OBV slope, CMF(20), MFI(14), volume ratio | 4 |
| **Statistical** | Returns (1/4/16/96 bar), momentum acceleration | 5 |
| **Crypto** | Funding rate, quote volume ratio, intraday range, BB position, price vs SMA | 6 |

### Warmup Period

Indicators require a warmup period before producing valid values. The first ~200 bars of computed data will contain NaN values. The backtest engine skips these bars.

| Indicator | Warmup Bars | Warmup Time |
|-----------|-------------|-------------|
| RSI(14) | 13 | ~3.25 hours |
| SMA(20) | 19 | ~5 hours |
| SMA(200) | 199 | ~50 hours |
| MACD(26) | 33 | ~8.25 hours |
| Bollinger(20) | 19 | ~5 hours |

### Data Validation

After computing indicators, the pipeline validates:
1. No unexpected NaN values after the warmup period
2. RSI is bounded [0, 100]
3. Bollinger Band ordering: lower < middle < upper
4. No look-ahead bias (future data doesn't affect past indicators)

## Programmatic Usage

```python
from aether_btc.data.pipeline import DataPipeline
from aether_btc.signals.indicators import Indicators

# Initialize pipeline
pipeline = DataPipeline(db_url="postgresql://...")

# Fetch and store
pipeline.fetch_and_store_candles("BTCUSDT", "2019-09-01")
pipeline.fetch_and_store_funding_rates("BTCUSDT", "2019-09-01")

# Load data
candles = pipeline.load_candles("BTCUSDT")
funding = pipeline.load_funding_rates("BTCUSDT")

# Compute indicators
candles = Indicators.add_all(candles)

# Merge funding rate as feature
candles = Indicators.add_crypto_features(candles, funding_rates=funding)
```

## Data Quality Checks

Run via unit tests:

```bash
pytest tests/unit/test_data_pipeline.py -v
```

Tests verify:
- 96 candles per day (24/7 market)
- No gaps in timestamps
- OHLC validity: low <= open,close <= high
- Positive volume
- 3 funding rates per day at 8-hour intervals
- Funding rates within ±0.5% bounds
