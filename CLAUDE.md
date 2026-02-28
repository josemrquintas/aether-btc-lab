# Aether BTC — Project Rules

## Project Structure
src/aether_btc/          -> Core Python package
  core/                  -> Enums, models, config
  signals/               -> Indicators + 5 signal strategies (momentum, mean reversion, trend, volatility, funding/volume)
  ga/                    -> GA engine + 52-gene chromosome (leverage, features, strategy weights)
  backtest/              -> 15min bar simulation with leverage, liquidation, funding rates
  risk/                  -> Crypto risk manager (liquidation buffer, max leverage, position sizing)
  data/                  -> Binance historical data fetcher + storage
  execution/             -> Binance Futures API live trading client
  alerts/                -> Telegram notifications
  portfolio/             -> Position management with leverage margin
scripts/                 -> CLI scripts (fetch_historical, compute_indicators, train_ga, etc.)
tests/                   -> pytest test suite (unit + integration)
docs/                    -> Architecture, chromosome, backtest guide, data pipeline, deployment

## Tech Stack
- Python 3.12, pandas, numpy, ta (technical analysis)
- python-binance (exchange API)
- SQLAlchemy + PostgreSQL (database: aether_btc)
- Pydantic (config), structlog (logging)
- Genetic Algorithm for signal optimization

## Git & Branch Workflow
- Work on `develop` branch (feature branches for large changes)
- Branch naming: `feat/US-XXX-description`
- PRs target `develop`
- Commit messages: `feat:`, `fix:`, `docs:`, `chore:`
- GitHub: josemrquintas/aether-btc

## Architecture: GA-Evolved Crypto Daytrader
- Single pair: BTC/USDT on Binance Futures
- Timeframe: 15-minute candles (96 per day, 24/7)
- GA evolves feature weights (26 raw + 14 strategy) + leverage genes + allocation genes
- 5 signal strategies precomputed before GA: Momentum, MeanReversion, TrendFollowing, VolatilityBreakout, FundingVolume
- Signal scoring: sigmoid(dot(40 features, 40 weights) + bias)
- Leverage: 1x-20x, GA-optimized per trade based on confidence + volatility

## GA Fitness (Sharpe + Calmar)
fitness = 0.6 * min(sharpe, 5.0) + 0.4 * min(calmar, 10.0)
Calmar = annualized_return / max(max_drawdown, 5%). Max possible fitness = 7.0.

## Key Formulas
- Position notional = quantity * price * leverage
- PnL = (exit_price - entry_price) / entry_price * leverage * margin_used
- Liquidation price (long) = entry_price * (1 - 1/leverage + maintenance_margin)
- Liquidation price (short) = entry_price * (1 + 1/leverage - maintenance_margin)
- Funding cost = position_notional * funding_rate (every 8h)
- Commission = notional_value * 0.04% (taker fee)

## Key Patterns
- Orders execute at NEXT bar OPEN (no look-ahead bias)
- All P&L is NET of commissions, slippage, and funding costs
- Funding rates applied every 32 bars (8h / 15min)
- Liquidation check every bar
- Circuit breaker: halt trading at 10% drawdown from peak
- Walk-forward: 6-month train, 2-month test windows

## Testing
- Run: `pytest tests/` from project root
- DB tests need AETHER_BTC_DB_URL env var

## Do NOT
- Push directly to main (merge develop -> main with squash)
- Hardcode DB credentials or API keys in code (use .env)
- Skip look-ahead bias prevention (always next-bar execution)
- Use gross P&L anywhere (always NET)
- Trade without liquidation buffer check
