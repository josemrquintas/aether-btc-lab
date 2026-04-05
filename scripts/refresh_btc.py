#!/usr/bin/env python3
"""BTC refresh pipeline — multi-timeframe.

Fetches the latest candle(s) from Binance, upserts into DB, computes
indicators + strategy signals, loads the best GA model, generates a
live signal, and stores everything in the appropriate timeframe tables.

Usage:
    python scripts/refresh_btc.py                          # default: 15m
    python scripts/refresh_btc.py --interval 1h
    python scripts/refresh_btc.py --interval 4h --bars 3
    python scripts/refresh_btc.py --interval 1d
    python scripts/refresh_btc.py --all                    # refresh all timeframes

Cron examples:
    */15 * * * *  cd /path && .venv/bin/python scripts/refresh_btc.py >> logs/refresh.log 2>&1
    0 * * * *     cd /path && .venv/bin/python scripts/refresh_btc.py --interval 1h >> logs/refresh.log 2>&1
    0 */4 * * *   cd /path && .venv/bin/python scripts/refresh_btc.py --interval 4h >> logs/refresh.log 2>&1
    5 0 * * *     cd /path && .venv/bin/python scripts/refresh_btc.py --interval 1d >> logs/refresh.log 2>&1
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Add project root so imports work from any directory
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))
load_dotenv(_project_root / ".env")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import structlog  # noqa: E402
from sqlalchemy import text  # noqa: E402

from aether_btc.data.binance_client import BinanceDataClient  # noqa: E402
from aether_btc.data.database import (  # noqa: E402
    INDICATOR_LOOKBACK,
    TIMEFRAMES,
    _candle_table_name,
    _indicators_table_name,
    _live_state_table_name,
    get_engine,
    get_session,
    init_db,
)
from aether_btc.ga.chromosome import Chromosome  # noqa: E402
from aether_btc.ga.fitness import compute_signal_score, generate_signal_with_exits  # noqa: E402
from aether_btc.signals import STRATEGIES, StrategyRunner  # noqa: E402
from aether_btc.signals.indicators import Indicators  # noqa: E402

log = structlog.get_logger()


def _upsert_candles(engine, pair: str, interval: str, df) -> int:
    """Upsert fetched candles into the appropriate candle table."""
    if df.empty:
        return 0

    table = _candle_table_name(interval)
    df = df.copy()
    df["pair"] = pair
    records = df[["pair", "timestamp", "open", "high", "low", "close",
                   "volume", "quote_volume", "trades_count"]].to_dict("records")

    with engine.connect() as conn:
        conn.execute(
            text(f"""
                INSERT INTO {table} (pair, timestamp, open, high, low, close, volume, quote_volume, trades_count)
                VALUES (:pair, :timestamp, :open, :high, :low, :close, :volume, :quote_volume, :trades_count)
                ON CONFLICT (pair, timestamp) DO UPDATE SET
                    open = EXCLUDED.open, high = EXCLUDED.high,
                    low = EXCLUDED.low, close = EXCLUDED.close,
                    volume = EXCLUDED.volume, quote_volume = EXCLUDED.quote_volume,
                    trades_count = EXCLUDED.trades_count
            """),
            records,
        )
        conn.commit()

    return len(records)


def _load_recent_candles(engine, pair: str, interval: str, limit: int):
    """Load recent candles from the appropriate candle table."""
    table = _candle_table_name(interval)

    with engine.connect() as conn:
        rows = conn.execute(
            text(f"""
                SELECT timestamp, open, high, low, close, volume, quote_volume, trades_count
                FROM {table}
                WHERE pair = :pair
                ORDER BY timestamp DESC
                LIMIT :limit
            """),
            {"pair": pair, "limit": limit},
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades_count",
    ])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.set_index("timestamp", inplace=True)
    return df


def _load_recent_funding(engine, pair: str, limit: int):
    """Load recent funding rates from DB."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT timestamp, funding_rate
                FROM funding_rates
                WHERE pair = :pair
                ORDER BY timestamp DESC
                LIMIT :limit
            """),
            {"pair": pair, "limit": limit},
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["timestamp", "funding_rate"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.set_index("timestamp", inplace=True)
    return df


def _load_best_model(session) -> tuple[Chromosome | None, str, float]:
    """Load the best GA model from DB."""
    row = session.execute(
        text("SELECT name, chromosome, fitness FROM models ORDER BY fitness DESC LIMIT 1")
    ).fetchone()

    if not row:
        return None, "", 0.0

    name, chromo_dict, fitness = row
    chromosome = Chromosome.from_dict(chromo_dict)
    chromosome.fitness = fitness or 0.0
    return chromosome, name or "unnamed", fitness or 0.0


def _extract_indicator_dict(df, bar_idx: int) -> dict:
    """Extract all indicator values for a bar as a JSON-safe dict."""
    row = df.iloc[bar_idx]
    ohlcv_cols = {"open", "high", "low", "close", "volume", "quote_volume", "trades_count"}
    indicators = {}
    for col in df.columns:
        if col in ohlcv_cols:
            continue
        val = row[col]
        if pd.isna(val) or (isinstance(val, float) and np.isinf(val)):
            indicators[col] = None
        else:
            indicators[col] = float(val)
    return indicators


def _extract_strategy_signals(df, bar_idx: int) -> dict:
    """Extract per-strategy signals for a bar."""
    row = df.iloc[bar_idx]
    strategies = {}
    for name in ["momentum", "mean_reversion", "trend_following", "volatility_breakout", "funding_volume"]:
        sig_col = f"sig_{name}"
        conf_col = f"sig_{name}_conf"
        if sig_col in df.columns:
            sig_val = row.get(sig_col, 0.0)
            conf_val = row.get(conf_col, 0.0)
            strategies[name] = {
                "signal": float(sig_val) if not pd.isna(sig_val) else 0.0,
                "confidence": float(conf_val) if not pd.isna(conf_val) else 0.0,
            }
    return strategies


def _upsert_live_state(engine, pair: str, interval: str, state: dict) -> None:
    """Upsert the latest live state into the appropriate live state table."""
    table = _live_state_table_name(interval)

    with engine.connect() as conn:
        conn.execute(
            text(f"""
                INSERT INTO {table} (pair, timestamp, price, signal_type, signal_score,
                    signal_confidence, indicators, strategy_signals, model_name, model_fitness, updated_at)
                VALUES (:pair, :timestamp, :price, :signal_type, :signal_score,
                    :signal_confidence, :indicators, :strategy_signals, :model_name, :model_fitness, NOW())
                ON CONFLICT (pair) DO UPDATE SET
                    timestamp = EXCLUDED.timestamp, price = EXCLUDED.price,
                    signal_type = EXCLUDED.signal_type, signal_score = EXCLUDED.signal_score,
                    signal_confidence = EXCLUDED.signal_confidence,
                    indicators = EXCLUDED.indicators, strategy_signals = EXCLUDED.strategy_signals,
                    model_name = EXCLUDED.model_name, model_fitness = EXCLUDED.model_fitness,
                    updated_at = NOW()
            """),
            state,
        )
        conn.commit()


def _upsert_indicators_history(
    engine, pair: str, interval: str, df, start_idx: int = 0, chromosome=None,
) -> int:
    """Upsert indicator rows into the appropriate indicators table."""
    table = _indicators_table_name(interval)
    records = []

    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        indicators = _extract_indicator_dict(df, i)

        signal_type = None
        signal_score = None
        signal_confidence = None

        if chromosome is not None and i >= 200:
            score = compute_signal_score(df, i, chromosome)
            signal = generate_signal_with_exits(df, i, chromosome)
            signal_score = score
            signal_confidence = abs(score - 0.5) * 2
            if signal is not None:
                signal_type = signal.signal_type.value

        records.append({
            "pair": pair,
            "timestamp": df.index[i],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "indicators": json.dumps(indicators),
            "signal_type": signal_type,
            "signal_score": signal_score,
            "signal_confidence": signal_confidence,
        })

    if not records:
        return 0

    with engine.connect() as conn:
        conn.execute(
            text(f"""
                INSERT INTO {table}
                    (pair, timestamp, open, high, low, close, volume, indicators,
                     signal_type, signal_score, signal_confidence)
                VALUES (:pair, :timestamp, :open, :high, :low, :close, :volume,
                        CAST(:indicators AS jsonb), :signal_type, :signal_score, :signal_confidence)
                ON CONFLICT (pair, timestamp) DO UPDATE SET
                    open = EXCLUDED.open, high = EXCLUDED.high,
                    low = EXCLUDED.low, close = EXCLUDED.close,
                    volume = EXCLUDED.volume, indicators = EXCLUDED.indicators,
                    signal_type = EXCLUDED.signal_type, signal_score = EXCLUDED.signal_score,
                    signal_confidence = EXCLUDED.signal_confidence
            """),
            records,
        )
        conn.commit()

    return len(records)


def refresh_single(pair: str, interval: str, bars: int, engine, session) -> None:
    """Run the full refresh pipeline for one timeframe."""
    now = datetime.now(timezone.utc)
    log.info("refresh_start", pair=pair, interval=interval, time=now.isoformat())

    lookback = INDICATOR_LOOKBACK.get(interval, 500)

    # Step 1: Fetch latest candle(s) from Binance
    binance = BinanceDataClient()
    recent_df = binance.fetch_recent_klines(pair=pair, interval=interval, limit=bars)
    if recent_df.empty:
        log.error("no_candles_fetched", interval=interval)
        return

    upserted = _upsert_candles(engine, pair, interval, recent_df)
    log.info("candles_upserted", interval=interval, count=upserted)

    # Step 2: Load recent candles from DB for indicator computation
    candles = _load_recent_candles(engine, pair, interval, limit=lookback)
    if len(candles) < 200:
        log.error("insufficient_candles", interval=interval, count=len(candles), required=200)
        return

    log.info("candles_loaded", interval=interval, count=len(candles))

    # Step 3: Load funding rates
    funding = _load_recent_funding(engine, pair, limit=lookback)
    log.info("funding_loaded", interval=interval, count=len(funding))

    # Step 4: Compute indicators
    candles = Indicators.add_all(candles, funding_rates=funding)
    log.info("indicators_computed", interval=interval, columns=len(candles.columns))

    # Step 5: Precompute strategy signals
    runner = StrategyRunner(STRATEGIES)
    candles = runner.precompute_all_bars(candles)
    log.info("strategies_computed", interval=interval)

    # Step 6: Load best GA model
    chromosome, model_name, model_fitness = _load_best_model(session)

    # Step 7: Generate signal for latest bar
    latest_idx = len(candles) - 1
    latest_bar = candles.iloc[latest_idx]
    price = float(latest_bar["close"])

    signal_type = "HOLD"
    signal_score = 0.5
    signal_confidence = 0.0

    if chromosome is not None:
        signal_score = compute_signal_score(candles, latest_idx, chromosome)
        signal = generate_signal_with_exits(candles, latest_idx, chromosome)
        signal_confidence = abs(signal_score - 0.5) * 2

        if signal is not None:
            signal_type = signal.signal_type.value
        log.info(
            "signal_generated", interval=interval,
            signal=signal_type, score=f"{signal_score:.4f}",
            confidence=f"{signal_confidence:.4f}", model=model_name,
        )
    else:
        log.warning("no_model_found", interval=interval, using="default_hold")

    # Step 8: Store live state
    indicator_dict = _extract_indicator_dict(candles, latest_idx)
    strategy_signals = _extract_strategy_signals(candles, latest_idx)

    _upsert_live_state(engine, pair, interval, {
        "pair": pair,
        "timestamp": candles.index[latest_idx],
        "price": price,
        "signal_type": signal_type,
        "signal_score": signal_score,
        "signal_confidence": signal_confidence,
        "indicators": json.dumps(indicator_dict),
        "strategy_signals": json.dumps(strategy_signals),
        "model_name": model_name or None,
        "model_fitness": model_fitness or None,
    })
    log.info("live_state_stored", interval=interval)

    # Step 9: Store indicator history (only the latest bars)
    store_start = max(0, len(candles) - bars)
    stored = _upsert_indicators_history(
        engine, pair, interval, candles, start_idx=store_start, chromosome=chromosome,
    )
    log.info("indicators_history_stored", interval=interval, count=stored)

    log.info(
        "refresh_complete", interval=interval,
        pair=pair, price=f"${price:,.2f}", signal=signal_type,
        score=f"{signal_score:.4f}", confidence=f"{signal_confidence:.4f}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC refresh pipeline (multi-timeframe)")
    parser.add_argument("--pair", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--interval", default="15m", choices=TIMEFRAMES, help="Timeframe interval")
    parser.add_argument("--bars", type=int, default=5, help="Number of recent bars to fetch")
    parser.add_argument("--all", action="store_true", help="Refresh all timeframes")
    args = parser.parse_args()

    init_db()
    engine = get_engine()
    session = get_session()

    try:
        if args.all:
            for tf in TIMEFRAMES:
                refresh_single(args.pair, tf, args.bars, engine, session)
        else:
            refresh_single(args.pair, args.interval, args.bars, engine, session)
    finally:
        session.close()


if __name__ == "__main__":
    main()
