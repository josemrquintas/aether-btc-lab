#!/usr/bin/env python3
"""Backfill indicators + per-strategy signals across full candle history.

Populates `btc_indicators_<interval>` for every bar in `candles_<interval>`,
not just the rolling window that `refresh_btc.py` maintains. This is a
one-shot script required before any historical backtest that reads from
the indicators table.

Writes pure indicator + per-strategy signal data (no GA chromosome signal —
chromosome=None). The GA signal columns remain NULL and are populated only
by the live refresh loop.

Usage:
    python scripts/backfill_indicators.py                # 15m, BTCUSDT, chunks of 5000
    python scripts/backfill_indicators.py --interval 1h
    python scripts/backfill_indicators.py --chunk-size 2000 --start 2020-01-01
"""

import argparse
import sys
import time
from pathlib import Path

import structlog
from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))
load_dotenv(_project_root / ".env")

from aether_btc.data.database import get_engine  # noqa: E402
from aether_btc.data.pipeline import DataPipeline  # noqa: E402
from aether_btc.signals import STRATEGIES, StrategyRunner  # noqa: E402
from aether_btc.signals.indicators import Indicators  # noqa: E402

# Reuse the upsert helper from refresh_btc
sys.path.insert(0, str(_project_root / "scripts"))
from refresh_btc import _upsert_indicators_history  # noqa: E402

log = structlog.get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill indicators across full candle history")
    parser.add_argument("--pair", default="BTCUSDT")
    parser.add_argument("--interval", default="15m", choices=["15m", "1h", "4h", "1d"])
    parser.add_argument("--start", default=None, help="Optional start date YYYY-MM-DD (filters candles)")
    parser.add_argument("--end", default=None, help="Optional end date YYYY-MM-DD")
    parser.add_argument("--chunk-size", type=int, default=5000,
                        help="Rows per upsert batch (default 5000)")
    args = parser.parse_args()

    engine = get_engine()
    pipeline = DataPipeline()

    t0 = time.time()
    log.info("loading_candles", pair=args.pair, interval=args.interval)
    candles = pipeline.load_candles(pair=args.pair, interval=args.interval)
    if candles.empty:
        log.error("no_candles_found", pair=args.pair, interval=args.interval)
        sys.exit(1)

    if args.start:
        candles = candles[candles.index >= args.start]
    if args.end:
        candles = candles[candles.index <= args.end]

    log.info("candles_loaded", count=len(candles),
             start=str(candles.index[0]), end=str(candles.index[-1]))

    log.info("loading_funding_rates")
    funding = pipeline.load_funding_rates(pair=args.pair)
    log.info("funding_rates_loaded", count=len(funding))

    log.info("computing_indicators")
    candles = Indicators.add_all(candles, funding_rates=funding)
    log.info("indicators_computed", columns=len(candles.columns))

    log.info("precomputing_strategies")
    runner = StrategyRunner(STRATEGIES)
    candles = runner.precompute_all_bars(candles)
    log.info("strategies_precomputed", columns=len(candles.columns))

    # Chunked upsert so we don't blow the SQL statement / memory buffer
    total = len(candles)
    chunk = args.chunk_size
    written = 0
    t_upsert = time.time()

    log.info("upsert_starting", total=total, chunk_size=chunk)
    for start_idx in range(0, total, chunk):
        end_idx = min(start_idx + chunk, total)
        sub = candles.iloc[start_idx:end_idx]
        n = _upsert_indicators_history(
            engine, args.pair, args.interval, sub,
            start_idx=0, chromosome=None,
        )
        written += n
        elapsed = time.time() - t_upsert
        rate = written / elapsed if elapsed > 0 else 0
        eta = (total - written) / rate if rate > 0 else 0
        log.info("chunk_upserted",
                 progress=f"{written}/{total}",
                 pct=f"{100 * written / total:.1f}%",
                 rate=f"{rate:.0f} rows/s",
                 eta_s=f"{eta:.0f}")

    log.info("backfill_complete",
             total_bars=written,
             total_seconds=f"{time.time() - t0:.1f}")


if __name__ == "__main__":
    main()
