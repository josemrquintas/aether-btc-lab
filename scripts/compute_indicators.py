#!/usr/bin/env python3
"""Compute all technical indicators on historical candle data.

Usage:
    python scripts/compute_indicators.py --pair BTCUSDT
"""

import argparse

import structlog
from dotenv import load_dotenv

from aether_btc.data.pipeline import DataPipeline
from aether_btc.signals import STRATEGIES, StrategyRunner
from aether_btc.signals.indicators import Indicators

load_dotenv()
log = structlog.get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute indicators on historical data")
    parser.add_argument("--pair", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--interval", default="15m", help="Candle interval (15m, 1h, 4h, 1d)")
    args = parser.parse_args()

    pipeline = DataPipeline()

    log.info("loading_candles", pair=args.pair, interval=args.interval)
    candles = pipeline.load_candles(pair=args.pair, interval=args.interval)
    if candles.empty:
        log.error("no_candles_found", pair=args.pair)
        return

    log.info("candles_loaded", count=len(candles))

    log.info("loading_funding_rates", pair=args.pair)
    funding = pipeline.load_funding_rates(pair=args.pair)
    log.info("funding_rates_loaded", count=len(funding))

    log.info("computing_indicators")
    candles = Indicators.add_all(candles, funding_rates=funding)

    log.info("precomputing_strategies")
    runner = StrategyRunner(STRATEGIES)
    candles = runner.precompute_all_bars(candles)

    # Check for NaN after warmup
    warmup = 200
    post_warmup = candles.iloc[warmup:]
    nan_counts = post_warmup.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]

    if not nan_cols.empty:
        log.warning("unexpected_nan_after_warmup", columns=dict(nan_cols))
    else:
        log.info("no_unexpected_nan")

    log.info(
        "indicators_complete",
        total_bars=len(candles),
        total_columns=len(candles.columns),
        warmup_bars=warmup,
    )


if __name__ == "__main__":
    main()
