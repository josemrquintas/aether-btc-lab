#!/usr/bin/env python3
"""Fetch historical BTC/USDT 15min candles and funding rates from Binance.

Usage:
    python scripts/fetch_historical.py --pair BTCUSDT --interval 15m --start "1 Sep 2019"
"""

import argparse

import structlog
from dotenv import load_dotenv

from aether_btc.data.binance_client import BinanceDataClient
from aether_btc.data.database import init_db
from aether_btc.data.pipeline import DataPipeline

load_dotenv()
log = structlog.get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical data from Binance")
    parser.add_argument("--pair", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--interval", default="15m", help="Candle interval")
    parser.add_argument("--start", default="1 Sep 2019", help="Start date")
    parser.add_argument("--end", default=None, help="End date (default: now)")
    parser.add_argument("--skip-candles", action="store_true", help="Skip candle fetching")
    parser.add_argument("--skip-funding", action="store_true", help="Skip funding rate fetching")
    args = parser.parse_args()

    log.info("initializing_database")
    init_db()

    pipeline = DataPipeline()

    if not args.skip_candles:
        log.info("fetching_candles", pair=args.pair, interval=args.interval, start=args.start)
        count = pipeline.fetch_and_store_candles(
            pair=args.pair, interval=args.interval, start=args.start, end=args.end,
        )
        log.info("candles_complete", count=count)

    if not args.skip_funding:
        log.info("fetching_funding_rates", pair=args.pair, start=args.start)
        count = pipeline.fetch_and_store_funding_rates(
            pair=args.pair, start=args.start, end=args.end,
        )
        log.info("funding_rates_complete", count=count)

    # Summary
    total_candles = pipeline.get_candle_count(args.pair)
    total_funding = pipeline.get_funding_rate_count(args.pair)
    log.info("fetch_complete", total_candles=total_candles, total_funding=total_funding)


if __name__ == "__main__":
    main()
