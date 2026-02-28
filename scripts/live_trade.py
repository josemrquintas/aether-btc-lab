#!/usr/bin/env python3
"""Live trading bot for BTC/USDT on Binance Futures.

Runs every 15 minutes:
1. Fetch latest candle
2. Compute indicators
3. Apply GA chromosome
4. Generate signal
5. Execute on Binance Futures (or paper trade on testnet)

Usage:
    python scripts/live_trade.py --model-id 1
    python scripts/live_trade.py --chromosome results/ga_result.json --paper
"""

import argparse
import asyncio
import json
import time

import structlog
from dotenv import load_dotenv

from aether_btc.core.config import AetherBTCConfig
from aether_btc.ga.chromosome import Chromosome

load_dotenv()
log = structlog.get_logger()


async def run_cycle(chromosome: Chromosome, config: AetherBTCConfig, paper: bool = True) -> None:
    """Run a single 15-minute trading cycle."""
    log.info("cycle_start", paper=paper)

    # TODO: Implement full cycle:
    # 1. Fetch latest 15min candle from Binance
    # 2. Load recent candles from DB for indicator computation
    # 3. Compute indicators
    # 4. Generate signal from chromosome
    # 5. Execute order on Binance Futures (or log for paper)
    # 6. Send Telegram alert

    log.info("cycle_complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live trading bot")
    parser.add_argument("--chromosome", default=None, help="Path to GA result JSON")
    parser.add_argument("--model-id", type=int, default=None, help="Model ID from DB")
    parser.add_argument("--paper", action="store_true", help="Paper trading mode")
    parser.add_argument("--once", action="store_true", help="Run single cycle and exit")
    args = parser.parse_args()

    config = AetherBTCConfig()

    # Load chromosome
    if args.chromosome:
        with open(args.chromosome) as f:
            data = json.load(f)
        chromosome = Chromosome.from_dict(data["best_genes"])
    elif args.model_id:
        from aether_btc.data.database import Model, get_session
        session = get_session()
        model = session.query(Model).get(args.model_id)
        if not model:
            log.error("model_not_found", id=args.model_id)
            return
        chromosome = Chromosome.from_dict(model.chromosome)
    else:
        log.error("must_specify_chromosome_or_model_id")
        return

    paper = args.paper or config.binance.testnet
    log.info("live_trade_starting", paper=paper)

    if args.once:
        asyncio.run(run_cycle(chromosome, config, paper))
    else:
        # Run on 15-minute schedule
        while True:
            try:
                asyncio.run(run_cycle(chromosome, config, paper))
            except Exception as e:
                log.error("cycle_error", error=str(e))

            # Wait until next 15-minute mark
            now = time.time()
            next_bar = (int(now / 900) + 1) * 900
            sleep_time = max(0, next_bar - now + 5)  # +5s buffer
            log.info("waiting_for_next_bar", sleep_seconds=int(sleep_time))
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
