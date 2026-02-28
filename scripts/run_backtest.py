#!/usr/bin/env python3
"""Run backtest with a trained chromosome.

Usage:
    python scripts/run_backtest.py --chromosome results/ga_result.json
    python scripts/run_backtest.py --model-id 1
"""

import argparse
import json

import structlog
from dotenv import load_dotenv

from aether_btc.backtest.engine import BacktestConfig, BacktestEngine
from aether_btc.data.database import Model, get_session, init_db
from aether_btc.data.pipeline import DataPipeline
from aether_btc.ga.chromosome import Chromosome
from aether_btc.ga.fitness import generate_signal_with_exits
from aether_btc.signals.indicators import Indicators

load_dotenv()
log = structlog.get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest with trained chromosome")
    parser.add_argument("--pair", default="BTCUSDT")
    parser.add_argument("--chromosome", default=None, help="Path to GA result JSON")
    parser.add_argument("--model-id", type=int, default=None, help="Model ID from database")
    parser.add_argument("--capital", type=float, default=1000.0)
    args = parser.parse_args()

    init_db()
    pipeline = DataPipeline()

    # Load chromosome
    if args.chromosome:
        with open(args.chromosome) as f:
            data = json.load(f)
        chromosome = Chromosome.from_dict(data["best_genes"])
    elif args.model_id:
        session = get_session()
        model = session.query(Model).get(args.model_id)
        if not model:
            log.error("model_not_found", id=args.model_id)
            return
        chromosome = Chromosome.from_dict(model.chromosome)
    else:
        log.info("using_default_chromosome")
        chromosome = Chromosome.from_defaults()

    # Load data
    log.info("loading_data", pair=args.pair)
    candles = pipeline.load_candles(pair=args.pair)
    funding = pipeline.load_funding_rates(pair=args.pair)

    if candles.empty:
        log.error("no_candles_found")
        return

    candles = Indicators.add_all(candles, funding_rates=funding)

    # Run backtest
    config = BacktestConfig(initial_capital=args.capital)
    engine = BacktestEngine(config, chromosome, signal_fn=generate_signal_with_exits)
    result = engine.run(candles, funding)

    # Print results
    log.info(
        "backtest_results",
        total_return=f"{result.total_return:.2%}",
        annualized_return=f"{result.annualized_return:.2%}",
        sharpe=f"{result.sharpe_ratio:.2f}",
        sortino=f"{result.sortino_ratio:.2f}",
        calmar=f"{result.calmar_ratio:.2f}",
        max_drawdown=f"{result.max_drawdown:.2%}",
        total_trades=result.total_trades,
        win_rate=f"{result.win_rate:.2%}",
        profit_factor=f"{result.profit_factor:.2f}",
        avg_leverage=f"{result.avg_leverage:.1f}",
        liquidations=result.liquidations,
        long_trades=result.long_trades,
        short_trades=result.short_trades,
        total_commissions=f"${result.total_commissions:.2f}",
        total_funding=f"${result.total_funding_costs:.2f}",
    )


if __name__ == "__main__":
    main()
