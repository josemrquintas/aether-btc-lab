#!/usr/bin/env python3
"""Train GA on historical data with walk-forward validation.

Usage:
    python scripts/train_ga.py --pair BTCUSDT --population 80 --generations 50
"""

import argparse
import json
import time
import uuid

import structlog
from dotenv import load_dotenv

from aether_btc.backtest.engine import BacktestConfig
from aether_btc.data.database import GATrainingProgress, Model, get_session, init_db
from aether_btc.data.pipeline import DataPipeline
from aether_btc.ga.chromosome import Chromosome
from aether_btc.ga.engine import GAConfig, GAEngine, GenerationStats
from aether_btc.ga.fitness import FitnessResult, evaluate_chromosome
from aether_btc.signals import STRATEGIES, StrategyRunner
from aether_btc.signals.indicators import Indicators

load_dotenv()
log = structlog.get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GA on historical data")
    parser.add_argument("--pair", default="BTCUSDT")
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default="results/ga_result.json")
    args = parser.parse_args()

    init_db()
    pipeline = DataPipeline()

    # Load data
    log.info("loading_data", pair=args.pair)
    candles = pipeline.load_candles(pair=args.pair)
    funding = pipeline.load_funding_rates(pair=args.pair)

    if candles.empty:
        log.error("no_candles_found")
        return

    # Compute indicators
    log.info("computing_indicators")
    candles = Indicators.add_all(candles, funding_rates=funding)

    # Precompute strategy signals
    log.info("precomputing_strategies")
    runner = StrategyRunner(STRATEGIES)
    candles = runner.precompute_all_bars(candles)

    # Setup fitness function (closure over data)
    backtest_config = BacktestConfig()

    def fitness_fn(chromosome: Chromosome) -> FitnessResult:
        return evaluate_chromosome(
            chromosome, candles, funding, config=backtest_config,
        )

    # Run GA
    run_id = str(uuid.uuid4())[:8]
    session = get_session()
    start_time = time.time()

    def progress_callback(gen: int, stats: GenerationStats) -> None:
        elapsed = time.time() - start_time
        progress = GATrainingProgress(
            run_id=run_id,
            generation=gen,
            best_fitness=stats.best_fitness,
            avg_fitness=stats.avg_fitness,
            best_sharpe=stats.best_result.sharpe_ratio if stats.best_result else 0,
            best_calmar=stats.best_result.calmar_ratio if stats.best_result else 0,
            best_max_drawdown=stats.best_result.max_drawdown if stats.best_result else 0,
            total_trades=stats.best_result.total_trades if stats.best_result else 0,
            elapsed_seconds=elapsed,
        )
        session.merge(progress)
        session.commit()

    ga_config = GAConfig(
        population_size=args.population,
        generations=args.generations,
        parallel_workers=args.workers,
        random_seed=args.seed,
    )

    engine = GAEngine(ga_config, fitness_fn, progress_callback)
    result = engine.run()

    # Save to DB
    model = Model(
        name=f"ga_{run_id}",
        chromosome=result.best_chromosome.genes,
        fitness=result.best_fitness,
        sharpe=result.best_result.sharpe_ratio,
        calmar=result.best_result.calmar_ratio,
        config={"population": args.population, "generations": args.generations},
    )
    session.add(model)
    session.commit()

    # Save to file
    engine.save_results(args.output)

    log.info(
        "training_complete",
        run_id=run_id,
        best_fitness=f"{result.best_fitness:.3f}",
        generations=result.generations_run,
        elapsed=f"{time.time() - start_time:.1f}s",
    )


if __name__ == "__main__":
    main()
