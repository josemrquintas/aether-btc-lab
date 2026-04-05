#!/usr/bin/env python3
"""Walk-forward GA training with per-window evolution + OOS analysis.

For each walk-forward window:
  1. Train a separate GA on the training data
  2. Test the best chromosome out-of-sample
  3. Collect gene weights for stability analysis

Usage:
    python scripts/train_walk_forward.py --population 80 --generations 50 --workers 4
    python scripts/train_walk_forward.py --population 10 --generations 5 --workers 1  # quick verify
"""

import argparse
import json
import time
import uuid
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import structlog
from dotenv import load_dotenv
from tqdm import tqdm

from aether_btc.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from aether_btc.backtest.walk_forward import (
    WalkForwardWindow,
    generate_walk_forward_windows,
)
from aether_btc.data.database import (
    GATrainingProgress,
    Model,
    WalkForwardProgress,
    get_session,
    init_db,
)
from aether_btc.data.pipeline import DataPipeline
from aether_btc.ga.chromosome import FEATURE_WEIGHT_NAMES, Chromosome
from aether_btc.ga.engine import GAConfig, GAEngine, GenerationStats
from aether_btc.ga.fitness import FitnessResult, evaluate_chromosome, generate_signal_with_exits
from aether_btc.signals import STRATEGIES, StrategyRunner
from aether_btc.signals.indicators import Indicators

load_dotenv()
log = structlog.get_logger()


@dataclass
class WindowResult:
    """Result for a single walk-forward window."""

    window_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_bars: int
    test_bars: int
    # Training
    train_fitness: float
    train_sharpe: float
    train_return: float
    train_max_dd: float
    train_trades: int
    generations_run: int
    # OOS test
    oos_sharpe: float
    oos_return: float
    oos_max_dd: float
    oos_trades: int
    oos_win_rate: float
    oos_calmar: float
    oos_liquidations: int
    # Best genes for this window
    best_genes: dict


def _make_fitness_fn(
    train_candles: pd.DataFrame,
    funding_rates: pd.DataFrame | None,
    backtest_config: BacktestConfig,
):
    """Factory to create a fitness closure over specific train data.

    Using a factory avoids the closure-over-loop-variable bug.
    """
    def fitness_fn(chromosome: Chromosome) -> FitnessResult:
        return evaluate_chromosome(
            chromosome, train_candles, funding_rates, config=backtest_config,
        )
    return fitness_fn


def train_single_window(
    window_idx: int,
    window: WalkForwardWindow,
    candles: pd.DataFrame,
    funding_rates: pd.DataFrame | None,
    backtest_config: BacktestConfig,
    ga_config: GAConfig,
    run_id: str,
    session: object = None,
) -> WindowResult:
    """Train GA on one window's training data, evaluate OOS on test data."""
    log.info(
        "window_start",
        window=window_idx + 1,
        train=f"{window.train_start.date()} to {window.train_end.date()}",
        test=f"{window.test_start.date()} to {window.test_end.date()}",
    )

    # Slice train and test data
    train_candles = candles.loc[window.train_start:window.train_end]
    test_candles = candles.loc[window.test_start:window.test_end]

    train_funding = None
    test_funding = None
    if funding_rates is not None:
        train_funding = funding_rates.loc[window.train_start:window.train_end]
        test_funding = funding_rates.loc[window.test_start:window.test_end]

    log.info(
        "window_data",
        window=window_idx + 1,
        train_bars=len(train_candles),
        test_bars=len(test_candles),
    )

    # Train GA on training data
    fitness_fn = _make_fitness_fn(train_candles, train_funding, backtest_config)
    window_run_id = f"wf_{run_id}_w{window_idx}"
    w_start = time.time()

    gen_bar = tqdm(
        total=ga_config.generations,
        desc=f"  W{window_idx + 1} GA",
        unit="gen",
        leave=False,
    )

    def _progress(gen: int, stats: GenerationStats) -> None:
        gen_bar.update(1)
        gen_bar.set_postfix(best=f"{stats.best_fitness:.2f}", avg=f"{stats.avg_fitness:.2f}")
        # Write per-generation progress to DB
        if session is not None:
            elapsed = time.time() - w_start
            progress = GATrainingProgress(
                run_id=window_run_id,
                generation=gen,
                best_fitness=float(stats.best_fitness),
                avg_fitness=float(stats.avg_fitness),
                best_sharpe=float(stats.best_result.sharpe_ratio) if stats.best_result else 0.0,
                best_calmar=float(stats.best_result.calmar_ratio) if stats.best_result else 0.0,
                best_max_drawdown=float(stats.best_result.max_drawdown) if stats.best_result else 0.0,
                total_trades=int(stats.best_result.total_trades) if stats.best_result else 0,
                elapsed_seconds=elapsed,
            )
            session.merge(progress)
            session.commit()

    engine = GAEngine(ga_config, fitness_fn, progress_callback=_progress)
    ga_result = engine.run()
    gen_bar.close()

    best_chromosome = ga_result.best_chromosome
    train_result = ga_result.best_result

    log.info(
        "window_train_done",
        window=window_idx + 1,
        fitness=f"{ga_result.best_fitness:.3f}",
        generations=ga_result.generations_run,
    )

    # Evaluate OOS on test data
    oos_engine = BacktestEngine(
        config=backtest_config,
        chromosome=best_chromosome,
        signal_fn=generate_signal_with_exits,
    )
    try:
        oos_result = oos_engine.run(test_candles, test_funding)
    except Exception as e:
        log.warning("window_oos_error", window=window_idx + 1, error=str(e))
        oos_result = BacktestResult(
            total_return=0.0, annualized_return=0.0, sharpe_ratio=0.0,
            sortino_ratio=0.0, calmar_ratio=0.0, max_drawdown=0.0,
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
            profit_factor=0.0, avg_trade_return=0.0, avg_hold_hours=0.0,
            long_trades=0, short_trades=0, long_pnl=0.0, short_pnl=0.0,
            total_commissions=0.0, total_slippage=0.0, total_funding_costs=0.0,
            cost_drag_percent=0.0, avg_leverage=0.0, max_leverage=0.0,
            liquidations=0, equity_curve=pd.DataFrame(), trades=[],
            bar_returns=pd.Series(dtype=float),
        )

    log.info(
        "window_oos_done",
        window=window_idx + 1,
        oos_sharpe=f"{oos_result.sharpe_ratio:.2f}",
        oos_return=f"{oos_result.total_return:.2%}",
        oos_trades=oos_result.total_trades,
    )

    return WindowResult(
        window_idx=window_idx,
        train_start=str(window.train_start.date()),
        train_end=str(window.train_end.date()),
        test_start=str(window.test_start.date()),
        test_end=str(window.test_end.date()),
        train_bars=len(train_candles),
        test_bars=len(test_candles),
        train_fitness=ga_result.best_fitness,
        train_sharpe=train_result.sharpe_ratio,
        train_return=train_result.total_return,
        train_max_dd=train_result.max_drawdown,
        train_trades=train_result.total_trades,
        generations_run=ga_result.generations_run,
        oos_sharpe=oos_result.sharpe_ratio,
        oos_return=oos_result.total_return,
        oos_max_dd=oos_result.max_drawdown,
        oos_trades=oos_result.total_trades,
        oos_win_rate=oos_result.win_rate,
        oos_calmar=oos_result.calmar_ratio,
        oos_liquidations=oos_result.liquidations,
        best_genes=best_chromosome.genes,
    )


def analyze_genes(window_results: list[WindowResult]) -> dict:
    """Analyze gene stability across walk-forward windows.

    Returns per-gene statistics (mean, std, min, max) and
    strategy gene ranking by absolute mean weight.
    """
    if not window_results:
        return {}

    # Collect all gene values across windows
    all_gene_names = list(window_results[0].best_genes.keys())
    gene_stats = {}

    for name in all_gene_names:
        values = [wr.best_genes[name] for wr in window_results]
        gene_stats[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "abs_mean": float(np.abs(np.mean(values))),
        }

    # Strategy gene ranking by abs_mean weight
    strategy_gene_names = [n for n in FEATURE_WEIGHT_NAMES if n.startswith("fw_sig_") or n in (
        "fw_n_buy_signals", "fw_n_sell_signals", "fw_signal_consensus", "fw_max_confidence",
    )]
    strategy_ranking = sorted(
        [(name, gene_stats[name]) for name in strategy_gene_names if name in gene_stats],
        key=lambda x: x[1]["abs_mean"],
        reverse=True,
    )

    return {
        "per_gene": gene_stats,
        "strategy_ranking": [
            {"gene": name, **stats} for name, stats in strategy_ranking
        ],
    }


def print_report(window_results: list[WindowResult], gene_analysis: dict) -> None:
    """Print formatted walk-forward report to console."""
    print("\n" + "=" * 100)
    print("WALK-FORWARD GA TRAINING REPORT")
    print("=" * 100)

    # Per-window table
    print(f"\n{'Win':>3} | {'Train Period':<25} | {'Test Period':<25} | "
          f"{'Train Fit':>9} | {'OOS Sharpe':>10} | {'OOS Ret':>8} | "
          f"{'OOS MaxDD':>9} | {'Trades':>6} | {'WinRate':>7}")
    print("-" * 120)

    for wr in window_results:
        print(
            f"{wr.window_idx + 1:>3} | "
            f"{wr.train_start} to {wr.train_end:<4} | "
            f"{wr.test_start} to {wr.test_end:<4} | "
            f"{wr.train_fitness:>9.3f} | "
            f"{wr.oos_sharpe:>10.2f} | "
            f"{wr.oos_return:>7.1%} | "
            f"{wr.oos_max_dd:>8.1%} | "
            f"{wr.oos_trades:>6} | "
            f"{wr.oos_win_rate:>6.1%}"
        )

    # Aggregate OOS stats
    oos_returns = [wr.oos_return for wr in window_results]
    oos_sharpes = [wr.oos_sharpe for wr in window_results]
    positive_windows = sum(1 for r in oos_returns if r > 0)
    consistency = positive_windows / len(oos_returns) if oos_returns else 0

    print("\n" + "=" * 100)
    print("AGGREGATE OOS METRICS")
    print("=" * 100)
    print(f"  Windows:           {len(window_results)}")
    print(f"  Consistency:       {consistency:.0%} ({positive_windows}/{len(window_results)} positive)")
    print(f"  Mean OOS Sharpe:   {np.mean(oos_sharpes):.2f}")
    print(f"  Mean OOS Return:   {np.mean(oos_returns):.2%}")
    print(f"  Worst OOS MaxDD:   {max(wr.oos_max_dd for wr in window_results):.2%}")
    print(f"  Total OOS Trades:  {sum(wr.oos_trades for wr in window_results)}")

    # Strategy gene analysis
    if gene_analysis.get("strategy_ranking"):
        print("\n" + "=" * 100)
        print("STRATEGY GENE RANKING (by abs_mean weight across windows)")
        print("=" * 100)
        print(f"  {'Gene':<35} | {'AbsMean':>8} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
        print("  " + "-" * 90)
        for item in gene_analysis["strategy_ranking"]:
            print(
                f"  {item['gene']:<35} | "
                f"{item['abs_mean']:>8.4f} | "
                f"{item['mean']:>8.4f} | "
                f"{item['std']:>8.4f} | "
                f"{item['min']:>8.4f} | "
                f"{item['max']:>8.4f}"
            )

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward GA training")
    parser.add_argument("--pair", default="BTCUSDT")
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-days", type=int, default=180)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--output", default="results/walk_forward_result.json")
    parser.add_argument("--start", default=None, help="Start date for data (e.g. '2024-01-01')")
    parser.add_argument("--end", default=None, help="End date for data (e.g. '2025-01-01')")
    args = parser.parse_args()

    init_db()
    pipeline = DataPipeline()
    run_id = str(uuid.uuid4())[:8]

    # Load data
    log.info("loading_data", pair=args.pair)
    candles = pipeline.load_candles(pair=args.pair)
    funding = pipeline.load_funding_rates(pair=args.pair)

    if candles.empty:
        log.error("no_candles_found")
        return

    # Trim to date range if specified
    if args.start:
        candles = candles[args.start:]
        if funding is not None and not funding.empty:
            funding = funding[args.start:]
    if args.end:
        candles = candles[:args.end]
        if funding is not None and not funding.empty:
            funding = funding[:args.end]

    log.info("data_loaded", bars=len(candles), start=str(candles.index[0]), end=str(candles.index[-1]))

    # Compute indicators + strategies ONCE on full data
    log.info("computing_indicators")
    candles = Indicators.add_all(candles, funding_rates=funding)

    log.info("precomputing_strategies")
    runner = StrategyRunner(STRATEGIES)
    candles = runner.precompute_all_bars(candles)

    # Generate walk-forward windows
    windows = generate_walk_forward_windows(candles, args.train_days, args.test_days)
    if not windows:
        log.error("no_windows_generated")
        return

    log.info("windows_generated", count=len(windows))

    # GA config (used for each window)
    ga_config = GAConfig(
        population_size=args.population,
        generations=args.generations,
        parallel_workers=args.workers,
        random_seed=args.seed,
    )
    backtest_config = BacktestConfig()

    # Train each window
    start_time = time.time()
    window_results: list[WindowResult] = []
    session = get_session()

    window_bar = tqdm(windows, desc="Walk-forward windows", unit="win")
    for i, window in enumerate(window_bar):
        window_bar.set_description(
            f"Window {i + 1}/{len(windows)} "
            f"[{window.train_start.date()} - {window.test_end.date()}]"
        )

        # Write window-level progress: training started
        wf_progress = WalkForwardProgress(
            run_id=run_id,
            total_windows=len(windows),
            current_window=i,
            status="training",
            train_start=str(window.train_start.date()),
            train_end=str(window.train_end.date()),
            test_start=str(window.test_start.date()),
            test_end=str(window.test_end.date()),
        )
        session.add(wf_progress)
        session.commit()

        w_start = time.time()
        wr = train_single_window(
            window_idx=i,
            window=window,
            candles=candles,
            funding_rates=funding,
            backtest_config=backtest_config,
            ga_config=ga_config,
            run_id=run_id,
            session=session,
        )
        window_results.append(wr)
        elapsed_w = time.time() - w_start

        # Update window-level progress: complete with results
        # Cast numpy types to native Python types for psycopg2 compatibility
        wf_progress.status = "complete"
        wf_progress.train_fitness = float(wr.train_fitness)
        wf_progress.oos_sharpe = float(wr.oos_sharpe)
        wf_progress.oos_return = float(wr.oos_return)
        wf_progress.oos_max_dd = float(wr.oos_max_dd)
        wf_progress.oos_trades = int(wr.oos_trades)
        wf_progress.oos_win_rate = float(wr.oos_win_rate)
        wf_progress.oos_calmar = float(wr.oos_calmar)
        wf_progress.generations_run = int(wr.generations_run)
        wf_progress.window_elapsed_seconds = float(elapsed_w)
        session.commit()

        # Save per-window model to DB
        model = Model(
            name=f"wf_{run_id}_w{i}",
            chromosome=wr.best_genes,
            fitness=float(wr.train_fitness),
            sharpe=float(wr.oos_sharpe),
            calmar=float(wr.oos_calmar),
            train_start=window.train_start,
            train_end=window.train_end,
            test_start=window.test_start,
            test_end=window.test_end,
            config={
                "population": args.population,
                "generations": args.generations,
                "run_id": run_id,
                "window": i,
            },
        )
        session.add(model)
        session.commit()

        window_bar.set_postfix(
            fitness=f"{wr.train_fitness:.2f}",
            oos_sharpe=f"{wr.oos_sharpe:.2f}",
            elapsed=f"{elapsed_w:.0f}s",
        )

    total_elapsed = time.time() - start_time

    # Gene analysis
    gene_analysis = analyze_genes(window_results)

    # Print report
    print_report(window_results, gene_analysis)

    # Save JSON output
    output = {
        "run_id": run_id,
        "config": {
            "pair": args.pair,
            "population": args.population,
            "generations": args.generations,
            "workers": args.workers,
            "train_days": args.train_days,
            "test_days": args.test_days,
        },
        "total_elapsed_seconds": total_elapsed,
        "windows": [asdict(wr) for wr in window_results],
        "aggregate": {
            "num_windows": len(window_results),
            "consistency": sum(1 for wr in window_results if wr.oos_return > 0) / len(window_results),
            "mean_oos_sharpe": float(np.mean([wr.oos_sharpe for wr in window_results])),
            "mean_oos_return": float(np.mean([wr.oos_return for wr in window_results])),
            "worst_oos_max_dd": float(max(wr.oos_max_dd for wr in window_results)),
            "total_oos_trades": sum(wr.oos_trades for wr in window_results),
        },
        "gene_analysis": gene_analysis,
    }

    from pathlib import Path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    log.info(
        "walk_forward_complete",
        run_id=run_id,
        windows=len(window_results),
        elapsed=f"{total_elapsed:.1f}s",
        output=str(output_path),
    )


if __name__ == "__main__":
    main()
