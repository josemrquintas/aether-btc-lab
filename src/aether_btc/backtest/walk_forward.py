"""Walk-forward validation for backtest robustness testing.

Adapted from Aether v3 with crypto-specific windows:
- Train: 6 months (180 days)
- Test: 2 months (60 days)
- Rolling windows with no test overlap
"""

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
import structlog

from aether_btc.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from aether_btc.ga.chromosome import Chromosome

log = structlog.get_logger()


@dataclass
class WalkForwardWindow:
    """A single train/test window."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class WalkForwardResult:
    """Aggregated results from all walk-forward windows."""

    windows: list[WalkForwardWindow]
    per_window_results: list[BacktestResult]
    aggregate_sharpe: float
    aggregate_return: float
    aggregate_max_drawdown: float
    out_of_sample_consistency: float


class WalkForwardRunner:
    """Runs walk-forward validation with 6-month train / 2-month test windows."""

    def __init__(
        self,
        backtest_config: BacktestConfig,
        chromosome: Chromosome,
        signal_fn: object,
        train_days: int = 180,
        test_days: int = 60,
    ) -> None:
        self.backtest_config = backtest_config
        self.chromosome = chromosome
        self.signal_fn = signal_fn
        self.train_days = train_days
        self.test_days = test_days

    def generate_windows(self, candles: pd.DataFrame) -> list[WalkForwardWindow]:
        """Split candle data into rolling train/test windows."""
        windows: list[WalkForwardWindow] = []
        total_start = candles.index[0]
        total_end = candles.index[-1]

        current = total_start
        while True:
            train_start = current
            train_end = train_start + timedelta(days=self.train_days)
            test_start = train_end + timedelta(minutes=15)  # Next bar
            test_end = test_start + timedelta(days=self.test_days)

            if test_end > total_end:
                test_end = total_end
                if test_start >= total_end:
                    break

            windows.append(WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            ))

            current = test_start

        log.info("walk_forward_windows", count=len(windows))
        return windows

    def run(
        self,
        candles: pd.DataFrame,
        funding_rates: pd.DataFrame | None = None,
    ) -> WalkForwardResult:
        """Run backtest on each test window, return aggregated results."""
        windows = self.generate_windows(candles)
        if not windows:
            raise ValueError("No walk-forward windows could be generated")

        per_window_results: list[BacktestResult] = []

        for i, window in enumerate(windows):
            log.info(
                "walk_forward_window",
                window=i + 1,
                total=len(windows),
                test_start=window.test_start,
                test_end=window.test_end,
            )

            # Extract test window data
            test_candles = candles.loc[window.test_start:window.test_end]
            if len(test_candles) < 96:  # Less than 1 day
                log.warning("walk_forward_skip_window", window=i + 1, bars=len(test_candles))
                continue

            test_funding = None
            if funding_rates is not None:
                test_funding = funding_rates.loc[window.test_start:window.test_end]

            try:
                engine = BacktestEngine(
                    config=self.backtest_config,
                    chromosome=self.chromosome,
                    signal_fn=self.signal_fn,
                )
                result = engine.run(test_candles, test_funding)
                per_window_results.append(result)
            except (ValueError, Exception) as e:
                log.warning("walk_forward_error", window=i + 1, error=str(e))

        if not per_window_results:
            raise ValueError("No walk-forward windows produced results")

        # Aggregate
        returns = [r.total_return for r in per_window_results]
        positive_windows = sum(1 for r in returns if r > 0)
        consistency = positive_windows / len(returns)

        all_bar_returns = pd.concat(
            [r.bar_returns for r in per_window_results if not r.bar_returns.empty],
            ignore_index=True,
        )
        if len(all_bar_returns) > 1:
            ann_factor = np.sqrt(35040)
            vol = all_bar_returns.std() * ann_factor
            ann_ret = all_bar_returns.mean() * 35040
            agg_sharpe = (ann_ret - 0.02) / vol if vol > 0 else 0.0
        else:
            agg_sharpe = 0.0

        agg_return = float(np.mean(returns)) if returns else 0.0
        agg_dd = max(r.max_drawdown for r in per_window_results) if per_window_results else 0.0

        log.info(
            "walk_forward_complete",
            windows=len(per_window_results),
            consistency=f"{consistency:.0%}",
            aggregate_sharpe=f"{agg_sharpe:.2f}",
        )

        return WalkForwardResult(
            windows=windows,
            per_window_results=per_window_results,
            aggregate_sharpe=agg_sharpe,
            aggregate_return=agg_return,
            aggregate_max_drawdown=agg_dd,
            out_of_sample_consistency=consistency,
        )
