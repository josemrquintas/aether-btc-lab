"""GA fitness function for crypto backtesting.

Evaluates a chromosome by running a backtest and computing
multi-objective fitness: Sharpe + Calmar with regularization.
"""

import numpy as np
import pandas as pd
import structlog

from aether_btc.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from aether_btc.core.enums import SignalType
from aether_btc.core.models import Signal
from aether_btc.ga.chromosome import (
    FEATURE_COLUMN_MAP,
    FEATURE_WEIGHT_NAMES,
    Chromosome,
)

log = structlog.get_logger()


def compute_signal_score(
    candles: pd.DataFrame,
    bar_idx: int,
    chromosome: Chromosome,
) -> float:
    """Compute signal score using GA-evolved feature weights.

    score = sigmoid(dot(features, weights) + bias)
    Returns score in [0, 1] where >0.5 = bullish, <0.5 = bearish
    """
    row = candles.iloc[bar_idx]

    # Extract features in order
    features = []
    weights = []
    for gene_name in FEATURE_WEIGHT_NAMES:
        col_name = FEATURE_COLUMN_MAP.get(gene_name)
        if col_name and col_name in candles.columns:
            val = row.get(col_name, 0.0)
            if pd.isna(val):
                val = 0.0
            features.append(float(val))
            weights.append(float(chromosome.get(gene_name)))
        else:
            features.append(0.0)
            weights.append(float(chromosome.get(gene_name)))

    bias = float(chromosome.get("fw_bias"))
    raw_score = np.dot(features, weights) + bias

    # Sigmoid
    score = 1.0 / (1.0 + np.exp(-np.clip(raw_score, -10, 10)))
    return float(score)


def generate_signal(
    candles: pd.DataFrame,
    bar_idx: int,
    chromosome: Chromosome,
) -> Signal | None:
    """Generate a trading signal from GA chromosome scoring.

    Returns Signal or None if score is not decisive enough.
    """
    # Need warmup period
    if bar_idx < 200:
        return None

    min_hold = int(chromosome.get("min_hold_bars"))
    cutoff = float(chromosome.get("min_confidence_cutoff"))
    funding_threshold = float(chromosome.get("funding_rate_threshold"))

    score = compute_signal_score(candles, bar_idx, chromosome)
    bar_ts = candles.index[bar_idx]

    # Check funding rate threshold
    if "funding_rate" in candles.columns:
        funding = candles.iloc[bar_idx].get("funding_rate", 0.0)
        if pd.notna(funding):
            # Positive funding = longs pay -> adverse for longs if above threshold
            if score > 0.5 and funding > funding_threshold:
                return None
            if score < 0.5 and -funding > funding_threshold:
                return None

    # Convert score to signal
    confidence = abs(score - 0.5) * 2  # Map [0,1] -> [0,1] confidence

    if confidence < cutoff:
        return None

    if score > 0.5:
        return Signal(
            pair="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=confidence,
            timestamp=bar_ts,
            metadata={"score": score},
        )
    else:
        return Signal(
            pair="BTCUSDT",
            signal_type=SignalType.SHORT,
            confidence=confidence,
            timestamp=bar_ts,
            metadata={"score": score},
        )


def generate_signal_with_exits(
    candles: pd.DataFrame,
    bar_idx: int,
    chromosome: Chromosome,
) -> Signal | None:
    """Generate signal including exit signals (SELL/COVER) based on score reversal."""
    if bar_idx < 200:
        return None

    score = compute_signal_score(candles, bar_idx, chromosome)
    cutoff = float(chromosome.get("min_confidence_cutoff"))
    confidence = abs(score - 0.5) * 2
    bar_ts = candles.index[bar_idx]

    # For exits: score crossing midpoint generates exit signal
    if score > 0.5 and confidence >= cutoff:
        return Signal(
            pair="BTCUSDT", signal_type=SignalType.BUY,
            confidence=confidence, timestamp=bar_ts,
        )
    elif score < 0.5 and confidence >= cutoff:
        return Signal(
            pair="BTCUSDT", signal_type=SignalType.SHORT,
            confidence=confidence, timestamp=bar_ts,
        )

    # Weak signal = exit
    if score > 0.4 and score <= 0.5:
        return Signal(
            pair="BTCUSDT", signal_type=SignalType.COVER,
            confidence=0.5, timestamp=bar_ts,
        )
    elif score < 0.6 and score >= 0.5:
        return Signal(
            pair="BTCUSDT", signal_type=SignalType.SELL,
            confidence=0.5, timestamp=bar_ts,
        )

    return None


def evaluate_chromosome(
    chromosome: Chromosome,
    candles: pd.DataFrame,
    funding_rates: pd.DataFrame | None = None,
    config: BacktestConfig | None = None,
    sharpe_weight: float = 0.6,
    calmar_weight: float = 0.4,
) -> "FitnessResult":
    """Evaluate a chromosome by running backtest and computing fitness.

    fitness = sharpe_weight * min(sharpe, 5.0) + calmar_weight * min(calmar, 10.0)
    With regularization penalties.
    """
    cfg = config or BacktestConfig()

    engine = BacktestEngine(
        config=cfg,
        chromosome=chromosome,
        signal_fn=generate_signal_with_exits,
    )

    try:
        result = engine.run(candles, funding_rates)
    except Exception as e:
        log.warning("backtest_failed", error=str(e))
        return FitnessResult(fitness=0.0)

    # Base fitness
    sharpe = min(result.sharpe_ratio, 5.0)
    calmar = min(result.calmar_ratio, 10.0)
    fitness = sharpe_weight * sharpe + calmar_weight * calmar

    # Penalties
    if result.total_trades == 0:
        # Harsh but not zero — still distinguishable from "traded but lost"
        fitness = -2.0
    elif result.total_trades < 10:
        fitness *= 0.5  # Hard penalty for very few trades

    if result.liquidations > 0:
        fitness -= result.liquidations * 0.5  # Heavy penalty for liquidations

    # Max drawdown penalty — soft penalty above 40% DD (only applied to positive fitness)
    if fitness > 0 and result.max_drawdown > 0.40:
        dd_excess = result.max_drawdown - 0.40
        fitness *= max(0.3, 1.0 - dd_excess)  # 60% DD -> 0.8x, 80% DD -> 0.6x

    # L2 regularization on feature weights
    l2_penalty = 0.0
    for name in FEATURE_WEIGHT_NAMES:
        w = chromosome.get(name)
        l2_penalty += w * w
    l2_penalty *= 0.001
    fitness -= l2_penalty

    return FitnessResult(
        fitness=fitness,
        sharpe_ratio=result.sharpe_ratio,
        calmar_ratio=result.calmar_ratio,
        total_return=result.total_return,
        max_drawdown=result.max_drawdown,
        win_rate=result.win_rate,
        total_trades=result.total_trades,
        liquidations=result.liquidations,
    )


from dataclasses import dataclass


@dataclass
class FitnessResult:
    """Result of fitness evaluation."""

    fitness: float
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    liquidations: int = 0

    def to_dict(self) -> dict:
        return {
            "fitness": self.fitness,
            "sharpe_ratio": self.sharpe_ratio,
            "calmar_ratio": self.calmar_ratio,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "liquidations": self.liquidations,
        }
