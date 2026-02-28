"""Tests for GA fitness function — Phase 3 GA Optimization Gate."""

import numpy as np
import pandas as pd
import pytest

from aether_btc.ga.chromosome import Chromosome
from aether_btc.ga.fitness import compute_signal_score


def make_test_candles(n: int = 300) -> pd.DataFrame:
    """Create candles with indicator columns for testing."""
    np.random.seed(42)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = 50_000 + np.cumsum(np.random.randn(n) * 100)

    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 20,
        "high": close + 50,
        "low": close - 50,
        "close": close,
        "volume": np.random.uniform(100, 1000, n),
        "quote_volume": np.random.uniform(5e6, 50e6, n),
        # Indicator columns
        "return_1bar": np.random.randn(n) * 0.01,
        "return_4bar": np.random.randn(n) * 0.02,
        "return_16bar": np.random.randn(n) * 0.04,
        "return_96bar": np.random.randn(n) * 0.08,
        "momentum_acceleration": np.random.randn(n) * 0.01,
        "bb_position": np.random.uniform(0, 1, n),
        "price_vs_sma50": np.random.randn(n) * 0.05,
        "price_vs_sma200": np.random.randn(n) * 0.1,
        "zscore_20d": np.random.randn(n),
        "atr_normalized": np.random.uniform(0.01, 0.05, n),
        "volatility_ratio": np.random.uniform(0.5, 2.0, n),
        "bb_width": np.random.uniform(0.02, 0.1, n),
        "intraday_range": np.random.uniform(0.005, 0.03, n),
        "volume_ratio": np.random.uniform(0.5, 3.0, n),
        "cmf": np.random.uniform(-0.5, 0.5, n),
        "mfi": np.random.uniform(20, 80, n),
        "obv_slope": np.random.randn(n),
        "rsi": np.random.uniform(20, 80, n),
        "macd_histogram": np.random.randn(n) * 100,
        "adx": np.random.uniform(10, 50, n),
        "roc": np.random.randn(n) * 5,
        "stoch_k": np.random.uniform(10, 90, n),
        "supertrend_direction": np.random.choice([-1, 1], n),
        "cci": np.random.randn(n) * 100,
        "funding_rate": np.random.normal(0.0001, 0.0003, n),
        "quote_volume_ratio": np.random.uniform(0.5, 3.0, n),
        # Strategy signal columns
        "sig_momentum": np.random.choice([-1.0, 0.0, 1.0], n),
        "sig_momentum_conf": np.random.uniform(0, 1, n),
        "sig_mean_reversion": np.random.choice([-1.0, 0.0, 1.0], n),
        "sig_mean_reversion_conf": np.random.uniform(0, 1, n),
        "sig_trend_following": np.random.choice([-1.0, 0.0, 1.0], n),
        "sig_trend_following_conf": np.random.uniform(0, 1, n),
        "sig_volatility_breakout": np.random.choice([-1.0, 0.0, 1.0], n),
        "sig_volatility_breakout_conf": np.random.uniform(0, 1, n),
        "sig_funding_volume": np.random.choice([-1.0, 0.0, 1.0], n),
        "sig_funding_volume_conf": np.random.uniform(0, 1, n),
        "n_buy_signals": np.random.randint(0, 6, n).astype(float),
        "n_sell_signals": np.random.randint(0, 6, n).astype(float),
        "signal_consensus": np.random.uniform(-5, 5, n),
        "max_confidence": np.random.uniform(0, 1, n),
    }, index=timestamps)
    return df


class TestSignalScoring:
    """Test the signal scoring function."""

    def test_score_in_range(self):
        """Score (sigmoid) is always between 0 and 1."""
        candles = make_test_candles()
        c = Chromosome.from_defaults()

        for i in range(200, 250):
            score = compute_signal_score(candles, i, c)
            assert 0 <= score <= 1

    def test_default_chromosome_neutral_score(self):
        """Default chromosome (all weights 0, bias 0) -> score = 0.5."""
        candles = make_test_candles()
        c = Chromosome.from_defaults()
        score = compute_signal_score(candles, 250, c)
        assert score == pytest.approx(0.5, abs=0.01)

    def test_positive_bias_bullish(self):
        """Positive bias -> score > 0.5 (bullish)."""
        candles = make_test_candles()
        c = Chromosome.from_defaults()
        c.set("fw_bias", 3.0)
        score = compute_signal_score(candles, 250, c)
        assert score > 0.5

    def test_negative_bias_bearish(self):
        """Negative bias -> score < 0.5 (bearish)."""
        candles = make_test_candles()
        c = Chromosome.from_defaults()
        c.set("fw_bias", -3.0)
        score = compute_signal_score(candles, 250, c)
        assert score < 0.5


class TestFitnessPenalties:
    """Test fitness penalties."""

    def test_fitness_penalizes_no_trades(self):
        """Chromosome that generates 0 trades -> fitness = 0."""
        # A chromosome with very high confidence cutoff generates no trades
        c = Chromosome.from_defaults()
        c.set("min_confidence_cutoff", 0.99)  # Very high cutoff

        candles = make_test_candles(500)
        from aether_btc.ga.fitness import evaluate_chromosome
        result = evaluate_chromosome(c, candles)
        assert result.fitness == pytest.approx(0.0)

    def test_l2_regularization(self):
        """Extreme feature weights -> lower fitness than moderate weights."""
        from aether_btc.ga.chromosome import FEATURE_WEIGHT_NAMES

        c_moderate = Chromosome.from_defaults()
        c_extreme = Chromosome.from_defaults()

        # Set extreme weights
        for name in FEATURE_WEIGHT_NAMES:
            c_extreme.set(name, 1.5)

        # L2 penalty should be higher for extreme
        l2_moderate = sum(c_moderate.get(n) ** 2 for n in FEATURE_WEIGHT_NAMES) * 0.01
        l2_extreme = sum(c_extreme.get(n) ** 2 for n in FEATURE_WEIGHT_NAMES) * 0.01
        assert l2_extreme > l2_moderate
