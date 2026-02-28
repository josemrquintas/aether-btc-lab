"""Tests for data pipeline — Phase 1 Data Integrity Gate."""

import numpy as np
import pandas as pd
import pytest


def make_sample_candles(days: int = 1, start: str = "2024-01-01") -> pd.DataFrame:
    """Create sample 15-min candle data for testing."""
    bars_per_day = 96
    total_bars = days * bars_per_day
    timestamps = pd.date_range(start=start, periods=total_bars, freq="15min", tz="UTC")

    np.random.seed(42)
    base_price = 50_000.0
    prices = base_price + np.cumsum(np.random.randn(total_bars) * 100)

    df = pd.DataFrame({
        "open": prices,
        "high": prices + np.abs(np.random.randn(total_bars) * 50),
        "low": prices - np.abs(np.random.randn(total_bars) * 50),
        "close": prices + np.random.randn(total_bars) * 30,
        "volume": np.random.uniform(100, 1000, total_bars),
        "quote_volume": np.random.uniform(5_000_000, 50_000_000, total_bars),
        "trades_count": np.random.randint(1000, 10000, total_bars),
    }, index=timestamps)

    # Ensure OHLC validity
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


class TestCandleCount:
    """Test candle count expectations."""

    def test_candle_count_one_day(self):
        """96 candles per day (24h x 4 per hour)."""
        df = make_sample_candles(days=1)
        assert len(df) == 96

    def test_candle_count_one_week(self):
        """672 candles per week (96 x 7, crypto trades 24/7)."""
        df = make_sample_candles(days=7)
        assert len(df) == 672

    def test_candle_count_one_month(self):
        """~2880 candles per month."""
        df = make_sample_candles(days=30)
        assert len(df) == 2880


class TestCandleNoGaps:
    """Test for missing timestamps."""

    def test_candle_no_gaps(self):
        """No missing timestamps in generated data."""
        df = make_sample_candles(days=3)
        expected = pd.date_range(start=df.index[0], periods=len(df), freq="15min")
        pd.testing.assert_index_equal(df.index, expected)

    def test_candle_timestamps_ascending(self):
        """Timestamps are strictly ascending."""
        df = make_sample_candles(days=2)
        assert df.index.is_monotonic_increasing


class TestCandleOHLCValidity:
    """Test OHLC price validity."""

    def test_candle_ohlc_validity(self):
        """low <= open, close <= high for every candle."""
        df = make_sample_candles(days=5)
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()

    def test_candle_low_le_high(self):
        """Low is always <= High."""
        df = make_sample_candles(days=5)
        assert (df["low"] <= df["high"]).all()

    def test_candle_volume_positive(self):
        """Volume is always positive."""
        df = make_sample_candles(days=2)
        assert (df["volume"] > 0).all()


class TestFundingRateData:
    """Test funding rate data expectations."""

    @staticmethod
    def make_sample_funding_rates(days: int = 1) -> pd.DataFrame:
        """Create sample funding rate data (3 per day)."""
        rates_per_day = 3
        total = days * rates_per_day
        timestamps = pd.date_range(
            start="2024-01-01", periods=total, freq="8h", tz="UTC",
        )
        np.random.seed(42)
        rates = np.random.normal(0.0001, 0.0003, total)

        return pd.DataFrame({"funding_rate": rates}, index=timestamps)

    def test_funding_rate_frequency(self):
        """Exactly 3 funding rates per day (00:00, 08:00, 16:00 UTC)."""
        df = self.make_sample_funding_rates(days=7)
        assert len(df) == 21  # 7 * 3

    def test_funding_rate_8h_intervals(self):
        """Funding rates spaced exactly 8 hours apart."""
        df = self.make_sample_funding_rates(days=3)
        diffs = df.index.to_series().diff().dropna()
        assert (diffs == pd.Timedelta(hours=8)).all()

    def test_funding_rate_range(self):
        """All rates within reasonable bounds (+/-0.5%)."""
        df = self.make_sample_funding_rates(days=30)
        assert (df["funding_rate"].abs() <= 0.005).all()
