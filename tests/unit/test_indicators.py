"""Tests for indicator computation — Phase 1 Data Integrity Gate."""

import numpy as np
import pandas as pd
import pytest

from aether_btc.signals.indicators import Indicators


def make_ohlcv(n: int = 500) -> pd.DataFrame:
    """Create sample OHLCV data."""
    np.random.seed(42)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = 50_000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    open_ = close + np.random.randn(n) * 20

    df = pd.DataFrame({
        "open": open_,
        "high": np.maximum(high, np.maximum(open_, close)),
        "low": np.minimum(low, np.minimum(open_, close)),
        "close": close,
        "volume": np.random.uniform(100, 1000, n),
    }, index=timestamps)
    return df


class TestNoLookaheadBias:
    """Test that indicators don't use future data."""

    def test_no_lookahead_bias_rsi(self):
        """Corrupt future data, verify RSI values don't change."""
        df = make_ohlcv(300)
        test_idx = 150

        rsi_full = Indicators.rsi(df["close"])
        val_full = rsi_full.iloc[test_idx]

        df_corrupt = df.copy()
        df_corrupt.iloc[test_idx + 1:, df_corrupt.columns.get_loc("close")] = np.nan
        rsi_corrupt = Indicators.rsi(df_corrupt["close"])
        val_corrupt = rsi_corrupt.iloc[test_idx]

        assert np.isclose(val_full, val_corrupt, equal_nan=True)

    def test_no_lookahead_bias_sma(self):
        df = make_ohlcv(300)
        test_idx = 150

        sma_full = Indicators.sma(df["close"], 20)
        val_full = sma_full.iloc[test_idx]

        df_corrupt = df.copy()
        df_corrupt.iloc[test_idx + 1:, df_corrupt.columns.get_loc("close")] = np.nan
        sma_corrupt = Indicators.sma(df_corrupt["close"], 20)
        val_corrupt = sma_corrupt.iloc[test_idx]

        assert np.isclose(val_full, val_corrupt)


class TestIndicatorWarmup:
    """Test that indicators have proper NaN warmup."""

    def test_indicator_warmup_nans_sma(self):
        """First N-1 rows of SMA(20) are NaN."""
        df = make_ohlcv(100)
        sma = Indicators.sma(df["close"], 20)
        assert sma.iloc[:19].isna().all()
        assert sma.iloc[19:].notna().all()

    def test_indicator_warmup_nans_rsi(self):
        """First N-1 rows of RSI(14) are NaN (ta library produces value at index period-1)."""
        df = make_ohlcv(100)
        rsi = Indicators.rsi(df["close"], 14)
        # RSI(14) produces NaN for first 13 rows (indices 0-12), value at index 13
        assert rsi.iloc[:13].isna().all()
        assert rsi.iloc[13:].notna().all()


class TestAllIndicatorsComputed:
    """Test no unexpected NaN after warmup period."""

    def test_all_indicators_computed(self):
        """After warmup (200 bars), no unexpected NaN in standard indicators."""
        df = make_ohlcv(500)
        df = Indicators.add_trend(df)
        df = Indicators.add_momentum(df)
        df = Indicators.add_volatility(df)

        # Check after warmup
        post_warmup = df.iloc[200:]
        indicator_cols = [c for c in post_warmup.columns if c not in ["open", "high", "low", "close", "volume"]]
        for col in indicator_cols:
            nan_count = post_warmup[col].isna().sum()
            assert nan_count == 0, f"{col} has {nan_count} NaN values after warmup"


class TestRSIBounds:
    """Test RSI bounds."""

    def test_rsi_bounds(self):
        """RSI always between 0 and 100."""
        df = make_ohlcv(500)
        rsi = Indicators.rsi(df["close"])
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()


class TestBollingerBandOrdering:
    """Test Bollinger Band ordering."""

    def test_bollinger_band_ordering(self):
        """lower < middle < upper always."""
        df = make_ohlcv(500)
        bb = Indicators.bollinger_bands(df["close"])
        valid = bb.dropna()
        assert (valid["bb_lower"] < valid["bb_middle"]).all()
        assert (valid["bb_middle"] < valid["bb_upper"]).all()

    def test_bollinger_band_width_positive(self):
        """Band width is always positive."""
        df = make_ohlcv(500)
        bb = Indicators.bollinger_bands(df["close"])
        valid = bb["bb_width"].dropna()
        assert (valid > 0).all()
