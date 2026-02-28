"""Safe indicator calculations for 15-minute crypto candles.

Adapted from Aether v3 indicators.py. All indicators:
1. Use only backward-looking calculations
2. Have explicit warm-up periods (NaN, not filled)
3. Are validated against lookahead bias
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import structlog
from ta.momentum import (
    ROCIndicator,
    RSIIndicator,
    StochasticOscillator,
    StochRSIIndicator,
    WilliamsRIndicator,
)
from ta.trend import MACD, ADXIndicator, AroonIndicator, EMAIndicator, PSARIndicator, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands, DonchianChannel, KeltnerChannel
from ta.volume import (
    ChaikinMoneyFlowIndicator,
    ForceIndexIndicator,
    MFIIndicator,
    OnBalanceVolumeIndicator,
    VolumeWeightedAveragePrice,
)

log = structlog.get_logger()


@dataclass
class IndicatorMeta:
    """Metadata for an indicator including lookback requirements."""
    name: str
    lookback: int
    description: str
    category: str


class Indicators:
    """Safe indicator calculations with lookahead bias prevention.

    All methods return NaN for warm-up period (no forward fill).
    Adapted from Aether v3 for any OHLCV DataFrame.
    """

    REGISTRY: dict[str, IndicatorMeta] = {}

    # === TREND INDICATORS ===

    @staticmethod
    def sma(series: pd.Series, period: int = 20) -> pd.Series:
        return SMAIndicator(series, window=period).sma_indicator()

    @staticmethod
    def ema(series: pd.Series, period: int = 20) -> pd.Series:
        return EMAIndicator(series, window=period).ema_indicator()

    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        macd = MACD(close, window_fast=fast, window_slow=slow, window_sign=signal)
        return pd.DataFrame({
            "macd": macd.macd(),
            "macd_signal": macd.macd_signal(),
            "macd_histogram": macd.macd_diff(),
        })

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
        adx = ADXIndicator(high, low, close, window=period)
        return pd.DataFrame({
            "adx": adx.adx(),
            "adx_pos": adx.adx_pos(),
            "adx_neg": adx.adx_neg(),
        })

    @staticmethod
    def aroon(high: pd.Series, low: pd.Series, period: int = 25) -> pd.DataFrame:
        aroon = AroonIndicator(high, low, window=period)
        return pd.DataFrame({
            "aroon_up": aroon.aroon_up(),
            "aroon_down": aroon.aroon_down(),
            "aroon_indicator": aroon.aroon_indicator(),
        })

    @staticmethod
    def psar(high: pd.Series, low: pd.Series, close: pd.Series,
             step: float = 0.02, max_step: float = 0.2) -> pd.DataFrame:
        psar = PSARIndicator(high, low, close, step=step, max_step=max_step)
        return pd.DataFrame({
            "psar": psar.psar(),
            "psar_up": psar.psar_up(),
            "psar_down": psar.psar_down(),
        })

    @staticmethod
    def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                   period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        atr = AverageTrueRange(high, low, close, window=period).average_true_range()
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=float)
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = -1

        for i in range(1, len(close)):
            if close.iloc[i] > supertrend.iloc[i - 1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1

            if direction.iloc[i] == 1 and lower_band.iloc[i] < supertrend.iloc[i - 1]:
                supertrend.iloc[i] = supertrend.iloc[i - 1]
            if direction.iloc[i] == -1 and upper_band.iloc[i] > supertrend.iloc[i - 1]:
                supertrend.iloc[i] = supertrend.iloc[i - 1]

        return pd.DataFrame({"supertrend": supertrend, "supertrend_direction": direction})

    # === MOMENTUM INDICATORS ===

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        return RSIIndicator(close, window=period).rsi()

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        stoch = StochasticOscillator(high, low, close, window=k_period, smooth_window=d_period)
        return pd.DataFrame({"stoch_k": stoch.stoch(), "stoch_d": stoch.stoch_signal()})

    @staticmethod
    def stoch_rsi(close: pd.Series, period: int = 14,
                  k_period: int = 3, d_period: int = 3) -> pd.DataFrame:
        stoch_rsi = StochRSIIndicator(close, window=period, smooth1=k_period, smooth2=d_period)
        return pd.DataFrame({
            "stochrsi": stoch_rsi.stochrsi(),
            "stochrsi_k": stoch_rsi.stochrsi_k(),
            "stochrsi_d": stoch_rsi.stochrsi_d(),
        })

    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
                   period: int = 14) -> pd.Series:
        return WilliamsRIndicator(high, low, close, lbp=period).williams_r()

    @staticmethod
    def roc(close: pd.Series, period: int = 12) -> pd.Series:
        return ROCIndicator(close, window=period).roc()

    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 20) -> pd.Series:
        from ta.trend import CCIIndicator
        return CCIIndicator(high, low, close, window=period).cci()

    # === VOLATILITY INDICATORS ===

    @staticmethod
    def bollinger_bands(close: pd.Series, period: int = 20,
                        std_dev: float = 2.0) -> pd.DataFrame:
        bb = BollingerBands(close, window=period, window_dev=std_dev)
        return pd.DataFrame({
            "bb_upper": bb.bollinger_hband(),
            "bb_middle": bb.bollinger_mavg(),
            "bb_lower": bb.bollinger_lband(),
            "bb_width": bb.bollinger_wband(),
            "bb_pct": bb.bollinger_pband(),
        })

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> pd.Series:
        return AverageTrueRange(high, low, close, window=period).average_true_range()

    @staticmethod
    def keltner_channel(high: pd.Series, low: pd.Series, close: pd.Series,
                        period: int = 20, atr_period: int = 10,
                        multiplier: float = 2.0) -> pd.DataFrame:
        kc = KeltnerChannel(high, low, close, window=period,
                            window_atr=atr_period, multiplier=multiplier)
        return pd.DataFrame({
            "kc_upper": kc.keltner_channel_hband(),
            "kc_middle": kc.keltner_channel_mband(),
            "kc_lower": kc.keltner_channel_lband(),
        })

    @staticmethod
    def donchian_channel(high: pd.Series, low: pd.Series, close: pd.Series,
                         period: int = 20) -> pd.DataFrame:
        dc = DonchianChannel(high, low, close, window=period)
        return pd.DataFrame({
            "dc_upper": dc.donchian_channel_hband(),
            "dc_middle": dc.donchian_channel_mband(),
            "dc_lower": dc.donchian_channel_lband(),
        })

    @staticmethod
    def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # === VOLUME INDICATORS ===

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        return OnBalanceVolumeIndicator(close, volume).on_balance_volume()

    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series,
             volume: pd.Series, period: int = 14) -> pd.Series:
        return VolumeWeightedAveragePrice(
            high, low, close, volume, window=period
        ).volume_weighted_average_price()

    @staticmethod
    def cmf(high: pd.Series, low: pd.Series, close: pd.Series,
            volume: pd.Series, period: int = 20) -> pd.Series:
        return ChaikinMoneyFlowIndicator(high, low, close, volume, window=period).chaikin_money_flow()

    @staticmethod
    def force_index(close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
        return ForceIndexIndicator(close, volume, window=period).force_index()

    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series,
            volume: pd.Series, period: int = 14) -> pd.Series:
        return MFIIndicator(high, low, close, volume, window=period).money_flow_index()

    @staticmethod
    def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
        return volume.rolling(window=period, min_periods=period).mean()

    @staticmethod
    def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
        avg_vol = volume.rolling(window=period, min_periods=period).mean()
        return volume / avg_vol

    # === STATISTICAL INDICATORS ===

    @staticmethod
    def zscore(series: pd.Series, period: int = 20) -> pd.Series:
        mean = series.rolling(window=period, min_periods=period).mean()
        std = series.rolling(window=period, min_periods=period).std()
        return (series - mean) / std

    @staticmethod
    def percentile(series: pd.Series, period: int = 252) -> pd.Series:
        def pct_rank(x: pd.Series) -> float:
            return (x.argsort().argsort().iloc[-1] / (len(x) - 1)) * 100 if len(x) > 1 else 50.0
        return series.rolling(window=period, min_periods=period).apply(pct_rank, raw=False)

    # === CRYPTO-SPECIFIC FEATURES ===

    @staticmethod
    def add_crypto_features(df: pd.DataFrame, funding_rates: pd.DataFrame | None = None) -> pd.DataFrame:
        """Add crypto-specific features.

        - Quote volume ratio (BTC volume in USDT terms)
        - Return features at multiple horizons (adapted for 15min bars)
        - Funding rate as feature (if provided)
        """
        df = df.copy()

        # Returns at multiple horizons (in 15min bars)
        df["return_1bar"] = df["close"].pct_change(1)
        df["return_4bar"] = df["close"].pct_change(4)    # 1 hour
        df["return_16bar"] = df["close"].pct_change(16)   # 4 hours
        df["return_96bar"] = df["close"].pct_change(96)   # 1 day
        df["return_672bar"] = df["close"].pct_change(672)  # 1 week

        # Momentum acceleration
        ret_4 = df["return_4bar"]
        ret_4_prev = ret_4.shift(4)
        df["momentum_acceleration"] = ret_4 - ret_4_prev

        # Normalized ATR (volatility relative to price)
        atr_val = Indicators.atr(df["high"], df["low"], df["close"])
        df["atr_normalized"] = atr_val / df["close"]

        # Volatility ratio (short vs long)
        vol_short = df["close"].pct_change().rolling(16).std()   # 4h volatility
        vol_long = df["close"].pct_change().rolling(96).std()    # 1d volatility
        df["volatility_ratio"] = vol_short / vol_long

        # Intrabar range
        df["intraday_range"] = (df["high"] - df["low"]) / df["close"]

        # Price vs SMAs
        sma50 = Indicators.sma(df["close"], 50)
        sma200 = Indicators.sma(df["close"], 200)
        df["price_vs_sma50"] = (df["close"] - sma50) / sma50
        df["price_vs_sma200"] = (df["close"] - sma200) / sma200

        # Bollinger Band position
        bb = Indicators.bollinger_bands(df["close"])
        df["bb_position"] = bb["bb_pct"]

        # Quote volume ratio
        if "quote_volume" in df.columns:
            avg_qvol = df["quote_volume"].rolling(96).mean()
            df["quote_volume_ratio"] = df["quote_volume"] / avg_qvol

        # Funding rate as feature (merge if provided)
        if funding_rates is not None and not funding_rates.empty:
            df["funding_rate"] = funding_rates["funding_rate"].reindex(df.index).ffill()
        else:
            df["funding_rate"] = 0.0

        return df

    # === CONVENIENCE METHODS ===

    @classmethod
    def add_momentum(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["rsi"] = cls.rsi(df["close"])
        stoch = cls.stochastic(df["high"], df["low"], df["close"])
        df["stoch_k"] = stoch["stoch_k"]
        df["stoch_d"] = stoch["stoch_d"]
        df["williams_r"] = cls.williams_r(df["high"], df["low"], df["close"])
        df["roc"] = cls.roc(df["close"])
        df["cci"] = cls.cci(df["high"], df["low"], df["close"])
        return df

    @classmethod
    def add_trend(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["sma_20"] = cls.sma(df["close"], 20)
        df["sma_50"] = cls.sma(df["close"], 50)
        df["sma_200"] = cls.sma(df["close"], 200)
        df["ema_9"] = cls.ema(df["close"], 9)
        df["ema_21"] = cls.ema(df["close"], 21)

        macd = cls.macd(df["close"])
        df["macd"] = macd["macd"]
        df["macd_signal"] = macd["macd_signal"]
        df["macd_histogram"] = macd["macd_histogram"]

        adx = cls.adx(df["high"], df["low"], df["close"])
        df["adx"] = adx["adx"]
        df["adx_pos"] = adx["adx_pos"]
        df["adx_neg"] = adx["adx_neg"]
        return df

    @classmethod
    def add_volatility(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        bb = cls.bollinger_bands(df["close"])
        df["bb_upper"] = bb["bb_upper"]
        df["bb_middle"] = bb["bb_middle"]
        df["bb_lower"] = bb["bb_lower"]
        df["bb_width"] = bb["bb_width"]
        df["bb_pct"] = bb["bb_pct"]
        df["atr"] = cls.atr(df["high"], df["low"], df["close"])
        kc = cls.keltner_channel(df["high"], df["low"], df["close"])
        df["kc_upper"] = kc["kc_upper"]
        df["kc_middle"] = kc["kc_middle"]
        df["kc_lower"] = kc["kc_lower"]
        return df

    @classmethod
    def add_volume(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["obv"] = cls.obv(df["close"], df["volume"])
        df["cmf"] = cls.cmf(df["high"], df["low"], df["close"], df["volume"])
        df["mfi"] = cls.mfi(df["high"], df["low"], df["close"], df["volume"])
        df["volume_sma"] = cls.volume_sma(df["volume"])
        df["volume_ratio"] = cls.volume_ratio(df["volume"])
        return df

    @classmethod
    def add_all(cls, df: pd.DataFrame, funding_rates: pd.DataFrame | None = None) -> pd.DataFrame:
        """Add all indicators + crypto features to DataFrame."""
        df = cls.add_trend(df)
        df = cls.add_momentum(df)
        df = cls.add_volatility(df)
        df = cls.add_volume(df)
        df = cls.add_crypto_features(df, funding_rates)
        log.info("added_all_indicators", total_columns=len(df.columns))
        return df


def validate_no_lookahead(
    indicator_func: Callable,
    df: pd.DataFrame,
    **kwargs: Any,
) -> bool:
    """Validate that an indicator function has no lookahead bias."""
    if len(df) < 100:
        raise ValueError("Need at least 100 rows for lookahead validation")

    test_idx = len(df) // 2
    full_result = indicator_func(**kwargs)
    if isinstance(full_result, pd.DataFrame):
        full_values = full_result.iloc[test_idx].copy()
    else:
        full_values = full_result.iloc[test_idx]

    df_corrupted = df.copy()
    df_corrupted.iloc[test_idx + 1:] = np.nan

    corrupted_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, pd.Series):
            corrupted_kwargs[key] = df_corrupted[value.name] if value.name in df_corrupted.columns else value
        else:
            corrupted_kwargs[key] = value

    corrupted_result = indicator_func(**corrupted_kwargs)
    if isinstance(corrupted_result, pd.DataFrame):
        corrupted_values = corrupted_result.iloc[test_idx]
    else:
        corrupted_values = corrupted_result.iloc[test_idx]

    if isinstance(full_values, pd.Series):
        return bool(np.allclose(full_values.values, corrupted_values.values, equal_nan=True))
    return bool(np.isclose(full_values, corrupted_values, equal_nan=True))


# Build indicator registry
Indicators.REGISTRY = {
    "sma": IndicatorMeta("SMA", 20, "Simple Moving Average", "trend"),
    "ema": IndicatorMeta("EMA", 60, "Exponential Moving Average", "trend"),
    "macd": IndicatorMeta("MACD", 35, "MACD", "trend"),
    "adx": IndicatorMeta("ADX", 28, "Average Directional Index", "trend"),
    "rsi": IndicatorMeta("RSI", 15, "Relative Strength Index", "momentum"),
    "stochastic": IndicatorMeta("Stochastic", 17, "Stochastic Oscillator", "momentum"),
    "bollinger_bands": IndicatorMeta("Bollinger Bands", 20, "Bollinger Bands", "volatility"),
    "atr": IndicatorMeta("ATR", 14, "Average True Range", "volatility"),
    "obv": IndicatorMeta("OBV", 1, "On Balance Volume", "volume"),
    "mfi": IndicatorMeta("MFI", 14, "Money Flow Index", "volume"),
}
