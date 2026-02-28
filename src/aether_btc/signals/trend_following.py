"""Trend following signal strategy.

Only trades in the direction of the dominant trend. Uses ADX as a gate —
if the market isn't trending (ADX < 20), no signals are generated.
"""

import pandas as pd

from aether_btc.signals.base import SignalStrategy, SignalType, StrategySignal


class TrendFollowingStrategy(SignalStrategy):
    """Evaluate EMA crossover, SMA200 trend, Supertrend, and MACD position with ADX gate."""

    ADX_GATE = 20.0
    SUB_SIGNALS = [
        ("ema_crossover", 0.8),
        ("sma200_trend", 0.6),
        ("supertrend_flip", 0.9),
        ("macd_position", 0.5),
    ]

    @property
    def name(self) -> str:
        return "trend_following"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def min_confirmations(self) -> int:
        return 2

    def get_required_columns(self) -> list[str]:
        return ["ema_9", "ema_21", "sma_200", "close", "supertrend_direction", "adx", "macd"]

    def generate_signal(self, df: pd.DataFrame, bar_idx: int) -> StrategySignal:
        if bar_idx < 1:
            return StrategySignal(SignalType.HOLD, 0.0, 0, len(self.SUB_SIGNALS))

        curr = df.iloc[bar_idx]
        prev = df.iloc[bar_idx - 1]
        meta: dict = {}

        # ADX gate — no signals in ranging markets
        adx_val = curr.get("adx")
        if pd.isna(adx_val) or adx_val < self.ADX_GATE:
            meta["adx_gate"] = "blocked"
            return StrategySignal(SignalType.HOLD, 0.0, 0, len(self.SUB_SIGNALS), meta)

        bullish_weight = 0.0
        bearish_weight = 0.0
        bullish_count = 0
        bearish_count = 0

        # 1. EMA crossover (9 crosses 21)
        ema9_curr = curr.get("ema_9")
        ema21_curr = curr.get("ema_21")
        ema9_prev = prev.get("ema_9")
        ema21_prev = prev.get("ema_21")
        if all(pd.notna(v) for v in [ema9_curr, ema21_curr, ema9_prev, ema21_prev]):
            if ema9_prev <= ema21_prev and ema9_curr > ema21_curr:
                bullish_weight += 0.8
                bullish_count += 1
                meta["ema"] = "bullish"
            elif ema9_prev >= ema21_prev and ema9_curr < ema21_curr:
                bearish_weight += 0.8
                bearish_count += 1
                meta["ema"] = "bearish"

        # 2. SMA200 trend filter
        close = curr.get("close")
        sma200 = curr.get("sma_200")
        if pd.notna(close) and pd.notna(sma200):
            if close > sma200:
                bullish_weight += 0.6
                bullish_count += 1
                meta["sma200"] = "bullish"
            elif close < sma200:
                bearish_weight += 0.6
                bearish_count += 1
                meta["sma200"] = "bearish"

        # 3. Supertrend flip
        st_curr = curr.get("supertrend_direction")
        st_prev = prev.get("supertrend_direction")
        if pd.notna(st_curr) and pd.notna(st_prev):
            if st_prev == -1 and st_curr == 1:
                bullish_weight += 0.9
                bullish_count += 1
                meta["supertrend"] = "bullish"
            elif st_prev == 1 and st_curr == -1:
                bearish_weight += 0.9
                bearish_count += 1
                meta["supertrend"] = "bearish"

        # 4. MACD position (above/below zero line)
        macd_val = curr.get("macd")
        if pd.notna(macd_val):
            if macd_val > 0:
                bullish_weight += 0.5
                bullish_count += 1
                meta["macd"] = "bullish"
            elif macd_val < 0:
                bearish_weight += 0.5
                bearish_count += 1
                meta["macd"] = "bearish"

        total_confirmations = max(bullish_count, bearish_count)
        if total_confirmations < self.min_confirmations:
            return StrategySignal(SignalType.HOLD, 0.0, 0, len(self.SUB_SIGNALS), meta)

        # ADX bonus for strong trends
        adx_bonus = min((adx_val - 20) / 30, 0.3)

        # Confidence with ADX bonus
        winning_weight = max(bullish_weight, bearish_weight)
        confidence = min((winning_weight + adx_bonus) / 2.5, 1.0)

        if bullish_weight > bearish_weight:
            signal_type = SignalType.BUY
        elif bearish_weight > bullish_weight:
            signal_type = SignalType.SHORT
        else:
            return StrategySignal(SignalType.HOLD, 0.0, 0, len(self.SUB_SIGNALS), meta)

        return StrategySignal(
            signal_type=signal_type,
            confidence=confidence,
            confirmations=total_confirmations,
            total_checks=len(self.SUB_SIGNALS),
            metadata=meta,
        )
