"""Volatility breakout signal strategy.

Detects volatility expansion after compression — Keltner/Donchian channel
breaks, BB squeeze releases, and strong bars. Particularly effective in
crypto around major news events and liquidation cascades.
"""

import pandas as pd

from aether_btc.signals.base import SignalStrategy, SignalType, StrategySignal


class VolatilityBreakoutStrategy(SignalStrategy):
    """Evaluate Keltner breaks, Donchian breaks, BB squeeze, and strong bars."""

    SUB_SIGNALS = [
        ("keltner_break", 0.8),
        ("donchian_break", 0.7),
        ("bb_squeeze_release", 0.9),
        ("strong_bar", 0.6),
    ]

    @property
    def name(self) -> str:
        return "volatility_breakout"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def min_confirmations(self) -> int:
        return 2

    def get_required_columns(self) -> list[str]:
        return [
            "kc_upper", "kc_lower", "dc_upper", "dc_lower",
            "bb_width", "bb_middle", "atr", "close", "high", "low",
        ]

    def generate_signal(self, df: pd.DataFrame, bar_idx: int) -> StrategySignal:
        if bar_idx < 1:
            return StrategySignal(SignalType.HOLD, 0.0, 0, len(self.SUB_SIGNALS))

        curr = df.iloc[bar_idx]
        prev = df.iloc[bar_idx - 1]
        bullish_weight = 0.0
        bearish_weight = 0.0
        bullish_count = 0
        bearish_count = 0
        meta: dict = {}

        close = curr.get("close")

        # 1. Keltner Channel break
        kc_upper = curr.get("kc_upper")
        kc_lower = curr.get("kc_lower")
        if pd.notna(close) and pd.notna(kc_upper) and pd.notna(kc_lower):
            if close > kc_upper:
                bullish_weight += 0.8
                bullish_count += 1
                meta["keltner"] = "bullish"
            elif close < kc_lower:
                bearish_weight += 0.8
                bearish_count += 1
                meta["keltner"] = "bearish"

        # 2. Donchian Channel break
        dc_upper = curr.get("dc_upper")
        dc_lower = curr.get("dc_lower")
        if pd.notna(close) and pd.notna(dc_upper) and pd.notna(dc_lower):
            if close >= dc_upper:
                bullish_weight += 0.7
                bullish_count += 1
                meta["donchian"] = "bullish"
            elif close <= dc_lower:
                bearish_weight += 0.7
                bearish_count += 1
                meta["donchian"] = "bearish"

        # 3. BB squeeze release
        bb_width_curr = curr.get("bb_width")
        bb_width_prev = prev.get("bb_width")
        bb_middle = curr.get("bb_middle")
        if all(pd.notna(v) for v in [bb_width_curr, bb_width_prev, bb_middle, close]):
            # Compute a simple SMA of bb_width over recent bars for squeeze detection
            start_idx = max(0, bar_idx - 19)
            bb_width_slice = df.iloc[start_idx:bar_idx + 1]["bb_width"]
            bb_width_sma = bb_width_slice.mean() if len(bb_width_slice) > 0 else bb_width_curr

            if bb_width_curr > bb_width_prev and bb_width_curr > bb_width_sma:
                if close > bb_middle:
                    bullish_weight += 0.9
                    bullish_count += 1
                    meta["bb_squeeze"] = "bullish"
                elif close < bb_middle:
                    bearish_weight += 0.9
                    bearish_count += 1
                    meta["bb_squeeze"] = "bearish"

        # 4. Strong bar (range > 1.5 * ATR with directional close)
        high = curr.get("high")
        low = curr.get("low")
        atr_val = curr.get("atr")
        if all(pd.notna(v) for v in [high, low, atr_val, close]) and atr_val > 0:
            bar_range = high - low
            if bar_range > 1.5 * atr_val:
                # Check where close is in bar range
                bar_position = (close - low) / bar_range if bar_range > 0 else 0.5
                if bar_position > 0.75:
                    bullish_weight += 0.6
                    bullish_count += 1
                    meta["strong_bar"] = "bullish"
                elif bar_position < 0.25:
                    bearish_weight += 0.6
                    bearish_count += 1
                    meta["strong_bar"] = "bearish"

        total_confirmations = max(bullish_count, bearish_count)
        if total_confirmations < self.min_confirmations:
            return StrategySignal(SignalType.HOLD, 0.0, 0, len(self.SUB_SIGNALS), meta)

        winning_weight = max(bullish_weight, bearish_weight)
        confidence = min(winning_weight / 2.0, 1.0)

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
