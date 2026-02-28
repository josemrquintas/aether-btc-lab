"""Mean reversion signal strategy.

Detects price extremes likely to revert to the mean. BTC regularly
overshoots and snaps back on 15-minute timeframes, especially during
high-funding-rate periods.
"""

import pandas as pd

from aether_btc.signals.base import SignalStrategy, SignalType, StrategySignal


class MeanReversionStrategy(SignalStrategy):
    """Evaluate BB position, Z-score, and RSI for mean reversion signals."""

    SUB_SIGNALS = [
        ("bb_extreme", 0.8),
        ("zscore_extreme", 0.9),
        ("rsi_extreme", 0.7),
    ]
    TOTAL_WEIGHT = sum(w for _, w in SUB_SIGNALS)

    @property
    def name(self) -> str:
        return "mean_reversion"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def min_confirmations(self) -> int:
        return 1

    def get_required_columns(self) -> list[str]:
        return ["bb_position", "zscore_20d", "rsi"]

    def generate_signal(self, df: pd.DataFrame, bar_idx: int) -> StrategySignal:
        curr = df.iloc[bar_idx]
        bullish_weight = 0.0
        bearish_weight = 0.0
        bullish_count = 0
        bearish_count = 0
        meta: dict = {}

        # 1. BB extreme
        bb_pos = curr.get("bb_position")
        if pd.notna(bb_pos):
            if bb_pos <= 0.0:
                bullish_weight += 0.8
                bullish_count += 1
                meta["bb"] = "bullish"
            elif bb_pos >= 1.0:
                bearish_weight += 0.8
                bearish_count += 1
                meta["bb"] = "bearish"

        # 2. Z-score extreme (with proportional weight)
        zscore = curr.get("zscore_20d")
        if pd.notna(zscore):
            if zscore <= -2.0:
                weight = 0.9 * min(abs(zscore) / 3.0, 1.0)
                bullish_weight += weight
                bullish_count += 1
                meta["zscore"] = f"bullish ({zscore:.2f})"
            elif zscore >= 2.0:
                weight = 0.9 * min(abs(zscore) / 3.0, 1.0)
                bearish_weight += weight
                bearish_count += 1
                meta["zscore"] = f"bearish ({zscore:.2f})"

        # 3. RSI extreme (deeper thresholds than momentum)
        rsi = curr.get("rsi")
        if pd.notna(rsi):
            if rsi < 25:
                bullish_weight += 0.7
                bullish_count += 1
                meta["rsi"] = "bullish"
            elif rsi > 75:
                bearish_weight += 0.7
                bearish_count += 1
                meta["rsi"] = "bearish"

        total_confirmations = max(bullish_count, bearish_count)
        if total_confirmations < self.min_confirmations:
            return StrategySignal(SignalType.HOLD, 0.0, 0, len(self.SUB_SIGNALS), meta)

        signal_type, confidence = self._calculate_confidence(
            bullish_weight, bearish_weight, self.TOTAL_WEIGHT,
        )
        return StrategySignal(
            signal_type=signal_type,
            confidence=confidence,
            confirmations=total_confirmations,
            total_checks=len(self.SUB_SIGNALS),
            metadata=meta,
        )
