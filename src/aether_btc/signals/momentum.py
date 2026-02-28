"""Momentum signal strategy.

Detects when multiple momentum indicators simultaneously confirm
directional conviction. Momentum works well in crypto's trend-heavy
microstructure on 15-minute bars.
"""

import pandas as pd

from aether_btc.signals.base import SignalStrategy, SignalType, StrategySignal


class MomentumStrategy(SignalStrategy):
    """Evaluate RSI, MACD, Stochastic, ROC, and CCI for momentum signals."""

    SUB_SIGNALS = [
        ("rsi_reversal", 0.8),
        ("macd_crossover", 0.7),
        ("stoch_crossover", 0.6),
        ("roc_direction", 0.5),
        ("cci_reversal", 0.6),
    ]
    TOTAL_WEIGHT = sum(w for _, w in SUB_SIGNALS)

    @property
    def name(self) -> str:
        return "momentum"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def min_confirmations(self) -> int:
        return 2

    def get_required_columns(self) -> list[str]:
        return ["rsi", "macd_histogram", "stoch_k", "stoch_d", "roc", "cci"]

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

        # 1. RSI reversal
        rsi_curr = curr.get("rsi")
        rsi_prev = prev.get("rsi")
        if pd.notna(rsi_curr) and pd.notna(rsi_prev):
            if rsi_curr < 30 and rsi_curr > rsi_prev:
                bullish_weight += 0.8
                bullish_count += 1
                meta["rsi"] = "bullish"
            elif rsi_curr > 70 and rsi_curr < rsi_prev:
                bearish_weight += 0.8
                bearish_count += 1
                meta["rsi"] = "bearish"

        # 2. MACD histogram crossover
        macd_curr = curr.get("macd_histogram")
        macd_prev = prev.get("macd_histogram")
        if pd.notna(macd_curr) and pd.notna(macd_prev):
            if macd_prev <= 0 and macd_curr > 0:
                bullish_weight += 0.7
                bullish_count += 1
                meta["macd"] = "bullish"
            elif macd_prev >= 0 and macd_curr < 0:
                bearish_weight += 0.7
                bearish_count += 1
                meta["macd"] = "bearish"

        # 3. Stochastic crossover in extreme zones
        k_curr = curr.get("stoch_k")
        k_prev = prev.get("stoch_k")
        d_curr = curr.get("stoch_d")
        d_prev = prev.get("stoch_d")
        if all(pd.notna(v) for v in [k_curr, k_prev, d_curr, d_prev]):
            if k_prev < d_prev and k_curr > d_curr and k_curr < 20:
                bullish_weight += 0.6
                bullish_count += 1
                meta["stoch"] = "bullish"
            elif k_prev > d_prev and k_curr < d_curr and k_curr > 80:
                bearish_weight += 0.6
                bearish_count += 1
                meta["stoch"] = "bearish"

        # 4. ROC direction
        roc_curr = curr.get("roc")
        roc_prev = prev.get("roc")
        if pd.notna(roc_curr) and pd.notna(roc_prev):
            if roc_curr > 0 and roc_curr > roc_prev:
                bullish_weight += 0.5
                bullish_count += 1
                meta["roc"] = "bullish"
            elif roc_curr < 0 and roc_curr < roc_prev:
                bearish_weight += 0.5
                bearish_count += 1
                meta["roc"] = "bearish"

        # 5. CCI reversal
        cci_curr = curr.get("cci")
        cci_prev = prev.get("cci")
        if pd.notna(cci_curr) and pd.notna(cci_prev):
            if cci_prev < -100 and cci_curr >= -100:
                bullish_weight += 0.6
                bullish_count += 1
                meta["cci"] = "bullish"
            elif cci_prev > 100 and cci_curr <= 100:
                bearish_weight += 0.6
                bearish_count += 1
                meta["cci"] = "bearish"

        # Determine signal
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
