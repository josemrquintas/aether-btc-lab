"""Funding rate + volume signal strategy (crypto-specific).

Combines crypto-native funding rate data with volume analysis. Funding
rates are a unique sentiment indicator: when funding is extremely positive,
the futures market is over-leveraged long — this historically precedes
corrections. This strategy has NO equivalent in equities.
"""

import pandas as pd

from aether_btc.signals.base import SignalStrategy, SignalType, StrategySignal


class FundingVolumeStrategy(SignalStrategy):
    """Evaluate funding rate (contrarian), volume spikes, OBV, MFI, and CMF."""

    SUB_SIGNALS = [
        ("funding_contrarian", 0.8),
        ("volume_spike", 0.7),
        ("obv_trend", 0.7),
        ("mfi_reversal", 0.6),
        ("cmf_direction", 0.5),
    ]
    TOTAL_WEIGHT = sum(w for _, w in SUB_SIGNALS)

    @property
    def name(self) -> str:
        return "funding_volume"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def min_confirmations(self) -> int:
        return 2

    def get_required_columns(self) -> list[str]:
        return ["funding_rate", "volume", "volume_ratio", "obv", "mfi", "cmf"]

    def generate_signal(self, df: pd.DataFrame, bar_idx: int) -> StrategySignal:
        if bar_idx < 20:
            return StrategySignal(SignalType.HOLD, 0.0, 0, len(self.SUB_SIGNALS))

        curr = df.iloc[bar_idx]
        prev = df.iloc[bar_idx - 1] if bar_idx >= 1 else curr
        bullish_weight = 0.0
        bearish_weight = 0.0
        bullish_count = 0
        bearish_count = 0
        meta: dict = {}

        # 1. Funding rate CONTRARIAN signal
        funding = curr.get("funding_rate")
        if pd.notna(funding):
            if funding < -0.0005:
                # Shorts paying = overcrowded shorts = bullish
                bullish_weight += 0.8
                bullish_count += 1
                meta["funding"] = f"bullish (rate={funding:.6f})"
            elif funding > 0.0005:
                # Longs paying = overcrowded longs = bearish
                bearish_weight += 0.8
                bearish_count += 1
                meta["funding"] = f"bearish (rate={funding:.6f})"

        # 2. Volume spike with direction
        vol_ratio = curr.get("volume_ratio")
        close = curr.get("close")
        open_price = curr.get("open")
        if pd.notna(vol_ratio) and pd.notna(close) and pd.notna(open_price):
            if vol_ratio > 2.0:
                if close > open_price:
                    bullish_weight += 0.7
                    bullish_count += 1
                    meta["volume"] = "bullish"
                elif close < open_price:
                    bearish_weight += 0.7
                    bearish_count += 1
                    meta["volume"] = "bearish"

        # 3. OBV trend (new high/low over 20 bars)
        obv_curr = curr.get("obv")
        if pd.notna(obv_curr) and bar_idx >= 20:
            start_idx = max(0, bar_idx - 20)
            obv_slice = df.iloc[start_idx:bar_idx]["obv"]
            obv_valid = obv_slice.dropna()
            if len(obv_valid) > 0:
                if obv_curr > obv_valid.max():
                    bullish_weight += 0.7
                    bullish_count += 1
                    meta["obv"] = "bullish"
                elif obv_curr < obv_valid.min():
                    bearish_weight += 0.7
                    bearish_count += 1
                    meta["obv"] = "bearish"

        # 4. MFI reversal (crossing thresholds)
        mfi_curr = curr.get("mfi")
        mfi_prev = prev.get("mfi")
        if pd.notna(mfi_curr) and pd.notna(mfi_prev):
            if mfi_prev < 20 and mfi_curr >= 20:
                bullish_weight += 0.6
                bullish_count += 1
                meta["mfi"] = "bullish"
            elif mfi_prev > 80 and mfi_curr <= 80:
                bearish_weight += 0.6
                bearish_count += 1
                meta["mfi"] = "bearish"

        # 5. CMF direction
        cmf_val = curr.get("cmf")
        if pd.notna(cmf_val):
            if cmf_val > 0.05:
                bullish_weight += 0.5
                bullish_count += 1
                meta["cmf"] = "bullish"
            elif cmf_val < -0.05:
                bearish_weight += 0.5
                bearish_count += 1
                meta["cmf"] = "bearish"

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
