"""Signal strategy base classes.

Defines the ABC contract for signal strategies and the StrategySignal
output dataclass. Each strategy analyzes a subset of indicators and
produces a directional signal with confidence score.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class SignalType(Enum):
    """Signal direction from a strategy."""
    BUY = "BUY"
    SHORT = "SHORT"
    HOLD = "HOLD"


@dataclass
class StrategySignal:
    """Output from a signal strategy evaluation."""
    signal_type: SignalType
    confidence: float  # [0, 1] — how many sub-indicators agree
    confirmations: int  # sub-indicators that fired in winning direction
    total_checks: int  # total sub-indicators evaluated
    metadata: dict = field(default_factory=dict)


class SignalStrategy(ABC):
    """Abstract base class for signal strategies.

    Each strategy evaluates a subset of indicators at a given bar
    and produces a StrategySignal with direction and confidence.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this strategy."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Semver version string."""

    @property
    @abstractmethod
    def min_confirmations(self) -> int:
        """Minimum sub-signals needed to fire a BUY/SHORT."""

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, bar_idx: int) -> StrategySignal:
        """Evaluate a single bar and return a signal.

        CRITICAL: May only access df.iloc[:bar_idx+1] — no future data.
        Crossover detection compares df.iloc[bar_idx] vs df.iloc[bar_idx-1].
        """

    @abstractmethod
    def get_required_columns(self) -> list[str]:
        """Indicator columns needed by this strategy."""

    def _calculate_confidence(
        self,
        bullish_count: float,
        bearish_count: float,
        total_weight: float,
    ) -> tuple[SignalType, float]:
        """Determine signal direction and confidence from weighted sub-signal counts.

        Returns (signal_type, confidence) where confidence = agreeing_weight / total_weight.
        """
        if total_weight == 0:
            return SignalType.HOLD, 0.0

        if bullish_count > bearish_count:
            confirmations = bullish_count
            signal = SignalType.BUY
        elif bearish_count > bullish_count:
            confirmations = bearish_count
            signal = SignalType.SHORT
        else:
            return SignalType.HOLD, 0.0

        confidence = min(confirmations / total_weight, 1.0)
        return signal, confidence
