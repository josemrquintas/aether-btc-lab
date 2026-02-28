"""Signal strategies for Aether BTC.

Provides 5 signal strategies that pre-compute directional signals
with confidence scores from technical indicators. The GA learns
to weight these alongside raw indicator features.
"""

from aether_btc.signals.base import SignalStrategy, SignalType, StrategySignal
from aether_btc.signals.funding_volume import FundingVolumeStrategy
from aether_btc.signals.mean_reversion import MeanReversionStrategy
from aether_btc.signals.momentum import MomentumStrategy
from aether_btc.signals.runner import ALL_STRATEGY_FEATURE_COLUMNS, StrategyRunner
from aether_btc.signals.trend_following import TrendFollowingStrategy
from aether_btc.signals.volatility_breakout import VolatilityBreakoutStrategy

STRATEGIES: list[SignalStrategy] = [
    MomentumStrategy(),
    MeanReversionStrategy(),
    TrendFollowingStrategy(),
    VolatilityBreakoutStrategy(),
    FundingVolumeStrategy(),
]

__all__ = [
    "SignalStrategy",
    "SignalType",
    "StrategySignal",
    "StrategyRunner",
    "STRATEGIES",
    "ALL_STRATEGY_FEATURE_COLUMNS",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "TrendFollowingStrategy",
    "VolatilityBreakoutStrategy",
    "FundingVolumeStrategy",
]
