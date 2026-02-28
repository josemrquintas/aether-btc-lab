"""Core enumerations for Aether BTC."""

from enum import Enum


class SignalType(Enum):
    """Signal direction."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    SHORT = "SHORT"
    COVER = "COVER"


class PositionSide(Enum):
    """Position direction."""

    LONG = "LONG"
    SHORT = "SHORT"


class OrderIntent(Enum):
    """Order intent - distinguishes opening/closing for both sides."""

    OPEN_LONG = "OPEN_LONG"
    CLOSE_LONG = "CLOSE_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    CLOSE_SHORT = "CLOSE_SHORT"


class OrderType(Enum):
    """Order type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"


class OrderStatus(Enum):
    """Order lifecycle status."""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class ExitReason(Enum):
    """Reason for closing a position."""

    SIGNAL = "SIGNAL"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    LIQUIDATION = "LIQUIDATION"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    MANUAL = "MANUAL"
    END_OF_BACKTEST = "END_OF_BACKTEST"


class SlippageModel(Enum):
    """Slippage model for backtesting."""

    FIXED_PCT = "fixed_pct"
    VOLATILITY_SCALED = "volatility_scaled"


class MarketRegime(Enum):
    """Market regime for adaptive strategies."""

    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"


class SignalState(Enum):
    """Signal lifecycle state."""

    GENERATED = "GENERATED"
    QUEUED = "QUEUED"
    EXECUTED = "EXECUTED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"
