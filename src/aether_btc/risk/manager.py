"""Crypto risk manager with leverage calculation, liquidation buffer, and circuit breaker.

Adapted from Aether v3 risk/manager.py for leveraged crypto trading.
Key differences:
- Leverage calculation based on GA genes (confidence, volatility, caps)
- Liquidation buffer enforcement
- Position sizing risks % of account per trade (not per notional)
- Tighter circuit breaker (10% vs 20%)
"""

from dataclasses import dataclass
from enum import Enum

import structlog

from aether_btc.core.enums import PositionSide
from aether_btc.core.models import MAINTENANCE_MARGIN_RATE

log = structlog.get_logger()


class PositionSizingMethod(Enum):
    FIXED_FRACTION = "fixed_fraction"
    ATR_BASED = "atr_based"
    EQUAL_WEIGHT = "equal_weight"


class StopLossMethod(Enum):
    FIXED_PERCENT = "fixed_percent"
    ATR_MULTIPLE = "atr_multiple"


@dataclass
class RiskLimits:
    """Portfolio risk limits for crypto trading."""

    max_risk_per_trade: float = 0.02  # Risk 2% of account per trade
    max_drawdown: float = 0.10  # Circuit breaker at 10% drawdown
    max_positions: int = 3
    max_position_size: float = 0.50  # Max 50% of capital per position margin
    min_position_size: float = 0.02  # Min 2% of capital
    max_leverage: float = 20.0  # Hard cap
    min_liquidation_buffer: float = 0.15  # Min 15% distance to liquidation


@dataclass
class PositionSizeResult:
    quantity: float
    margin_value: float
    margin_percent: float
    risk_amount: float
    risk_percent: float
    leverage: float
    method: PositionSizingMethod
    capped: bool = False
    reason: str | None = None


@dataclass
class StopLevels:
    stop_loss: float
    take_profit: float
    stop_distance_percent: float
    target_distance_percent: float
    risk_reward_ratio: float
    atr_value: float | None = None


class RiskManager:
    """Manages risk with GA-evolved leverage and crypto-specific limits."""

    def __init__(
        self,
        capital: float,
        limits: RiskLimits | None = None,
        default_sizing_method: PositionSizingMethod = PositionSizingMethod.ATR_BASED,
        default_stop_method: StopLossMethod = StopLossMethod.ATR_MULTIPLE,
    ) -> None:
        self.capital = capital
        self.limits = limits or RiskLimits()
        self.default_sizing_method = default_sizing_method
        self.default_stop_method = default_stop_method
        self._peak_capital = capital
        self._current_drawdown = 0.0

    def update_capital(self, capital: float) -> None:
        self.capital = capital
        if capital > self._peak_capital:
            self._peak_capital = capital
        self._current_drawdown = (self._peak_capital - capital) / self._peak_capital

    @property
    def current_drawdown(self) -> float:
        return self._current_drawdown

    @property
    def circuit_breaker_triggered(self) -> bool:
        return self._current_drawdown >= self.limits.max_drawdown

    def calculate_leverage(
        self,
        base_leverage: float,
        confidence: float,
        confidence_scale: float,
        volatility: float,
        volatility_dampen: float,
        max_cap: float,
    ) -> float:
        """Calculate leverage for a trade based on GA-evolved parameters.

        leverage = base * (1 + confidence_scale * confidence) * (1 - volatility_dampen * volatility)
        Clamped to [1.0, min(max_cap, limits.max_leverage)]
        """
        lev = base_leverage * (1 + confidence_scale * confidence)

        # Dampen by volatility (higher vol = lower leverage)
        vol_factor = max(0.1, 1.0 - volatility_dampen * min(volatility, 1.0))
        lev *= vol_factor

        # Clamp
        hard_cap = min(max_cap, self.limits.max_leverage)
        lev = max(1.0, min(lev, hard_cap))
        return round(lev, 1)

    def calculate_liquidation_price(
        self,
        entry_price: float,
        leverage: float,
        side: PositionSide,
    ) -> float:
        """Calculate liquidation price given entry, leverage, and side."""
        mm = MAINTENANCE_MARGIN_RATE
        if side == PositionSide.LONG:
            return entry_price * (1 - 1 / leverage + mm)
        return entry_price * (1 + 1 / leverage - mm)

    def check_liquidation_buffer(
        self,
        entry_price: float,
        current_price: float,
        leverage: float,
        side: PositionSide,
        buffer_pct: float | None = None,
    ) -> bool:
        """Check if current price is sufficiently far from liquidation.

        Returns True if position is safe (enough buffer).
        """
        buffer = buffer_pct or self.limits.min_liquidation_buffer
        liq_price = self.calculate_liquidation_price(entry_price, leverage, side)

        if side == PositionSide.LONG:
            distance = (current_price - liq_price) / current_price
        else:
            distance = (liq_price - current_price) / current_price

        return distance >= buffer

    def calculate_position_size(
        self,
        entry_price: float,
        leverage: float,
        side: PositionSide = PositionSide.LONG,
        stop_loss: float | None = None,
        atr: float | None = None,
        method: PositionSizingMethod | None = None,
        current_positions: int = 0,
    ) -> PositionSizeResult:
        """Calculate position size. Risks % of account per trade, not per notional."""
        method = method or self.default_sizing_method

        if self.circuit_breaker_triggered:
            return PositionSizeResult(
                quantity=0, margin_value=0, margin_percent=0,
                risk_amount=0, risk_percent=0, leverage=leverage,
                method=method, capped=True,
                reason="Circuit breaker: max drawdown reached",
            )

        if current_positions >= self.limits.max_positions:
            return PositionSizeResult(
                quantity=0, margin_value=0, margin_percent=0,
                risk_amount=0, risk_percent=0, leverage=leverage,
                method=method, capped=True,
                reason=f"Max positions ({self.limits.max_positions}) reached",
            )

        # Calculate risk-based margin
        risk_amount = self.capital * self.limits.max_risk_per_trade

        if method == PositionSizingMethod.ATR_BASED and stop_loss is not None:
            stop_distance = abs(entry_price - stop_loss) / entry_price
            if stop_distance > 0:
                # margin = risk / (stop_distance * leverage)
                margin_value = risk_amount / (stop_distance * leverage)
            else:
                margin_value = self.capital * self.limits.max_position_size
        elif method == PositionSizingMethod.EQUAL_WEIGHT:
            margin_value = self.capital / self.limits.max_positions
        else:
            margin_value = self.capital * self.limits.max_position_size

        # Apply limits
        capped = False
        reason = None

        max_margin = self.capital * self.limits.max_position_size
        if margin_value > max_margin:
            margin_value = max_margin
            capped = True
            reason = "Capped at max position size"

        min_margin = self.capital * self.limits.min_position_size
        if margin_value < min_margin:
            return PositionSizeResult(
                quantity=0, margin_value=0, margin_percent=0,
                risk_amount=0, risk_percent=0, leverage=leverage,
                method=method, capped=True,
                reason="Below minimum position size",
            )

        # Can't exceed available cash
        if margin_value > self.capital:
            margin_value = self.capital
            capped = True
            reason = "Limited by available capital"

        # Calculate quantity from margin
        notional = margin_value * leverage
        quantity = notional / entry_price if entry_price > 0 else 0

        margin_percent = margin_value / self.capital if self.capital > 0 else 0
        risk_percent = risk_amount / self.capital if self.capital > 0 else 0

        return PositionSizeResult(
            quantity=quantity,
            margin_value=margin_value,
            margin_percent=margin_percent,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
            leverage=leverage,
            method=method,
            capped=capped,
            reason=reason,
        )

    def calculate_stop_levels(
        self,
        entry_price: float,
        side: PositionSide = PositionSide.LONG,
        atr: float | None = None,
        method: StopLossMethod | None = None,
        atr_multiplier: float = 2.0,
        fixed_stop_percent: float = 0.02,
        risk_reward_ratio: float = 2.0,
    ) -> StopLevels:
        """Calculate stop/target levels for both LONG and SHORT."""
        method = method or self.default_stop_method

        if method == StopLossMethod.ATR_MULTIPLE and atr is not None:
            stop_distance = atr * atr_multiplier
        else:
            stop_distance = entry_price * fixed_stop_percent

        if side == PositionSide.LONG:
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + stop_distance * risk_reward_ratio
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - stop_distance * risk_reward_ratio

        stop_pct = stop_distance / entry_price if entry_price > 0 else 0
        target_distance = stop_distance * risk_reward_ratio
        target_pct = target_distance / entry_price if entry_price > 0 else 0

        return StopLevels(
            stop_loss=stop_loss,
            take_profit=take_profit,
            stop_distance_percent=stop_pct,
            target_distance_percent=target_pct,
            risk_reward_ratio=risk_reward_ratio,
            atr_value=atr,
        )
