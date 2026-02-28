"""Portfolio manager with leverage margin tracking for crypto.

Adapted from Aether v3 with crypto-specific margin model:
- No Reg T rules — simple isolated margin per position
- Leverage determines margin (margin = notional / leverage)
- Liquidation check per position
"""

from datetime import datetime

import structlog

from aether_btc.core.enums import ExitReason, PositionSide
from aether_btc.core.models import ClosedPosition, Position

log = structlog.get_logger()


class PortfolioManager:
    """Manages leveraged crypto positions with isolated margin."""

    def __init__(self, initial_capital: float) -> None:
        self.initial_capital = initial_capital
        self._cash = initial_capital
        self._positions: dict[str, Position] = {}
        self._closed_trades: list[ClosedPosition] = []

    @property
    def positions(self) -> dict[str, Position]:
        return self._positions.copy()

    @property
    def closed_trades(self) -> list[ClosedPosition]:
        return self._closed_trades.copy()

    @property
    def position_count(self) -> int:
        return len(self._positions)

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def total_margin_in_use(self) -> float:
        return sum(p.margin_used for p in self._positions.values())

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.net_unrealized_pnl for p in self._positions.values())

    @property
    def equity(self) -> float:
        """Total equity = cash + unrealized PnL (accounting for leverage)."""
        return self._cash + self.total_unrealized_pnl

    @property
    def leverage_ratio(self) -> float:
        """Current gross leverage = total notional / equity."""
        total_notional = sum(p.current_notional for p in self._positions.values())
        eq = self.equity
        if eq <= 0:
            return 0.0
        return total_notional / eq

    @property
    def gross_long_exposure(self) -> float:
        return sum(
            p.current_notional for p in self._positions.values()
            if p.side == PositionSide.LONG
        )

    @property
    def gross_short_exposure(self) -> float:
        return sum(
            p.current_notional for p in self._positions.values()
            if p.side == PositionSide.SHORT
        )

    @property
    def net_exposure(self) -> float:
        return self.gross_long_exposure - self.gross_short_exposure

    def has_position(self, pair: str) -> bool:
        return pair in self._positions

    def get_position(self, pair: str) -> Position | None:
        return self._positions.get(pair)

    def open_position(
        self,
        pair: str,
        quantity: float,
        entry_price: float,
        side: PositionSide,
        leverage: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        commission: float = 0.0,
        opened_at: datetime | None = None,
    ) -> Position | None:
        """Open a new leveraged position. Returns None if insufficient margin."""
        if pair in self._positions:
            log.warning("position_already_exists", pair=pair)
            return None

        margin_required = quantity * entry_price / leverage
        if margin_required + commission > self._cash:
            log.warning("insufficient_margin", pair=pair, required=margin_required, available=self._cash)
            return None

        position = Position(
            pair=pair,
            quantity=quantity,
            entry_price=entry_price,
            side=side,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=entry_price,
            opened_at=opened_at or datetime.now(),
            entry_commission=commission,
        )

        # Deduct margin + commission from cash
        self._cash -= (margin_required + commission)
        self._positions[pair] = position

        log.info(
            "opened_position",
            pair=pair,
            side=side.value,
            leverage=leverage,
            quantity=quantity,
            entry_price=entry_price,
            margin=margin_required,
        )
        return position

    def close_position(
        self,
        pair: str,
        exit_price: float,
        exit_reason: ExitReason,
        commission: float = 0.0,
        slippage: float = 0.0,
        closed_at: datetime | None = None,
    ) -> ClosedPosition | None:
        """Close an existing position. Returns ClosedPosition with NET P&L."""
        if pair not in self._positions:
            log.warning("no_position_to_close", pair=pair)
            return None

        position = self._positions[pair]
        close_time = closed_at or datetime.now()

        closed = ClosedPosition(
            pair=pair,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=exit_price,
            side=position.side,
            leverage=position.leverage,
            opened_at=position.opened_at,
            closed_at=close_time,
            exit_reason=exit_reason,
            entry_commission=position.entry_commission,
            exit_commission=commission,
            entry_slippage=0.0,
            exit_slippage=slippage,
            funding_cost=position.funding_cost_accrued,
        )

        # Return margin + PnL to cash (minus exit costs)
        margin_return = position.margin_used + closed.gross_pnl - commission - slippage
        # For liquidation, margin is lost
        if exit_reason == ExitReason.LIQUIDATION:
            margin_return = 0.0

        self._cash += margin_return
        self._closed_trades.append(closed)
        del self._positions[pair]

        log.info(
            "closed_position",
            pair=pair,
            exit_reason=exit_reason.value,
            net_pnl=closed.net_pnl,
            hold_hours=closed.hold_duration_hours,
        )
        return closed

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current prices for all positions."""
        for pair, price in prices.items():
            if pair in self._positions:
                self._positions[pair].update_price(price)

    def accrue_funding_costs(self, funding_rates: dict[str, float]) -> float:
        """Accrue funding costs for all positions. Returns total cost."""
        total = 0.0
        for pair, rate in funding_rates.items():
            if pair in self._positions:
                cost = self._positions[pair].accrue_funding_cost(rate)
                self._cash -= cost  # Deduct from cash
                total += cost
        return total

    def check_liquidations(self, prices: dict[str, float]) -> list[str]:
        """Check which positions would be liquidated at given prices.

        Returns list of pair names that should be liquidated.
        """
        liquidated = []
        for pair, price in prices.items():
            pos = self._positions.get(pair)
            if pos and pos.is_liquidated(price):
                liquidated.append(pair)
        return liquidated

    def get_positions_by_loss(self) -> list[tuple[str, Position]]:
        """Get positions sorted by loss (biggest losers first)."""
        items = list(self._positions.items())
        items.sort(key=lambda x: x[1].net_unrealized_pnl)
        return items

    def get_summary(self) -> dict:
        return {
            "cash": self.cash,
            "equity": self.equity,
            "position_count": self.position_count,
            "leverage_ratio": self.leverage_ratio,
            "total_margin_in_use": self.total_margin_in_use,
            "gross_long_exposure": self.gross_long_exposure,
            "gross_short_exposure": self.gross_short_exposure,
            "total_unrealized_pnl": self.total_unrealized_pnl,
        }
