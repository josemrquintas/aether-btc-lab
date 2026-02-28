"""Tests for core models — Phase 1 Data Integrity Gate."""

from datetime import datetime

import pytest

from aether_btc.core.enums import ExitReason, PositionSide, SignalType
from aether_btc.core.models import ClosedPosition, Position, Signal


class TestPositionLeveragePnL:
    """Test that leverage correctly multiplies P&L."""

    def test_position_leverage_pnl_long(self):
        """Long at $50k, 10x leverage, price +1% -> PnL = +10% of margin."""
        pos = Position(
            pair="BTCUSDT", quantity=0.1, entry_price=50_000,
            side=PositionSide.LONG, leverage=10.0, current_price=50_500,
        )
        # Margin = 0.1 * 50000 / 10 = 500
        assert pos.margin_used == pytest.approx(500.0)
        # Price change = +1%, leveraged PnL = 0.01 * 10 * 500 = 50
        # That's 10% of margin (50/500 = 10%)
        assert pos.gross_unrealized_pnl == pytest.approx(50.0)
        assert pos.gross_unrealized_pnl / pos.margin_used == pytest.approx(0.10)

    def test_position_leverage_pnl_short(self):
        """Short at $50k, 5x leverage, price -2% -> PnL = +10% of margin."""
        pos = Position(
            pair="BTCUSDT", quantity=0.1, entry_price=50_000,
            side=PositionSide.SHORT, leverage=5.0, current_price=49_000,
        )
        # Margin = 0.1 * 50000 / 5 = 1000
        assert pos.margin_used == pytest.approx(1000.0)
        # Price change = -2%, short profits, leveraged PnL = 2% * 5 * 1000 = 100
        assert pos.gross_unrealized_pnl == pytest.approx(100.0)

    def test_position_leverage_loss_short(self):
        """Short at $50k, 5x leverage, price +2% -> PnL = -10% of margin."""
        pos = Position(
            pair="BTCUSDT", quantity=0.1, entry_price=50_000,
            side=PositionSide.SHORT, leverage=5.0, current_price=51_000,
        )
        assert pos.gross_unrealized_pnl == pytest.approx(-100.0)


class TestLiquidationPrice:
    """Test liquidation price calculations."""

    def test_liquidation_price_long(self):
        """10x leverage long -> liquidation at ~90% of entry."""
        pos = Position(
            pair="BTCUSDT", quantity=0.1, entry_price=50_000,
            side=PositionSide.LONG, leverage=10.0,
        )
        liq = pos.liquidation_price
        # liq = 50000 * (1 - 1/10 + 0.004) = 50000 * 0.904 = 45200
        assert liq == pytest.approx(45_200.0)
        assert liq < 50_000  # Below entry for long

    def test_liquidation_price_short(self):
        """10x leverage short -> liquidation at ~110% of entry."""
        pos = Position(
            pair="BTCUSDT", quantity=0.1, entry_price=50_000,
            side=PositionSide.SHORT, leverage=10.0,
        )
        liq = pos.liquidation_price
        # liq = 50000 * (1 + 1/10 - 0.004) = 50000 * 1.096 = 54800
        assert liq == pytest.approx(54_800.0)
        assert liq > 50_000  # Above entry for short

    def test_is_liquidated_long(self):
        """Long position is liquidated when price falls below liquidation price."""
        pos = Position(
            pair="BTCUSDT", quantity=0.1, entry_price=50_000,
            side=PositionSide.LONG, leverage=10.0,
        )
        assert not pos.is_liquidated(46_000)  # Above liquidation
        assert pos.is_liquidated(45_000)  # Below liquidation

    def test_is_liquidated_short(self):
        """Short position is liquidated when price rises above liquidation price."""
        pos = Position(
            pair="BTCUSDT", quantity=0.1, entry_price=50_000,
            side=PositionSide.SHORT, leverage=10.0,
        )
        assert not pos.is_liquidated(54_000)  # Below liquidation
        assert pos.is_liquidated(55_000)  # Above liquidation


class TestFundingCost:
    """Test funding cost accrual."""

    def test_funding_cost_accrual(self):
        """0.01% funding rate on $100k notional = $10 per 8h cycle."""
        pos = Position(
            pair="BTCUSDT", quantity=1.0, entry_price=100_000,
            side=PositionSide.LONG, leverage=10.0,
        )
        # Notional = 1.0 * 100000 = 100000
        cost = pos.accrue_funding_cost(0.0001)  # 0.01%
        # Long pays when rate is positive: cost = 100000 * 0.0001 = 10
        assert cost == pytest.approx(10.0)
        assert pos.funding_cost_accrued == pytest.approx(10.0)

    def test_funding_cost_short_receives(self):
        """Short receives funding when rate is positive."""
        pos = Position(
            pair="BTCUSDT", quantity=1.0, entry_price=100_000,
            side=PositionSide.SHORT, leverage=10.0,
        )
        cost = pos.accrue_funding_cost(0.0001)
        # Short receives: cost = -(100000 * 0.0001) = -10
        assert cost == pytest.approx(-10.0)
        assert pos.funding_cost_accrued == pytest.approx(-10.0)

    def test_funding_cost_negative_rate(self):
        """Negative funding rate: shorts pay, longs receive."""
        pos = Position(
            pair="BTCUSDT", quantity=1.0, entry_price=100_000,
            side=PositionSide.LONG, leverage=5.0,
        )
        cost = pos.accrue_funding_cost(-0.0001)
        # Long receives when rate is negative: cost = 100000 * (-0.0001) = -10
        assert cost == pytest.approx(-10.0)


class TestSignalEnums:
    """Test signal types serialize/deserialize correctly."""

    def test_signal_types(self):
        for st in [SignalType.BUY, SignalType.SELL, SignalType.SHORT, SignalType.COVER]:
            assert SignalType(st.value) == st

    def test_signal_creation(self):
        sig = Signal(
            pair="BTCUSDT", signal_type=SignalType.BUY,
            confidence=0.75, timestamp=datetime.now(),
        )
        d = sig.to_dict()
        assert d["signal_type"] == "BUY"
        assert d["confidence"] == 0.75

    def test_signal_confidence_validation(self):
        with pytest.raises(ValueError):
            Signal(pair="BTCUSDT", signal_type=SignalType.BUY,
                   confidence=1.5, timestamp=datetime.now())


class TestClosedPositionNetPnl:
    """Test closed position NET P&L deducts all costs."""

    def test_closed_position_net_pnl(self):
        """net_pnl deducts commission + funding + slippage."""
        closed = ClosedPosition(
            pair="BTCUSDT", quantity=0.1, entry_price=50_000, exit_price=51_000,
            side=PositionSide.LONG, leverage=5.0,
            opened_at=datetime.now(), closed_at=datetime.now(),
            exit_reason=ExitReason.SIGNAL,
            entry_commission=2.0, exit_commission=2.04,
            entry_slippage=0.5, exit_slippage=0.51,
            funding_cost=3.0,
        )
        # Gross PnL: (51000-50000)/50000 * 5 * (0.1*50000/5) = 0.02 * 5 * 1000 = 100
        assert closed.gross_pnl == pytest.approx(100.0)
        # Costs: 2.0 + 2.04 + 0.5 + 0.51 + 3.0 = 8.05
        assert closed.total_costs == pytest.approx(8.05)
        # Net: 100 - 8.05 = 91.95
        assert closed.net_pnl == pytest.approx(91.95)

    def test_closed_position_losing_trade(self):
        """Losing trade with costs makes loss larger."""
        closed = ClosedPosition(
            pair="BTCUSDT", quantity=0.1, entry_price=50_000, exit_price=49_000,
            side=PositionSide.LONG, leverage=5.0,
            opened_at=datetime.now(), closed_at=datetime.now(),
            exit_reason=ExitReason.STOP_LOSS,
            entry_commission=2.0, exit_commission=1.96,
            funding_cost=1.0,
        )
        # Gross PnL: -2% * 5 * 1000 = -100
        assert closed.gross_pnl == pytest.approx(-100.0)
        # Net PnL = -100 - 4.96 = -104.96
        assert closed.net_pnl == pytest.approx(-104.96)
