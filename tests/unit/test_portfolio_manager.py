"""Tests for portfolio manager — Phase 2 Backtest Correctness Gate."""

import pytest

from aether_btc.core.enums import ExitReason, PositionSide
from aether_btc.portfolio.manager import PortfolioManager


class TestMarginTracking:
    """Test margin tracking with leverage."""

    def test_margin_tracking(self):
        """Opening 10x position with $100 margin -> $1000 notional, $100 margin reserved."""
        pm = PortfolioManager(initial_capital=1000)
        pos = pm.open_position(
            pair="BTCUSDT", quantity=0.02, entry_price=50_000,
            side=PositionSide.LONG, leverage=10.0,
        )
        assert pos is not None
        # Margin = 0.02 * 50000 / 10 = 100
        assert pos.margin_used == pytest.approx(100.0)
        # Cash should be reduced by margin
        assert pm.cash == pytest.approx(900.0)

    def test_multiple_positions_margin(self):
        """Two open positions use correct total margin."""
        pm = PortfolioManager(initial_capital=2000)
        pm.open_position(
            pair="BTCUSDT", quantity=0.02, entry_price=50_000,
            side=PositionSide.LONG, leverage=10.0,
        )
        pm.open_position(
            pair="ETHUSDT", quantity=0.5, entry_price=3_000,
            side=PositionSide.LONG, leverage=5.0,
        )
        # BTC margin: 0.02 * 50000 / 10 = 100
        # ETH margin: 0.5 * 3000 / 5 = 300
        assert pm.total_margin_in_use == pytest.approx(400.0)
        assert pm.cash == pytest.approx(1600.0)

    def test_equity_calculation(self):
        """equity = cash + unrealized_pnl (accounting for leverage)."""
        pm = PortfolioManager(initial_capital=1000)
        pm.open_position(
            pair="BTCUSDT", quantity=0.02, entry_price=50_000,
            side=PositionSide.LONG, leverage=10.0,
        )
        # Cash = 1000 - 100 = 900
        # Update price: +1% -> unrealized = 0.02 * 500 = 10... but with leverage
        # PnL = (50500-50000)/50000 * 10 * 100 = 10
        pm.update_prices({"BTCUSDT": 50_500})
        # Equity = 900 + 10 = 910
        assert pm.equity == pytest.approx(910.0)


class TestPositionClose:
    """Test position closing returns margin + PnL."""

    def test_close_profitable_long(self):
        """Close profitable long returns margin + profit."""
        pm = PortfolioManager(initial_capital=1000)
        pm.open_position(
            pair="BTCUSDT", quantity=0.02, entry_price=50_000,
            side=PositionSide.LONG, leverage=10.0,
        )
        assert pm.cash == pytest.approx(900.0)

        closed = pm.close_position(
            pair="BTCUSDT", exit_price=51_000,
            exit_reason=ExitReason.SIGNAL,
        )
        assert closed is not None
        # Gross PnL = (51000-50000)/50000 * 10 * 100 = 20
        assert closed.gross_pnl == pytest.approx(20.0)
        # Cash should be initial + profit = 1000 + 20 = 1020
        assert pm.cash == pytest.approx(1020.0)

    def test_close_losing_short(self):
        """Close losing short returns margin - loss."""
        pm = PortfolioManager(initial_capital=1000)
        pm.open_position(
            pair="BTCUSDT", quantity=0.02, entry_price=50_000,
            side=PositionSide.SHORT, leverage=5.0,
        )
        # Margin = 0.02 * 50000 / 5 = 200, cash = 800

        closed = pm.close_position(
            pair="BTCUSDT", exit_price=51_000,
            exit_reason=ExitReason.STOP_LOSS,
        )
        # Gross PnL: short lost, (51000-50000)/50000 * 5 * 200 = 0.02 * 5 * 200 = -20
        # Price went up for short = loss
        assert closed.gross_pnl == pytest.approx(-20.0)
        # Cash = 800 + 200 - 20 = 980
        assert pm.cash == pytest.approx(980.0)

    def test_liquidation_loses_margin(self):
        """Liquidation returns $0 (margin is lost)."""
        pm = PortfolioManager(initial_capital=1000)
        pm.open_position(
            pair="BTCUSDT", quantity=0.02, entry_price=50_000,
            side=PositionSide.LONG, leverage=10.0,
        )
        # Cash = 900, margin = 100

        closed = pm.close_position(
            pair="BTCUSDT", exit_price=45_000,
            exit_reason=ExitReason.LIQUIDATION,
        )
        # Liquidation: margin_return = 0
        assert pm.cash == pytest.approx(900.0)  # Only get back what was NOT in margin


class TestFundingCosts:
    """Test funding cost accrual through portfolio."""

    def test_funding_cost_reduces_cash(self):
        """Funding cost for long reduces available cash."""
        pm = PortfolioManager(initial_capital=1000)
        pm.open_position(
            pair="BTCUSDT", quantity=0.1, entry_price=50_000,
            side=PositionSide.LONG, leverage=5.0,
        )
        initial_cash = pm.cash

        pm.accrue_funding_costs({"BTCUSDT": 0.0001})
        # Cost = 0.1 * 50000 * 0.0001 = 0.5 (long pays)
        assert pm.cash < initial_cash
