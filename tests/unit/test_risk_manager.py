"""Tests for risk manager — Phase 2 Backtest Correctness Gate."""

import pytest

from aether_btc.core.enums import PositionSide
from aether_btc.risk.manager import RiskLimits, RiskManager


class TestLeverageCalculation:
    """Test GA-evolved leverage calculation."""

    def test_leverage_calculation_high_confidence(self):
        """High confidence signal -> higher leverage."""
        rm = RiskManager(capital=1000)
        lev_high = rm.calculate_leverage(
            base_leverage=3.0, confidence=0.9,
            confidence_scale=1.0, volatility=0.02,
            volatility_dampen=0.5, max_cap=20.0,
        )
        lev_low = rm.calculate_leverage(
            base_leverage=3.0, confidence=0.3,
            confidence_scale=1.0, volatility=0.02,
            volatility_dampen=0.5, max_cap=20.0,
        )
        assert lev_high > lev_low

    def test_leverage_calculation_high_volatility(self):
        """High ATR -> leverage dampened down."""
        rm = RiskManager(capital=1000)
        lev_low_vol = rm.calculate_leverage(
            base_leverage=5.0, confidence=0.7,
            confidence_scale=1.0, volatility=0.01,
            volatility_dampen=0.5, max_cap=20.0,
        )
        lev_high_vol = rm.calculate_leverage(
            base_leverage=5.0, confidence=0.7,
            confidence_scale=1.0, volatility=0.5,
            volatility_dampen=0.5, max_cap=20.0,
        )
        assert lev_low_vol > lev_high_vol

    def test_leverage_never_exceeds_cap(self):
        """leverage_cap gene of 15 -> no trade gets >15x."""
        rm = RiskManager(capital=1000)
        lev = rm.calculate_leverage(
            base_leverage=10.0, confidence=1.0,
            confidence_scale=2.0, volatility=0.0,
            volatility_dampen=0.0, max_cap=15.0,
        )
        assert lev <= 15.0

    def test_leverage_never_below_1x(self):
        """Leverage is always >= 1x."""
        rm = RiskManager(capital=1000)
        lev = rm.calculate_leverage(
            base_leverage=1.0, confidence=0.0,
            confidence_scale=0.0, volatility=1.0,
            volatility_dampen=1.0, max_cap=20.0,
        )
        assert lev >= 1.0


class TestPositionSizing:
    """Test position sizing."""

    def test_position_size_risks_percent_of_account(self):
        """Risk 2% per trade, $1000 account -> max $20 risk per trade."""
        rm = RiskManager(capital=1000, limits=RiskLimits(max_risk_per_trade=0.02))
        result = rm.calculate_position_size(
            entry_price=50_000, leverage=5.0,
            stop_loss=49_000,
        )
        assert result.risk_amount == pytest.approx(20.0)

    def test_max_concurrent_positions(self):
        """With $1k, limit concurrent positions."""
        rm = RiskManager(capital=1000, limits=RiskLimits(max_positions=2))
        result = rm.calculate_position_size(
            entry_price=50_000, leverage=5.0,
            current_positions=2,
        )
        assert result.quantity == 0
        assert result.reason == "Max positions (2) reached"

    def test_circuit_breaker_blocks_sizing(self):
        """Circuit breaker prevents new positions."""
        rm = RiskManager(capital=1000)
        rm.update_capital(890)  # 11% drawdown > 10% limit
        assert rm.circuit_breaker_triggered
        result = rm.calculate_position_size(entry_price=50_000, leverage=5.0)
        assert result.quantity == 0


class TestLiquidationBuffer:
    """Test liquidation buffer enforcement."""

    def test_liquidation_buffer_enforcement(self):
        """Won't open trade if entry-to-liquidation distance < buffer."""
        rm = RiskManager(capital=1000)

        # 2x leverage: liquidation ~50% away -> should pass with 15% buffer
        safe = rm.check_liquidation_buffer(
            entry_price=50_000, current_price=50_000,
            leverage=2.0, side=PositionSide.LONG,
            buffer_pct=0.15,
        )
        assert safe

        # 20x leverage: liquidation ~5% away -> should fail with 15% buffer
        unsafe = rm.check_liquidation_buffer(
            entry_price=50_000, current_price=50_000,
            leverage=20.0, side=PositionSide.LONG,
            buffer_pct=0.15,
        )
        assert not unsafe

    def test_liquidation_price_calculation(self):
        """Verify liquidation price formulas."""
        rm = RiskManager(capital=1000)

        # Long 10x: liq = 50000 * (1 - 0.1 + 0.004) = 50000 * 0.904 = 45200
        liq_long = rm.calculate_liquidation_price(50_000, 10.0, PositionSide.LONG)
        assert liq_long == pytest.approx(45_200)

        # Short 10x: liq = 50000 * (1 + 0.1 - 0.004) = 50000 * 1.096 = 54800
        liq_short = rm.calculate_liquidation_price(50_000, 10.0, PositionSide.SHORT)
        assert liq_short == pytest.approx(54_800)
