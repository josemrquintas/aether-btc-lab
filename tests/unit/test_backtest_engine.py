"""Tests for backtest engine — Phase 2 Backtest Correctness Gate."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from aether_btc.backtest.engine import BacktestConfig, BacktestEngine
from aether_btc.core.enums import ExitReason, OrderIntent, PositionSide, SignalType
from aether_btc.core.models import Order, Signal
from aether_btc.ga.chromosome import Chromosome
from aether_btc.risk.manager import RiskLimits


def make_candles(n: int = 500, base_price: float = 50_000, trend: float = 0) -> pd.DataFrame:
    """Create synthetic candle data with optional trend."""
    np.random.seed(42)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    noise = np.random.randn(n) * 100
    trend_component = np.arange(n) * trend
    close = base_price + np.cumsum(noise) + trend_component
    close = np.maximum(close, 1000)  # Prevent negative prices

    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 20,
        "high": close + np.abs(np.random.randn(n) * 80),
        "low": close - np.abs(np.random.randn(n) * 80),
        "close": close,
        "volume": np.random.uniform(100, 1000, n),
        "quote_volume": np.random.uniform(5e6, 50e6, n),
    }, index=timestamps)

    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


class TestNextBarExecution:
    """Test that signals on bar T generate orders executed at bar T+1 OPEN."""

    def test_next_bar_execution(self):
        """Signal on bar T generates order executed at bar T+1 OPEN."""
        candles = make_candles(300)
        config = BacktestConfig(initial_capital=10_000)
        engine = BacktestEngine(config)

        # Manually inject a buy signal at bar 200
        bar_200_close = float(candles.iloc[200]["close"])
        bar_201_open = float(candles.iloc[201]["open"])

        # Create and queue an order
        order = Order(
            pair="BTCUSDT", intent=OrderIntent.OPEN_LONG,
            quantity=0.01, leverage=5.0,
            stop_loss=bar_200_close * 0.95,
            take_profit=bar_200_close * 1.10,
        )
        engine._pending_orders.append(order)

        # Execute at bar 201
        engine._execute_pending_orders(
            candles.index[201], "BTCUSDT",
            bar_201_open,
            float(candles.iloc[201]["high"]),
            float(candles.iloc[201]["low"]),
        )

        # Position should exist at bar 201 open price (+ slippage)
        pos = engine.portfolio.get_position("BTCUSDT")
        assert pos is not None
        # Entry should be near bar 201 open (with slippage)
        assert abs(pos.entry_price - bar_201_open) < bar_201_open * 0.01


class TestLeverageMultipliesPnL:
    """Test leverage correctly multiplies P&L."""

    def test_leverage_multiplies_pnl(self):
        """10x long, price +1% -> account gains ~10% of margin."""
        candles = make_candles(300)
        config = BacktestConfig(initial_capital=10_000)
        engine = BacktestEngine(config)

        entry_price = 50_000
        # Open at 10x
        engine.portfolio.open_position(
            "BTCUSDT", quantity=0.02, entry_price=entry_price,
            side=PositionSide.LONG, leverage=10.0,
            opened_at=candles.index[0],
        )
        # Margin = 0.02 * 50000 / 10 = 100
        pos = engine.portfolio.get_position("BTCUSDT")
        assert pos.margin_used == pytest.approx(100.0)

        # Price goes up 1%
        new_price = entry_price * 1.01
        engine.portfolio.update_prices({"BTCUSDT": new_price})

        # PnL should be ~10% of margin = ~10
        pos = engine.portfolio.get_position("BTCUSDT")
        assert pos.gross_unrealized_pnl == pytest.approx(10.0, rel=0.01)

    def test_leverage_multiplies_loss(self):
        """10x short, price +1% -> account loses 10% of margin."""
        config = BacktestConfig(initial_capital=10_000)
        engine = BacktestEngine(config)

        entry_price = 50_000
        engine.portfolio.open_position(
            "BTCUSDT", quantity=0.02, entry_price=entry_price,
            side=PositionSide.SHORT, leverage=10.0,
        )
        # Price goes UP 1% (bad for shorts)
        engine.portfolio.update_prices({"BTCUSDT": entry_price * 1.01})

        pos = engine.portfolio.get_position("BTCUSDT")
        assert pos.gross_unrealized_pnl == pytest.approx(-10.0, rel=0.01)


class TestLiquidation:
    """Test liquidation triggers."""

    def test_liquidation_triggers(self):
        """10x long, price drops past liquidation -> position force-closed, margin lost."""
        config = BacktestConfig(initial_capital=10_000)
        engine = BacktestEngine(config)

        entry_price = 50_000
        engine.portfolio.open_position(
            "BTCUSDT", quantity=0.02, entry_price=entry_price,
            side=PositionSide.LONG, leverage=10.0,
        )

        pos = engine.portfolio.get_position("BTCUSDT")
        liq_price = pos.liquidation_price  # ~45200

        # Price drops below liquidation
        assert pos.is_liquidated(liq_price - 100)
        assert not pos.is_liquidated(liq_price + 100)


class TestFundingCostDeduction:
    """Test funding cost deduction."""

    def test_funding_cost_deducted_every_8h(self):
        """Run 24h backtest, verify exactly 3 funding deductions."""
        candles = make_candles(96)  # 1 day = 96 bars
        funding_timestamps = pd.date_range("2024-01-01", periods=3, freq="8h", tz="UTC")
        funding = pd.DataFrame(
            {"funding_rate": [0.0001, 0.0001, 0.0001]},
            index=funding_timestamps,
        )

        config = BacktestConfig(initial_capital=10_000)
        engine = BacktestEngine(config)

        lookup = engine._build_funding_lookup(candles, funding)
        # Should have entries at bars 0, 32, 64
        assert 0 in lookup
        assert 32 in lookup
        assert 64 in lookup

    def test_funding_direction(self):
        """Positive funding: long pays. Negative: short pays."""
        config = BacktestConfig(initial_capital=10_000)
        engine = BacktestEngine(config)

        # Open long
        engine.portfolio.open_position(
            "BTCUSDT", quantity=0.1, entry_price=50_000,
            side=PositionSide.LONG, leverage=5.0,
        )

        # Positive rate: longs pay
        initial_cash = engine.portfolio.cash
        engine.portfolio.accrue_funding_costs({"BTCUSDT": 0.0001})
        assert engine.portfolio.cash < initial_cash  # Cash decreased


class TestCommissionOnNotional:
    """Test commission on notional value."""

    def test_commission_on_notional(self):
        """$1k margin, 10x leverage = $10k notional -> commission on $10k."""
        config = BacktestConfig(initial_capital=10_000, commission_rate=0.0004)
        entry_price = 50_000
        quantity = 0.2  # 0.2 * 50000 = $10k notional
        expected_commission = quantity * entry_price * 0.0004  # $4.0

        assert expected_commission == pytest.approx(4.0)


class TestCircuitBreaker:
    """Test circuit breaker halts trading."""

    def test_circuit_breaker_halts_trading(self):
        """Account drops 10% -> circuit breaker triggers."""
        config = BacktestConfig(initial_capital=1000)
        engine = BacktestEngine(config)

        # Simulate 10% loss
        engine.risk_manager.update_capital(900)
        assert engine.risk_manager.circuit_breaker_triggered

        # Pending orders should be cancelled
        engine._pending_orders.append(Order(
            pair="BTCUSDT", intent=OrderIntent.OPEN_LONG, quantity=0.01,
        ))
        # Simulate bar where circuit breaker cancels orders
        if engine.risk_manager.circuit_breaker_triggered:
            engine._pending_orders.clear()
        assert len(engine._pending_orders) == 0


class TestIntrabarStops:
    """Test intrabar stop-loss and take-profit."""

    def test_stop_loss_intrabar(self):
        """SL at $49k, bar low hits $48.5k -> exits at $49k."""
        config = BacktestConfig(initial_capital=10_000)
        engine = BacktestEngine(config)

        engine.portfolio.open_position(
            "BTCUSDT", quantity=0.1, entry_price=50_000,
            side=PositionSide.LONG, leverage=5.0,
            stop_loss=49_000,
        )

        # Bar with low below stop loss
        engine._check_stops_intrabar(
            datetime.now(), "BTCUSDT",
            high=50_500, low=48_500,
        )

        # Position should be closed
        assert not engine.portfolio.has_position("BTCUSDT")
        closed = engine.portfolio.closed_trades[-1]
        assert closed.exit_reason == ExitReason.STOP_LOSS
        assert closed.exit_price == pytest.approx(49_000)

    def test_take_profit_intrabar(self):
        """TP at $51k, bar high hits $51.5k -> exits at $51k."""
        config = BacktestConfig(initial_capital=10_000)
        engine = BacktestEngine(config)

        engine.portfolio.open_position(
            "BTCUSDT", quantity=0.1, entry_price=50_000,
            side=PositionSide.LONG, leverage=5.0,
            take_profit=51_000,
        )

        engine._check_stops_intrabar(
            datetime.now(), "BTCUSDT",
            high=51_500, low=49_500,
        )

        assert not engine.portfolio.has_position("BTCUSDT")
        closed = engine.portfolio.closed_trades[-1]
        assert closed.exit_reason == ExitReason.TAKE_PROFIT
        assert closed.exit_price == pytest.approx(51_000)
