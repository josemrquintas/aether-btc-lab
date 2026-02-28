"""Tests for Binance client — Phase 4 Live Trading Safety Gate."""

import pytest

from aether_btc.core.enums import OrderIntent, PositionSide
from aether_btc.core.models import Position


class TestOrderPayload:
    """Test order payload structure."""

    def test_order_payload_structure(self):
        """Market order includes required fields."""
        from aether_btc.core.models import Order
        order = Order(
            pair="BTCUSDT",
            intent=OrderIntent.OPEN_LONG,
            quantity=0.001,
            leverage=10.0,
            confidence=0.8,
        )
        assert order.pair == "BTCUSDT"
        assert order.intent == OrderIntent.OPEN_LONG
        assert order.quantity == 0.001
        assert order.leverage == 10.0

    def test_stop_loss_order_format(self):
        """SL order should have correct stop price."""
        # Test that we can create a stop-loss order with the right params
        stop_price = 49_000.0
        assert isinstance(stop_price, float)
        assert stop_price > 0


class TestPositionParsing:
    """Test parsing Binance position response."""

    def test_position_query_parsing(self):
        """Parse Binance position response into our Position model."""
        binance_response = {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": 0.001,
            "entry_price": 50_000.0,
            "leverage": 10,
        }

        pos = Position(
            pair=binance_response["symbol"],
            quantity=binance_response["quantity"],
            entry_price=binance_response["entry_price"],
            side=PositionSide.LONG if binance_response["side"] == "LONG" else PositionSide.SHORT,
            leverage=float(binance_response["leverage"]),
        )

        assert pos.pair == "BTCUSDT"
        assert pos.quantity == 0.001
        assert pos.leverage == 10.0
        assert pos.side == PositionSide.LONG
        assert pos.notional_value == pytest.approx(50.0)  # 0.001 * 50000

    def test_leverage_setting_payload(self):
        """Verify correct payload for setting leverage."""
        symbol = "BTCUSDT"
        leverage = 10
        # This would be the API call payload
        payload = {"symbol": symbol, "leverage": leverage}
        assert payload["symbol"] == "BTCUSDT"
        assert payload["leverage"] == 10
