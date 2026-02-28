"""Binance Futures execution client for live trading.

Handles:
- Setting leverage per symbol
- Placing market/limit orders
- Setting stop-loss / take-profit (STOP_MARKET / TAKE_PROFIT_MARKET)
- Querying positions and account balance
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog
from binance.client import Client
from binance.enums import (
    FUTURE_ORDER_TYPE_MARKET,
    FUTURE_ORDER_TYPE_STOP_MARKET,
    FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
    SIDE_BUY,
    SIDE_SELL,
)

from aether_btc.core.enums import OrderIntent, PositionSide
from aether_btc.core.models import Position

log = structlog.get_logger()


@dataclass
class OrderResult:
    """Result of an order execution."""

    order_id: int
    symbol: str
    side: str
    quantity: float
    price: float
    status: str
    timestamp: datetime


class BinanceFuturesClient:
    """Async-ready client for Binance Futures trading."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True) -> None:
        self.client = Client(api_key, api_secret, testnet=testnet)
        self.testnet = testnet

    def set_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        """Set leverage for a symbol."""
        result = self.client.futures_change_leverage(
            symbol=symbol, leverage=leverage,
        )
        log.info("leverage_set", symbol=symbol, leverage=leverage)
        return result

    def get_account_balance(self) -> dict[str, Any]:
        """Get futures account balance."""
        account = self.client.futures_account()
        return {
            "total_balance": float(account["totalWalletBalance"]),
            "available_balance": float(account["availableBalance"]),
            "unrealized_pnl": float(account["totalUnrealizedProfit"]),
            "total_margin": float(account["totalInitialMargin"]),
        }

    def get_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Get open positions."""
        positions = self.client.futures_position_information()
        if symbol:
            positions = [p for p in positions if p["symbol"] == symbol]
        return [
            {
                "symbol": p["symbol"],
                "side": "LONG" if float(p["positionAmt"]) > 0 else "SHORT",
                "quantity": abs(float(p["positionAmt"])),
                "entry_price": float(p["entryPrice"]),
                "unrealized_pnl": float(p["unRealizedProfit"]),
                "leverage": int(p["leverage"]),
                "liquidation_price": float(p["liquidationPrice"]),
            }
            for p in positions
            if float(p["positionAmt"]) != 0
        ]

    def place_market_order(
        self,
        symbol: str,
        intent: OrderIntent,
        quantity: float,
    ) -> OrderResult:
        """Place a market order."""
        side = SIDE_BUY if intent in (OrderIntent.OPEN_LONG, OrderIntent.CLOSE_SHORT) else SIDE_SELL

        result = self.client.futures_create_order(
            symbol=symbol,
            side=side,
            type=FUTURE_ORDER_TYPE_MARKET,
            quantity=round(quantity, 3),
        )

        log.info(
            "order_placed",
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_id=result["orderId"],
        )

        return OrderResult(
            order_id=result["orderId"],
            symbol=symbol,
            side=side,
            quantity=float(result["origQty"]),
            price=float(result.get("avgPrice", 0)),
            status=result["status"],
            timestamp=datetime.fromtimestamp(result["updateTime"] / 1000),
        )

    def place_stop_loss(
        self,
        symbol: str,
        side: PositionSide,
        stop_price: float,
        quantity: float,
    ) -> OrderResult:
        """Place a stop-loss (STOP_MARKET) order."""
        close_side = SIDE_SELL if side == PositionSide.LONG else SIDE_BUY

        result = self.client.futures_create_order(
            symbol=symbol,
            side=close_side,
            type=FUTURE_ORDER_TYPE_STOP_MARKET,
            stopPrice=round(stop_price, 2),
            quantity=round(quantity, 3),
            closePosition=False,
        )

        log.info("stop_loss_placed", symbol=symbol, stop_price=stop_price)
        return OrderResult(
            order_id=result["orderId"],
            symbol=symbol,
            side=close_side,
            quantity=float(result["origQty"]),
            price=stop_price,
            status=result["status"],
            timestamp=datetime.fromtimestamp(result["updateTime"] / 1000),
        )

    def place_take_profit(
        self,
        symbol: str,
        side: PositionSide,
        take_profit_price: float,
        quantity: float,
    ) -> OrderResult:
        """Place a take-profit (TAKE_PROFIT_MARKET) order."""
        close_side = SIDE_SELL if side == PositionSide.LONG else SIDE_BUY

        result = self.client.futures_create_order(
            symbol=symbol,
            side=close_side,
            type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
            stopPrice=round(take_profit_price, 2),
            quantity=round(quantity, 3),
            closePosition=False,
        )

        log.info("take_profit_placed", symbol=symbol, price=take_profit_price)
        return OrderResult(
            order_id=result["orderId"],
            symbol=symbol,
            side=close_side,
            quantity=float(result["origQty"]),
            price=take_profit_price,
            status=result["status"],
            timestamp=datetime.fromtimestamp(result["updateTime"] / 1000),
        )

    def cancel_all_orders(self, symbol: str) -> None:
        """Cancel all open orders for a symbol."""
        self.client.futures_cancel_all_open_orders(symbol=symbol)
        log.info("all_orders_cancelled", symbol=symbol)

    def parse_position(self, position_data: dict[str, Any]) -> Position:
        """Parse Binance position response into our Position model."""
        side = PositionSide.LONG if position_data["side"] == "LONG" else PositionSide.SHORT
        return Position(
            pair=position_data["symbol"],
            quantity=position_data["quantity"],
            entry_price=position_data["entry_price"],
            side=side,
            leverage=float(position_data["leverage"]),
            current_price=position_data["entry_price"],
        )
