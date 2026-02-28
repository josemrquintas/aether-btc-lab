"""Telegram notifications for trade alerts.

Sends formatted messages for:
- Trade entries (pair, side, leverage, entry price, SL, TP)
- Trade exits (pair, side, exit price, PnL, exit reason)
- System alerts (circuit breaker, liquidation warnings)
"""

import structlog
from telegram import Bot

from aether_btc.core.config import AlertConfig
from aether_btc.core.models import ClosedPosition, Position

log = structlog.get_logger()


class TelegramAlerter:
    """Send trade alerts via Telegram."""

    def __init__(self, config: AlertConfig | None = None) -> None:
        self.config = config or AlertConfig()
        self._bot: Bot | None = None

    @property
    def bot(self) -> Bot:
        if self._bot is None:
            self._bot = Bot(token=self.config.telegram_token)
        return self._bot

    @property
    def enabled(self) -> bool:
        return bool(self.config.telegram_token and self.config.telegram_chat_id)

    async def send_message(self, text: str) -> None:
        """Send a message to the configured chat."""
        if not self.enabled:
            log.debug("telegram_disabled")
            return
        try:
            await self.bot.send_message(
                chat_id=self.config.telegram_chat_id,
                text=text,
                parse_mode="HTML",
            )
        except Exception as e:
            log.error("telegram_send_failed", error=str(e))

    async def alert_entry(self, position: Position) -> None:
        """Send entry alert."""
        emoji = "\u2b06\ufe0f" if position.side.value == "LONG" else "\u2b07\ufe0f"
        text = (
            f"{emoji} <b>NEW POSITION</b>\n"
            f"Pair: {position.pair}\n"
            f"Side: {position.side.value}\n"
            f"Leverage: {position.leverage}x\n"
            f"Entry: ${position.entry_price:,.2f}\n"
            f"Quantity: {position.quantity:.6f} BTC\n"
            f"Margin: ${position.margin_used:,.2f}\n"
            f"SL: ${position.stop_loss:,.2f}" if position.stop_loss else ""
        )
        if position.take_profit:
            text += f"\nTP: ${position.take_profit:,.2f}"
        text += f"\nLiq: ${position.liquidation_price:,.2f}"

        await self.send_message(text)

    async def alert_exit(self, closed: ClosedPosition) -> None:
        """Send exit alert."""
        pnl_emoji = "\u2705" if closed.net_pnl > 0 else "\u274c"
        text = (
            f"{pnl_emoji} <b>POSITION CLOSED</b>\n"
            f"Pair: {closed.pair}\n"
            f"Side: {closed.side.value}\n"
            f"Leverage: {closed.leverage}x\n"
            f"Entry: ${closed.entry_price:,.2f}\n"
            f"Exit: ${closed.exit_price:,.2f}\n"
            f"PnL: ${closed.net_pnl:,.2f} ({closed.net_pnl_percent:+.1f}%)\n"
            f"Reason: {closed.exit_reason.value}\n"
            f"Duration: {closed.hold_duration_hours:.1f}h"
        )
        await self.send_message(text)

    async def alert_system(self, title: str, message: str) -> None:
        """Send system alert (circuit breaker, liquidation warning, etc.)."""
        text = f"\u26a0\ufe0f <b>{title}</b>\n{message}"
        await self.send_message(text)
