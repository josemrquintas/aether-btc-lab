"""Backtesting engine for Aether BTC — THE CRITICAL MODULE.

Adapted from Aether v3 for 15-minute crypto bars with:
- Leverage simulation (1x-20x)
- Liquidation checks every bar
- Funding rate costs every 8h (32 bars)
- Intrabar stop-loss / take-profit
- Commission on notional value (not margin)
- Circuit breaker at 10% drawdown

Simulation loop per bar:
1. Update prices to OPEN
2. Check liquidation -> force-close if liquidated
3. Check stop-loss/take-profit at bar HIGH/LOW (intrabar)
4. Execute pending orders at current bar OPEN + slippage
5. Apply funding rates every 32 bars
6. Update prices to CLOSE
7. Generate signals using data up to current bar CLOSE
8. Convert signals to orders (queued for NEXT bar)
9. Record equity snapshot
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import structlog

from aether_btc.core.enums import (
    ExitReason,
    OrderIntent,
    PositionSide,
    SignalType,
    SlippageModel,
)
from aether_btc.core.models import ClosedPosition, EquitySnapshot, Order, Signal, Trade
from aether_btc.ga.chromosome import Chromosome
from aether_btc.portfolio.manager import PortfolioManager
from aether_btc.risk.manager import RiskManager, RiskLimits

log = structlog.get_logger()


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    initial_capital: float = 1000.0
    commission_rate: float = 0.0004  # 0.04% taker
    slippage_rate: float = 0.0001  # 0.01%
    slippage_model: SlippageModel = SlippageModel.FIXED_PCT
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    bars_per_funding: int = 32  # 8h / 15min
    forced_liquidation_slippage_multiplier: float = 2.0


@dataclass
class BacktestResult:
    """Results from a backtest run. All metrics are NET of costs."""

    config: BacktestConfig
    initial_capital: float
    final_capital: float
    # Returns
    total_return: float
    annualized_return: float
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    avg_hold_hours: float
    # Long/short attribution
    long_trades: int = 0
    short_trades: int = 0
    long_pnl: float = 0.0
    short_pnl: float = 0.0
    # Cost breakdown
    total_commissions: float = 0.0
    total_slippage: float = 0.0
    total_funding_costs: float = 0.0
    cost_drag_percent: float = 0.0
    # Leverage
    avg_leverage: float = 1.0
    max_leverage: float = 1.0
    liquidations: int = 0
    # Data
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: list[ClosedPosition] = field(default_factory=list)
    bar_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


class BacktestEngine:
    """Event-driven backtesting engine for leveraged crypto trading."""

    def __init__(
        self,
        config: BacktestConfig,
        chromosome: Chromosome | None = None,
        signal_fn: object | None = None,
    ) -> None:
        self.config = config
        self.chromosome = chromosome
        self.signal_fn = signal_fn  # Callable[[pd.DataFrame, int, Chromosome], Signal | None]
        self.portfolio = PortfolioManager(initial_capital=config.initial_capital)
        self.risk_manager = RiskManager(
            capital=config.initial_capital,
            limits=config.risk_limits,
        )
        self._pending_orders: list[Order] = []
        self._equity_snapshots: list[EquitySnapshot] = []
        self._leverage_history: list[float] = []
        self._liquidation_count = 0
        self._bar_count = 0

    def run(
        self,
        candles: pd.DataFrame,
        funding_rates: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """Run backtest simulation on 15-minute candle data.

        Args:
            candles: DataFrame with DatetimeIndex and columns: open, high, low, close, volume
                     Plus any indicator columns needed by signal_fn.
            funding_rates: DataFrame with DatetimeIndex and column: funding_rate
        """
        if candles.empty:
            raise ValueError("No candle data provided")

        log.info(
            "backtest_starting",
            bars=len(candles),
            start=candles.index[0],
            end=candles.index[-1],
        )

        # Build funding rate lookup: bar_index -> rate (for bars where funding applies)
        funding_lookup = self._build_funding_lookup(candles, funding_rates)

        # Simulate each bar
        for i in range(len(candles)):
            self._simulate_bar(i, candles, funding_lookup)

        # Close remaining positions at end
        if self.portfolio.position_count > 0:
            last_bar = candles.iloc[-1]
            pair = "BTCUSDT"
            close_price = float(last_bar["close"])
            for pair_key in list(self.portfolio.positions.keys()):
                pos = self.portfolio.get_position(pair_key)
                if pos:
                    commission = close_price * pos.quantity * self.config.commission_rate
                    self.portfolio.close_position(
                        pair_key, close_price, ExitReason.END_OF_BACKTEST, commission=commission,
                        closed_at=candles.index[-1],
                    )

        result = self._calculate_metrics(candles)
        log.info(
            "backtest_complete",
            total_return=f"{result.total_return:.2%}",
            sharpe=f"{result.sharpe_ratio:.2f}",
            trades=result.total_trades,
            liquidations=result.liquidations,
        )
        return result

    def _build_funding_lookup(
        self,
        candles: pd.DataFrame,
        funding_rates: pd.DataFrame | None,
    ) -> dict[int, float]:
        """Build bar_index -> funding_rate lookup for bars where funding is charged."""
        lookup: dict[int, float] = {}
        if funding_rates is None or funding_rates.empty:
            return lookup

        for i in range(0, len(candles), self.config.bars_per_funding):
            bar_ts = candles.index[i]
            # Find closest funding rate at or before this timestamp
            mask = funding_rates.index <= bar_ts
            if mask.any():
                rate = float(funding_rates.loc[mask].iloc[-1]["funding_rate"])
                lookup[i] = rate

        return lookup

    def _simulate_bar(
        self,
        bar_idx: int,
        candles: pd.DataFrame,
        funding_lookup: dict[int, float],
    ) -> None:
        """Simulate one 15-minute bar."""
        bar = candles.iloc[bar_idx]
        bar_ts = candles.index[bar_idx]
        pair = "BTCUSDT"

        open_price = float(bar["open"])
        high_price = float(bar["high"])
        low_price = float(bar["low"])
        close_price = float(bar["close"])

        # Step 1: Update prices to OPEN
        self.portfolio.update_prices({pair: open_price})

        # Step 2: Check liquidation
        liquidated = self.portfolio.check_liquidations({pair: low_price if True else high_price})
        # Check both extremes for liquidation
        for pos_pair in list(self.portfolio.positions.keys()):
            pos = self.portfolio.get_position(pos_pair)
            if pos is None:
                continue
            # For longs, check if low breached liquidation; for shorts, check if high did
            check_price = low_price if pos.side == PositionSide.LONG else high_price
            if pos.is_liquidated(check_price):
                commission = pos.notional_value * self.config.commission_rate
                self.portfolio.close_position(
                    pos_pair, pos.liquidation_price, ExitReason.LIQUIDATION,
                    commission=commission, closed_at=bar_ts,
                )
                self._liquidation_count += 1
                log.warning("liquidation", pair=pos_pair, bar=bar_idx, price=check_price)

        # Step 3: Check stop-loss / take-profit at HIGH/LOW (intrabar)
        self._check_stops_intrabar(bar_ts, pair, high_price, low_price)

        # Step 3.5: Cancel pending orders if circuit breaker
        if self.risk_manager.circuit_breaker_triggered and self._pending_orders:
            self._pending_orders.clear()

        # Step 4: Execute pending orders at OPEN + slippage
        self._execute_pending_orders(bar_ts, pair, open_price, high_price, low_price)

        # Step 5: Apply funding rates
        if bar_idx in funding_lookup:
            rate = funding_lookup[bar_idx]
            self.portfolio.accrue_funding_costs({pair: rate})

        # Step 6: Update prices to CLOSE
        self.portfolio.update_prices({pair: close_price})

        # Step 7: Generate signals using data up to current bar
        if self.signal_fn and self.chromosome:
            signal = self.signal_fn(candles, bar_idx, self.chromosome)
            if signal:
                # Step 8: Convert signal to order
                self._signal_to_order(signal, bar_ts, pair, close_price, candles, bar_idx)

        # Step 9: Record equity snapshot
        self._record_equity(bar_ts)
        self._leverage_history.append(self.portfolio.leverage_ratio)
        self.risk_manager.update_capital(self.portfolio.equity)
        self._bar_count += 1

    def _check_stops_intrabar(
        self, bar_ts: datetime, pair: str, high: float, low: float,
    ) -> None:
        """Check stop-loss/take-profit at bar HIGH/LOW."""
        for pos_pair in list(self.portfolio.positions.keys()):
            pos = self.portfolio.get_position(pos_pair)
            if pos is None:
                continue

            # Stop-loss check
            sl_check_price = low if pos.side == PositionSide.LONG else high
            if pos.should_stop_loss(sl_check_price):
                exit_price = pos.stop_loss if pos.stop_loss else sl_check_price
                # Ensure exit price is within bar range
                exit_price = max(low, min(high, exit_price))
                commission = exit_price * pos.quantity * self.config.commission_rate
                self.portfolio.close_position(
                    pos_pair, exit_price, ExitReason.STOP_LOSS,
                    commission=commission, closed_at=bar_ts,
                )
                continue

            # Take-profit check
            tp_check_price = high if pos.side == PositionSide.LONG else low
            if pos.should_take_profit(tp_check_price):
                exit_price = pos.take_profit if pos.take_profit else tp_check_price
                exit_price = max(low, min(high, exit_price))
                commission = exit_price * pos.quantity * self.config.commission_rate
                self.portfolio.close_position(
                    pos_pair, exit_price, ExitReason.TAKE_PROFIT,
                    commission=commission, closed_at=bar_ts,
                )

    def _execute_pending_orders(
        self, bar_ts: datetime, pair: str,
        open_price: float, high: float, low: float,
    ) -> None:
        """Execute orders queued from PREVIOUS bar at current bar OPEN + slippage."""
        executed = []
        for order in self._pending_orders:
            exec_price = self._apply_slippage(open_price, order.intent)

            if order.intent == OrderIntent.OPEN_LONG:
                commission = exec_price * order.quantity * self.config.commission_rate
                pos = self.portfolio.open_position(
                    pair=pair, quantity=order.quantity, entry_price=exec_price,
                    side=PositionSide.LONG, leverage=order.leverage,
                    stop_loss=order.stop_loss, take_profit=order.take_profit,
                    commission=commission, opened_at=bar_ts,
                )
                if pos:
                    executed.append(order)

            elif order.intent == OrderIntent.CLOSE_LONG:
                commission = exec_price * order.quantity * self.config.commission_rate
                closed = self.portfolio.close_position(
                    pair=pair, exit_price=exec_price, exit_reason=ExitReason.SIGNAL,
                    commission=commission, closed_at=bar_ts,
                )
                if closed:
                    executed.append(order)

            elif order.intent == OrderIntent.OPEN_SHORT:
                commission = exec_price * order.quantity * self.config.commission_rate
                pos = self.portfolio.open_position(
                    pair=pair, quantity=order.quantity, entry_price=exec_price,
                    side=PositionSide.SHORT, leverage=order.leverage,
                    stop_loss=order.stop_loss, take_profit=order.take_profit,
                    commission=commission, opened_at=bar_ts,
                )
                if pos:
                    executed.append(order)

            elif order.intent == OrderIntent.CLOSE_SHORT:
                commission = exec_price * order.quantity * self.config.commission_rate
                closed = self.portfolio.close_position(
                    pair=pair, exit_price=exec_price, exit_reason=ExitReason.SIGNAL,
                    commission=commission, closed_at=bar_ts,
                )
                if closed:
                    executed.append(order)

        self._pending_orders = [o for o in self._pending_orders if o not in executed]

    def _apply_slippage(self, price: float, intent: OrderIntent) -> float:
        """Apply slippage. Buys pay more, sells receive less."""
        slip = price * self.config.slippage_rate
        if intent in (OrderIntent.OPEN_LONG, OrderIntent.CLOSE_SHORT):
            return price + slip
        return price - slip

    def _signal_to_order(
        self,
        signal: Signal,
        bar_ts: datetime,
        pair: str,
        close_price: float,
        candles: pd.DataFrame,
        bar_idx: int,
    ) -> None:
        """Convert a signal to an order queued for next bar execution."""
        has_position = self.portfolio.has_position(pair)
        position = self.portfolio.get_position(pair)

        # Calculate ATR for stop levels
        if bar_idx >= 14:
            recent = candles.iloc[max(0, bar_idx - 14):bar_idx + 1]
            atr_val = float((recent["high"] - recent["low"]).mean())
        else:
            atr_val = close_price * 0.02

        if signal.signal_type == SignalType.BUY and not has_position:
            # Calculate leverage from chromosome
            leverage = self._get_leverage(signal.confidence, candles, bar_idx)

            stops = self.risk_manager.calculate_stop_levels(
                entry_price=close_price, side=PositionSide.LONG, atr=atr_val,
            )

            # Check liquidation buffer
            if not self.risk_manager.check_liquidation_buffer(
                close_price, close_price, leverage, PositionSide.LONG
            ):
                return

            size = self.risk_manager.calculate_position_size(
                entry_price=close_price, leverage=leverage,
                side=PositionSide.LONG, stop_loss=stops.stop_loss, atr=atr_val,
                current_positions=self.portfolio.position_count,
            )

            if size.quantity > 0:
                self._pending_orders.append(Order(
                    pair=pair, intent=OrderIntent.OPEN_LONG,
                    quantity=size.quantity, leverage=leverage,
                    stop_loss=stops.stop_loss, take_profit=stops.take_profit,
                    confidence=signal.confidence, generated_at=bar_ts,
                ))

        elif signal.signal_type == SignalType.SELL and has_position:
            if position and position.side == PositionSide.LONG:
                self._pending_orders.append(Order(
                    pair=pair, intent=OrderIntent.CLOSE_LONG,
                    quantity=position.quantity, confidence=signal.confidence,
                    generated_at=bar_ts,
                ))

        elif signal.signal_type == SignalType.SHORT and not has_position:
            leverage = self._get_leverage(signal.confidence, candles, bar_idx)

            stops = self.risk_manager.calculate_stop_levels(
                entry_price=close_price, side=PositionSide.SHORT, atr=atr_val,
            )

            if not self.risk_manager.check_liquidation_buffer(
                close_price, close_price, leverage, PositionSide.SHORT
            ):
                return

            size = self.risk_manager.calculate_position_size(
                entry_price=close_price, leverage=leverage,
                side=PositionSide.SHORT, stop_loss=stops.stop_loss, atr=atr_val,
                current_positions=self.portfolio.position_count,
            )

            if size.quantity > 0:
                self._pending_orders.append(Order(
                    pair=pair, intent=OrderIntent.OPEN_SHORT,
                    quantity=size.quantity, leverage=leverage,
                    stop_loss=stops.stop_loss, take_profit=stops.take_profit,
                    confidence=signal.confidence, generated_at=bar_ts,
                ))

        elif signal.signal_type == SignalType.COVER and has_position:
            if position and position.side == PositionSide.SHORT:
                self._pending_orders.append(Order(
                    pair=pair, intent=OrderIntent.CLOSE_SHORT,
                    quantity=position.quantity, confidence=signal.confidence,
                    generated_at=bar_ts,
                ))

    def _get_leverage(self, confidence: float, candles: pd.DataFrame, bar_idx: int) -> float:
        """Get leverage from chromosome genes or default."""
        if self.chromosome is None:
            return 3.0

        # Calculate recent volatility (normalized ATR)
        lookback = min(96, bar_idx)
        if lookback > 0:
            recent = candles.iloc[bar_idx - lookback:bar_idx + 1]
            vol = float((recent["high"] - recent["low"]).mean() / recent["close"].mean())
        else:
            vol = 0.02

        return self.risk_manager.calculate_leverage(
            base_leverage=self.chromosome.get("leverage_base"),
            confidence=confidence,
            confidence_scale=self.chromosome.get("leverage_confidence_scale"),
            volatility=vol,
            volatility_dampen=self.chromosome.get("leverage_volatility_dampen"),
            max_cap=self.chromosome.get("max_leverage_cap"),
        )

    def _record_equity(self, bar_ts: datetime) -> None:
        snapshot = EquitySnapshot(
            timestamp=bar_ts,
            cash=self.portfolio.cash,
            margin_in_use=self.portfolio.total_margin_in_use,
            unrealized_pnl=self.portfolio.total_unrealized_pnl,
            total_equity=self.portfolio.equity,
            leverage_ratio=self.portfolio.leverage_ratio,
        )

        if self._equity_snapshots:
            prev = self._equity_snapshots[-1].total_equity
            if prev > 0:
                snapshot.bar_return = (snapshot.total_equity - prev) / prev

        self._equity_snapshots.append(snapshot)

    def _calculate_metrics(self, candles: pd.DataFrame) -> BacktestResult:
        """Calculate performance metrics. All NET of costs."""
        if not self._equity_snapshots:
            return BacktestResult(
                config=self.config, initial_capital=self.config.initial_capital,
                final_capital=self.config.initial_capital,
                total_return=0, annualized_return=0,
                sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                max_drawdown=0, total_trades=0, winning_trades=0,
                losing_trades=0, win_rate=0, profit_factor=0,
                avg_trade_return=0, avg_win=0, avg_loss=0, avg_hold_hours=0,
            )

        # Equity curve
        equity_data = [{
            "timestamp": s.timestamp,
            "cash": s.cash,
            "equity": s.total_equity,
            "leverage": s.leverage_ratio,
            "return": s.bar_return,
        } for s in self._equity_snapshots]
        equity_df = pd.DataFrame(equity_data).set_index("timestamp")
        bar_returns = equity_df["return"].dropna()

        initial = self.config.initial_capital
        final = self._equity_snapshots[-1].total_equity
        total_return = (final - initial) / initial

        # Annualize: 96 bars/day, 365 days/year = 35040 bars/year
        total_bars = len(candles)
        years = total_bars / 35040
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Risk metrics
        if len(bar_returns) > 1:
            # Annualization factor: sqrt(bars_per_year)
            ann_factor = np.sqrt(35040)
            vol = bar_returns.std() * ann_factor
            sharpe = (annualized_return - 0.02) / vol if vol > 0 else 0

            downside = bar_returns[bar_returns < 0]
            downside_std = downside.std() * ann_factor if len(downside) > 0 else vol
            sortino = (annualized_return - 0.02) / downside_std if downside_std > 0 else 0

            cumulative = (1 + bar_returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0

            calmar = annualized_return / max_dd if max_dd > 0 else 0
        else:
            sharpe = sortino = calmar = max_dd = 0.0

        # Trade metrics (NET)
        trades = self.portfolio.closed_trades
        winning = [t for t in trades if t.net_pnl > 0]
        losing = [t for t in trades if t.net_pnl <= 0]
        total_trades = len(trades)
        win_rate = len(winning) / total_trades if total_trades > 0 else 0

        avg_win = float(np.mean([t.net_pnl for t in winning])) if winning else 0
        avg_loss = float(np.mean([t.net_pnl for t in losing])) if losing else 0

        gross_profit = sum(t.net_pnl for t in winning)
        gross_loss = abs(sum(t.net_pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_trade_return = float(np.mean([t.net_pnl_percent for t in trades])) if trades else 0
        avg_hold_hours = float(np.mean([t.hold_duration_hours for t in trades])) if trades else 0

        # Attribution
        long_trades = [t for t in trades if t.side == PositionSide.LONG]
        short_trades = [t for t in trades if t.side == PositionSide.SHORT]

        # Costs
        total_comm = sum(t.entry_commission + t.exit_commission for t in trades)
        total_slip = sum(t.entry_slippage + t.exit_slippage for t in trades)
        total_fund = sum(t.funding_cost for t in trades)
        cost_drag = (total_comm + total_slip + total_fund) / initial * 100 if initial > 0 else 0

        # Leverage
        avg_lev = float(np.mean(self._leverage_history)) if self._leverage_history else 0
        max_lev = max(self._leverage_history) if self._leverage_history else 0

        return BacktestResult(
            config=self.config,
            initial_capital=initial,
            final_capital=final,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            total_trades=total_trades,
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=avg_trade_return,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_hold_hours=avg_hold_hours,
            long_trades=len(long_trades),
            short_trades=len(short_trades),
            long_pnl=sum(t.net_pnl for t in long_trades),
            short_pnl=sum(t.net_pnl for t in short_trades),
            total_commissions=total_comm,
            total_slippage=total_slip,
            total_funding_costs=total_fund,
            cost_drag_percent=cost_drag,
            avg_leverage=avg_lev,
            max_leverage=max_lev,
            liquidations=self._liquidation_count,
            equity_curve=equity_df,
            trades=trades,
            bar_returns=bar_returns,
        )
