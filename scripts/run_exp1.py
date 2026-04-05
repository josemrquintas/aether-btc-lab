#!/usr/bin/env python3
"""Experiment 1 — Equal-weight ensemble baseline backtest.

Scientifically minimal execution policy to establish the raw edge of the
5 precomputed strategies (momentum, mean_reversion, trend_following,
volatility_breakout, funding_volume) as an equal-weight ensemble.

Policy:
  - Persistent per-strategy state (values already -1/0/+1 in the DB)
  - Target state = sign(sum of 5 strategy states)
  - On zero net vote: HOLD current position (no whipsaw on ties)
  - Full flip at NEXT bar open on direction change
  - $1000 fixed notional, 1x leverage, no SL/TP/time-stop
  - 0.04% commission per side, no slippage, no funding costs (1x = zero)
  - 30-day warmup period skipped (strategy state stabilisation)

Free post-hoc analyses (same data, no extra runs):
  - Per-strategy solo equity curves (if each strategy alone drove position)
  - Correlation matrix between the 5 persistent-state time series
  - Buy-and-hold benchmark over identical bars, identical costs
  - Alpha = strategy_return - buy_and_hold_return
  - Year-by-year breakdown for ensemble + buy-and-hold

Writes to aether_btc_lab DB: exp1_trades, exp1_summary, exp1_equity_curve.

Usage:
    python scripts/run_exp1.py
    python scripts/run_exp1.py --notional 1000 --capital 10000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from dotenv import load_dotenv
from sqlalchemy import text

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))
load_dotenv(_project_root / ".env")

from aether_btc.data.database import get_engine  # noqa: E402

log = structlog.get_logger()

STRATEGIES = [
    "sig_momentum",
    "sig_mean_reversion",
    "sig_trend_following",
    "sig_volatility_breakout",
    "sig_funding_volume",
]
WARMUP_DAYS = 30
BARS_PER_DAY = 96  # 15m
COMMISSION_RATE = 0.0004  # 0.04%


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_exp1_data(engine, pair: str = "BTCUSDT") -> pd.DataFrame:
    """Load all bars with per-strategy persistent states from btc_indicators_15m."""
    sql = text("""
        SELECT
            timestamp,
            open, high, low, close,
            (indicators->>'sig_momentum')::float           AS sig_momentum,
            (indicators->>'sig_mean_reversion')::float     AS sig_mean_reversion,
            (indicators->>'sig_trend_following')::float    AS sig_trend_following,
            (indicators->>'sig_volatility_breakout')::float AS sig_volatility_breakout,
            (indicators->>'sig_funding_volume')::float     AS sig_funding_volume
        FROM btc_indicators_15m
        WHERE pair = :pair
          AND indicators IS NOT NULL
          AND indicators ? 'sig_momentum'
        ORDER BY timestamp
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"pair": pair}, parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    # Any strategy that is still null at this point is a warmup gap — forward fill from 0
    for col in STRATEGIES:
        df[col] = df[col].fillna(0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------------
def simulate_ensemble(
    df: pd.DataFrame,
    starting_capital: float,
    notional: float,  # unused — kept for signature compatibility; we use 100% of equity
) -> tuple[list[dict], pd.DataFrame]:
    """Run the equal-weight ensemble backtest.

    Returns (trades, equity_curve).
    trades = list of {entry_time, exit_time, side, entry_price, exit_price,
                      pnl, commission, bars_held, exit_reason, net_vote_at_entry}
    equity_curve = DataFrame with columns [equity, position, net_vote]
    """
    bars = df.reset_index()  # numeric index for .iloc access
    n = len(bars)

    # net vote per bar = sum of strategy state columns
    net_vote = bars[STRATEGIES].sum(axis=1).astype(int).values

    equity = starting_capital
    position_side = 0  # -1 short, 0 flat, +1 long
    entry_price = 0.0
    entry_time = None
    entry_vote = 0
    entry_idx = 0
    trade_notional = 0.0  # notional of the currently open position
    trades: list[dict] = []

    equity_records = []

    warmup_bars = WARMUP_DAYS * BARS_PER_DAY

    for i in range(n):
        vote = net_vote[i]

        # Before warmup completes, do nothing (no positions, no trades recorded)
        if i < warmup_bars:
            equity_records.append({
                "timestamp": bars["timestamp"].iat[i],
                "equity": equity,
                "position": 0,
                "net_vote": vote,
            })
            continue

        # Determine target direction (hold on zero)
        if vote > 0:
            target = 1
        elif vote < 0:
            target = -1
        else:
            target = position_side  # hold

        # If direction changes, close current at this bar's OPEN of next bar (i+1) — we use close+1 as proxy
        # For no-look-ahead: decide at bar i (close), execute at bar i+1 open
        if target != position_side and i + 1 < n:
            exec_price = float(bars["open"].iat[i + 1])

            # Close existing position if any
            if position_side != 0:
                # 100% of trade_notional (current equity at entry time) in the trade
                # PnL = (price_pct_change * side) * trade_notional - commissions
                price_return = (exec_price - entry_price) / entry_price * position_side
                gross_pnl = price_return * trade_notional
                exit_commission = trade_notional * COMMISSION_RATE
                entry_commission = trade_notional * COMMISSION_RATE
                pnl = gross_pnl - exit_commission - entry_commission
                equity += pnl
                if equity < 0:
                    equity = 0  # ruin — account wiped, can't trade
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": bars["timestamp"].iat[i + 1],
                    "side": "LONG" if position_side == 1 else "SHORT",
                    "entry_price": float(entry_price),
                    "exit_price": float(exec_price),
                    "bars_held": int((i + 1) - entry_idx),
                    "pnl": float(pnl),
                    "commission": float(entry_commission + exit_commission),
                    "exit_reason": "flip" if target != 0 else "flat",
                    "net_vote_at_entry": int(entry_vote),
                })

            # Open new position if target is non-zero — use 100% of current equity
            if target != 0 and equity > 0:
                entry_price = exec_price
                entry_time = bars["timestamp"].iat[i + 1]
                entry_idx = i + 1
                entry_vote = vote
                trade_notional = equity  # full allocation
                position_side = target
            else:
                position_side = 0

        equity_records.append({
            "timestamp": bars["timestamp"].iat[i],
            "equity": equity,
            "position": position_side,
            "net_vote": vote,
        })

    # Close any open position at the last bar
    if position_side != 0:
        exec_price = float(bars["close"].iat[-1])
        price_return = (exec_price - entry_price) / entry_price * position_side
        gross_pnl = price_return * trade_notional
        exit_commission = trade_notional * COMMISSION_RATE
        entry_commission = trade_notional * COMMISSION_RATE
        pnl = gross_pnl - exit_commission - entry_commission
        equity += pnl
        if equity < 0:
            equity = 0
        trades.append({
            "entry_time": entry_time,
            "exit_time": bars["timestamp"].iat[-1],
            "side": "LONG" if position_side == 1 else "SHORT",
            "entry_price": float(entry_price),
            "exit_price": float(exec_price),
            "bars_held": int((n - 1) - entry_idx),
            "pnl": float(pnl),
            "commission": float(entry_commission + exit_commission),
            "exit_reason": "end_of_data",
            "net_vote_at_entry": int(entry_vote),
        })

    equity_df = pd.DataFrame(equity_records).set_index("timestamp")
    return trades, equity_df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(equity_curve: pd.DataFrame, trades: list[dict],
                    starting_capital: float, label: str) -> dict:
    """Compute return/risk/trade metrics from an equity curve."""
    eq = equity_curve["equity"].values
    bar_returns = np.diff(eq) / eq[:-1]
    bar_returns = bar_returns[~np.isnan(bar_returns)]

    total_return = (eq[-1] - starting_capital) / starting_capital
    days = (equity_curve.index[-1] - equity_curve.index[0]).days or 1
    years = days / 365.25
    annualized = (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else 0.0

    # Sharpe (bar-level, annualized — 96 bars/day × 365 days = 35040 bars/year for 15m)
    bars_per_year = 96 * 365.25
    if len(bar_returns) > 1 and bar_returns.std() > 0:
        sharpe = (bar_returns.mean() / bar_returns.std()) * np.sqrt(bars_per_year)
    else:
        sharpe = 0.0

    # Max DD
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    calmar = annualized / abs(max_dd) if max_dd < 0 else 0.0

    # Trade stats
    n_trades = len(trades)
    if n_trades > 0:
        pnls = np.array([t["pnl"] for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        win_rate = len(wins) / n_trades
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float("inf")
        avg_bars_held = float(np.mean([t["bars_held"] for t in trades]))
    else:
        win_rate = avg_win = avg_loss = profit_factor = avg_bars_held = 0.0

    return {
        "label": label,
        "starting_capital": starting_capital,
        "final_equity": float(eq[-1]),
        "total_return": float(total_return),
        "annualized_return": float(annualized),
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "max_drawdown": max_dd,
        "total_trades": n_trades,
        "win_rate": float(win_rate),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": float(profit_factor) if profit_factor != float("inf") else 999.0,
        "avg_bars_held": avg_bars_held,
        "period_days": days,
        "period_years": round(years, 2),
    }


def buy_and_hold_equity(df: pd.DataFrame, starting_capital: float, notional: float) -> pd.DataFrame:
    """Buy-and-hold benchmark: 100% of starting capital into BTC at first post-warmup bar.

    Includes entry + exit commissions so the comparison is fair."""
    warmup_bars = WARMUP_DAYS * BARS_PER_DAY
    if len(df) <= warmup_bars:
        return pd.DataFrame(columns=["equity"])
    entry_bar = warmup_bars
    entry_price = float(df["open"].iat[entry_bar])
    btc_units = starting_capital / entry_price  # entry commission eats a tiny bit
    entry_commission = starting_capital * COMMISSION_RATE
    capital_after_entry = starting_capital - entry_commission
    btc_units = capital_after_entry / entry_price

    records = []
    for i in range(len(df)):
        if i < entry_bar:
            equity = starting_capital
        else:
            price = float(df["close"].iat[i])
            equity = btc_units * price
            if i == len(df) - 1:
                equity -= equity * COMMISSION_RATE  # exit commission
        records.append({"timestamp": df.index[i], "equity": equity})

    return pd.DataFrame(records).set_index("timestamp")


def solo_strategy_equity(df: pd.DataFrame, strategy_col: str,
                         starting_capital: float, notional: float) -> tuple[pd.DataFrame, int]:
    """Compute equity curve as if ONE strategy's persistent state drove position (100% of equity)."""
    warmup_bars = WARMUP_DAYS * BARS_PER_DAY
    states = df[strategy_col].astype(int).values
    opens = df["open"].values

    equity = starting_capital
    pos = 0
    entry_price = 0.0
    trade_notional = 0.0
    n_trades = 0
    records = []

    for i in range(len(df)):
        ts = df.index[i]
        if i < warmup_bars:
            records.append({"timestamp": ts, "equity": equity})
            continue

        target = states[i]
        if target != pos and i + 1 < len(df):
            exec_price = opens[i + 1]
            if pos != 0:
                price_ret = (exec_price - entry_price) / entry_price * pos
                gross = price_ret * trade_notional
                pnl = gross - 2 * trade_notional * COMMISSION_RATE
                equity += pnl
                if equity < 0:
                    equity = 0
                n_trades += 1
            if target != 0 and equity > 0:
                entry_price = exec_price
                trade_notional = equity
                pos = target
            else:
                pos = 0

        records.append({"timestamp": ts, "equity": equity})

    if pos != 0:
        exec_price = float(df["close"].iat[-1])
        price_ret = (exec_price - entry_price) / entry_price * pos
        gross = price_ret * trade_notional
        pnl = gross - 2 * trade_notional * COMMISSION_RATE
        equity += pnl
        if equity < 0:
            equity = 0
        n_trades += 1
        records[-1]["equity"] = equity

    return pd.DataFrame(records).set_index("timestamp"), n_trades


def year_breakdown(equity_curve: pd.DataFrame) -> dict[int, float]:
    """Annual returns from an equity curve."""
    eq = equity_curve["equity"]
    eq.index = pd.to_datetime(eq.index)
    yearly = eq.resample("YE").last()
    if len(yearly) < 2:
        return {}
    start_vals = eq.resample("YE").first()
    returns = (yearly - start_vals) / start_vals
    return {int(d.year): float(r) for d, r in returns.items() if not pd.isna(r)}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def ensure_tables(engine) -> None:
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS exp1_trades (
                id SERIAL PRIMARY KEY,
                run_id TEXT NOT NULL,
                entry_time TIMESTAMPTZ,
                exit_time TIMESTAMPTZ,
                side TEXT,
                entry_price DOUBLE PRECISION,
                exit_price DOUBLE PRECISION,
                bars_held INT,
                pnl DOUBLE PRECISION,
                commission DOUBLE PRECISION,
                exit_reason TEXT,
                net_vote_at_entry INT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS exp1_summary (
                id SERIAL PRIMARY KEY,
                run_id TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now(),
                label TEXT,
                metrics JSONB
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS exp1_equity_curve (
                id SERIAL PRIMARY KEY,
                run_id TEXT NOT NULL,
                label TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                equity DOUBLE PRECISION
            )
        """))
        conn.commit()


def save_results(engine, run_id: str, trades: list[dict],
                 summaries: list[dict], equity_curves: dict[str, pd.DataFrame]) -> None:
    with engine.connect() as conn:
        # Clear any prior run with same id
        conn.execute(text("DELETE FROM exp1_trades WHERE run_id = :r"), {"r": run_id})
        conn.execute(text("DELETE FROM exp1_summary WHERE run_id = :r"), {"r": run_id})
        conn.execute(text("DELETE FROM exp1_equity_curve WHERE run_id = :r"), {"r": run_id})

        if trades:
            trade_records = [{"run_id": run_id, **t} for t in trades]
            conn.execute(text("""
                INSERT INTO exp1_trades
                (run_id, entry_time, exit_time, side, entry_price, exit_price,
                 bars_held, pnl, commission, exit_reason, net_vote_at_entry)
                VALUES (:run_id, :entry_time, :exit_time, :side, :entry_price,
                        :exit_price, :bars_held, :pnl, :commission, :exit_reason,
                        :net_vote_at_entry)
            """), trade_records)

        for s in summaries:
            conn.execute(text("""
                INSERT INTO exp1_summary (run_id, label, metrics)
                VALUES (:run_id, :label, CAST(:metrics AS jsonb))
            """), {"run_id": run_id, "label": s["label"], "metrics": json.dumps(s)})

        for label, eq_df in equity_curves.items():
            # downsample to daily to keep the table small
            daily = eq_df["equity"].resample("D").last().dropna()
            records = [
                {"run_id": run_id, "label": label, "timestamp": ts, "equity": float(v)}
                for ts, v in daily.items()
            ]
            if records:
                conn.execute(text("""
                    INSERT INTO exp1_equity_curve (run_id, label, timestamp, equity)
                    VALUES (:run_id, :label, :timestamp, :equity)
                """), records)
        conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1 — Equal-weight ensemble baseline")
    parser.add_argument("--pair", default="BTCUSDT")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--notional", type=float, default=1000.0)
    parser.add_argument("--run-id", default="exp1_ensemble_v1")
    args = parser.parse_args()

    engine = get_engine()
    ensure_tables(engine)

    log.info("loading_data", pair=args.pair)
    df = load_exp1_data(engine, pair=args.pair)
    if df.empty or len(df) < WARMUP_DAYS * BARS_PER_DAY + 100:
        log.error("insufficient_data", rows=len(df))
        sys.exit(1)
    log.info("data_loaded", bars=len(df),
             start=str(df.index[0]), end=str(df.index[-1]))

    # Signal activity stats
    active = (df[STRATEGIES].abs().sum(axis=1) > 0).sum()
    log.info("signal_activity", bars_with_any_signal=int(active),
             pct=f"{100 * active / len(df):.1f}")

    # Strategy correlation matrix (on persistent state columns)
    corr = df[STRATEGIES].corr()
    log.info("strategy_correlation_matrix")
    print("\nStrategy state correlation matrix:")
    print(corr.round(3).to_string())

    # Run the ensemble backtest
    log.info("running_ensemble")
    trades, equity = simulate_ensemble(df, args.capital, args.notional)
    ensemble_metrics = compute_metrics(equity, trades, args.capital, "ensemble")

    # Buy-and-hold benchmark
    log.info("computing_buy_and_hold")
    bh_equity = buy_and_hold_equity(df, args.capital, args.notional)
    bh_metrics = compute_metrics(bh_equity, [], args.capital, "buy_and_hold")

    # Alpha
    alpha_total = ensemble_metrics["total_return"] - bh_metrics["total_return"]
    alpha_annualized = ensemble_metrics["annualized_return"] - bh_metrics["annualized_return"]

    # Per-strategy solo equity curves (free attribution)
    log.info("computing_per_strategy_solo")
    solo_equities = {}
    solo_metrics = []
    for strat in STRATEGIES:
        solo_eq, solo_n = solo_strategy_equity(df, strat, args.capital, args.notional)
        # Build a dummy trades list just so total_trades reports correctly
        fake_trades = [{"pnl": 0.0, "bars_held": 0}] * solo_n
        m = compute_metrics(solo_eq, fake_trades, args.capital, f"solo_{strat}")
        m["total_trades"] = solo_n
        solo_equities[strat] = solo_eq
        solo_metrics.append(m)

    # Year-by-year breakdown
    ensemble_years = year_breakdown(equity)
    bh_years = year_breakdown(bh_equity)
    year_alphas = {y: ensemble_years.get(y, 0) - bh_years.get(y, 0)
                   for y in sorted(set(ensemble_years) | set(bh_years))}

    # ---- Report ----
    print("\n" + "=" * 72)
    print("EXPERIMENT 1 — EQUAL-WEIGHT ENSEMBLE BASELINE")
    print("=" * 72)
    print(f"Period: {df.index[0]} → {df.index[-1]} ({ensemble_metrics['period_years']} years)")
    print(f"Bars:   {len(df):,}  (warmup {WARMUP_DAYS*BARS_PER_DAY:,} bars excluded)")
    print(f"Signal activity: {active:,} bars had at least one strategy non-flat")
    print()

    def print_row(label, m):
        print(f"  {label:<20s}  "
              f"ret={m['total_return']:+7.1%}  "
              f"ann={m['annualized_return']:+6.1%}  "
              f"Sharpe={m['sharpe']:6.2f}  "
              f"Calmar={m['calmar']:6.2f}  "
              f"MaxDD={m['max_drawdown']:6.1%}  "
              f"trades={m['total_trades']:5d}  "
              f"win={m['win_rate']:5.1%}")

    print("HEADLINE")
    print_row("ENSEMBLE", ensemble_metrics)
    print_row("BUY & HOLD", bh_metrics)
    print(f"\n  ALPHA (total):      {alpha_total:+.1%}")
    print(f"  ALPHA (annualized): {alpha_annualized:+.2%}")
    print()

    print("PER-STRATEGY SOLO ATTRIBUTION")
    for m in solo_metrics:
        print_row(m["label"].replace("solo_sig_", ""), m)
    print()

    print("YEAR-BY-YEAR RETURNS")
    print(f"  {'Year':<6s} {'Ensemble':>10s} {'Buy&Hold':>10s} {'Alpha':>10s}")
    for y in sorted(year_alphas):
        ens_r = ensemble_years.get(y, 0)
        bh_r = bh_years.get(y, 0)
        alpha = year_alphas[y]
        print(f"  {y:<6d} {ens_r:>+9.1%} {bh_r:>+9.1%} {alpha:>+9.1%}")
    print()

    # Persist
    all_equities = {"ensemble": equity, "buy_and_hold": bh_equity, **solo_equities}
    all_summaries = [ensemble_metrics, bh_metrics,
                     {"label": "alpha", "alpha_total": alpha_total,
                      "alpha_annualized": alpha_annualized,
                      "year_alphas": year_alphas},
                     *solo_metrics]
    save_results(engine, args.run_id, trades, all_summaries, all_equities)
    log.info("results_saved", run_id=args.run_id)


if __name__ == "__main__":
    main()
