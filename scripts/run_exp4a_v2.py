#!/usr/bin/env python3
"""Experiment 4a — Trailing stop on top of Exp 3 winner policy.

Motivated by Q3 of analyze_exp3_trades: 45.9% of losing trades had MFE
> +0.5% before reversing into losses. Adding a trailing stop should
convert many of those "gave back the gain" losers into scratches or
small wins.

Policy (fixed from Exp 3 winner):
  Ke = 3, Ne = 2, Kx = 0 (asymmetric gate with FLAT state)

New variables (this experiment):
  activation_mfe ∈ {0.5%, 1%, 2%}   # MFE threshold to activate trail stop
  trail_distance ∈ {0.3%, 0.5%, 1%} # distance below MFE for the stop

Breakeven floor: stop never drops below entry price once activated.

Look-ahead safety: stop level is computed from MFE through the PREVIOUS
bar; current bar's low/high is checked against that stop BEFORE updating
MFE with current bar data.

Precedence: trailing stop checked first each bar; if not triggered,
signal-based K_exit rule checks next.

Writes to aether_btc_lab: exp4a_v2_grid.

Usage:
    python scripts/run_exp4a.py
"""

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
    "sig_momentum", "sig_mean_reversion", "sig_trend_following",
    "sig_volatility_breakout", "sig_funding_volume",
]
WARMUP_DAYS = 30
BARS_PER_DAY = 96
COMMISSION_RATE = 0.0004
BARS_PER_YEAR = 96 * 365.25

# Exp 3 winner (fixed)
K_ENTRY = 3
N_ENTRY = 2
K_EXIT = 0

# Grid — v2 uses BTC-scale volatility ranges (ATR is ~1-2%, daily range ~3-5%)
ACTIVATION_MFE_VALUES = [0.02, 0.03, 0.05, 0.10]
TRAIL_DISTANCE_VALUES = [0.005, 0.01, 0.02]
MIN_TRADES_VALID = 30


def load_data(engine, pair: str = "BTCUSDT") -> pd.DataFrame:
    sql = text("""
        SELECT timestamp, open, high, low, close,
            (indicators->>'sig_momentum')::float           AS sig_momentum,
            (indicators->>'sig_mean_reversion')::float     AS sig_mean_reversion,
            (indicators->>'sig_trend_following')::float    AS sig_trend_following,
            (indicators->>'sig_volatility_breakout')::float AS sig_volatility_breakout,
            (indicators->>'sig_funding_volume')::float     AS sig_funding_volume
        FROM btc_indicators_15m
        WHERE pair = :pair AND indicators IS NOT NULL
          AND indicators ? 'sig_momentum'
        ORDER BY timestamp
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"pair": pair}, parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    for col in STRATEGIES:
        df[col] = df[col].fillna(0).astype(int)
    return df


def simulate_with_trailing(df: pd.DataFrame, activation_mfe: float,
                            trail_distance: float, starting_capital: float) -> dict:
    """Exp 3 winner policy + trailing stop layer."""
    bars = df.reset_index()
    n = len(bars)
    net_vote = bars[STRATEGIES].sum(axis=1).astype(int).values
    opens = bars["open"].values.astype(float)
    highs = bars["high"].values.astype(float)
    lows = bars["low"].values.astype(float)
    closes = bars["close"].values.astype(float)

    warmup_bars = WARMUP_DAYS * BARS_PER_DAY

    equity = starting_capital
    pos_side = 0
    entry_price = 0.0
    entry_idx = 0
    trade_notional = 0.0
    n_trades = 0
    n_wins = 0
    n_trail_exits = 0
    n_signal_exits = 0

    # Trailing stop state (per-position, reset on entry)
    mfe_price = 0.0           # best price seen so far (for long: max high; for short: min low)
    trail_stop_price = 0.0    # current stop level
    trail_active = False

    equity_arr = np.empty(n)

    for i in range(n):
        equity_arr[i] = equity

        # If in position, FIRST check if previous-bar trailing stop hits current bar's range
        stopped_out = False
        if pos_side != 0 and trail_active:
            if pos_side == 1 and lows[i] <= trail_stop_price:
                # Long stopped out
                exec_price = trail_stop_price
                price_ret = (exec_price - entry_price) / entry_price
                gross = price_ret * trade_notional
                pnl = gross - 2 * trade_notional * COMMISSION_RATE
                equity += pnl
                if equity < 0:
                    equity = 0
                if pnl > 0:
                    n_wins += 1
                n_trades += 1
                n_trail_exits += 1
                pos_side = 0
                trade_notional = 0.0
                trail_active = False
                stopped_out = True
            elif pos_side == -1 and highs[i] >= trail_stop_price:
                # Short stopped out
                exec_price = trail_stop_price
                price_ret = (entry_price - exec_price) / entry_price
                gross = price_ret * trade_notional
                pnl = gross - 2 * trade_notional * COMMISSION_RATE
                equity += pnl
                if equity < 0:
                    equity = 0
                if pnl > 0:
                    n_wins += 1
                n_trades += 1
                n_trail_exits += 1
                pos_side = 0
                trade_notional = 0.0
                trail_active = False
                stopped_out = True

        # Now update MFE using current bar (only if still in position)
        if pos_side != 0 and not stopped_out:
            if pos_side == 1:
                if highs[i] > mfe_price:
                    mfe_price = highs[i]
                current_mfe_pct = (mfe_price - entry_price) / entry_price
            else:
                if lows[i] < mfe_price or mfe_price == 0.0:
                    mfe_price = lows[i]
                current_mfe_pct = (entry_price - mfe_price) / entry_price

            # Activate / update trailing stop if MFE threshold exceeded
            if current_mfe_pct >= activation_mfe:
                if pos_side == 1:
                    # Stop at max(entry, mfe_price - trail_distance * entry_price)
                    raw_stop = mfe_price - trail_distance * entry_price
                    trail_stop_price = max(entry_price, raw_stop)
                else:
                    raw_stop = mfe_price + trail_distance * entry_price
                    trail_stop_price = min(entry_price, raw_stop)
                trail_active = True

        if i < warmup_bars or i < N_ENTRY:
            continue

        # If not stopped out by trail, check signal-based entry/exit
        if stopped_out:
            continue

        action = None
        if pos_side == 0:
            window = net_vote[i - N_ENTRY: i + 1]
            if np.all(window >= K_ENTRY):
                action = "enter_long"
            elif np.all(window <= -K_ENTRY):
                action = "enter_short"
        elif pos_side == 1:
            if net_vote[i] < K_EXIT:
                action = "exit"
        else:  # SHORT
            if net_vote[i] > -K_EXIT:
                action = "exit"

        if action is not None and i + 1 < n:
            exec_price = opens[i + 1]

            if action == "exit" and pos_side != 0:
                price_ret = (exec_price - entry_price) / entry_price * pos_side
                gross = price_ret * trade_notional
                pnl = gross - 2 * trade_notional * COMMISSION_RATE
                equity += pnl
                if equity < 0:
                    equity = 0
                if pnl > 0:
                    n_wins += 1
                n_trades += 1
                n_signal_exits += 1
                pos_side = 0
                trade_notional = 0.0
                trail_active = False
                mfe_price = 0.0

            elif action in ("enter_long", "enter_short") and equity > 0:
                entry_price = opens[i + 1]
                entry_idx = i + 1
                trade_notional = equity
                pos_side = 1 if action == "enter_long" else -1
                mfe_price = entry_price
                trail_active = False
                trail_stop_price = 0.0

    # Close hanging position
    if pos_side != 0 and trade_notional > 0:
        exec_price = float(closes[-1])
        price_ret = (exec_price - entry_price) / entry_price * pos_side
        gross = price_ret * trade_notional
        pnl = gross - 2 * trade_notional * COMMISSION_RATE
        equity += pnl
        if equity < 0:
            equity = 0
        if pnl > 0:
            n_wins += 1
        n_trades += 1
        n_signal_exits += 1
    equity_arr[-1] = equity

    # Metrics
    total_return = (equity - starting_capital) / starting_capital
    valid_mask = equity_arr[:-1] > 0
    rets = np.zeros(len(equity_arr) - 1)
    rets[valid_mask] = (equity_arr[1:][valid_mask] - equity_arr[:-1][valid_mask]) / equity_arr[:-1][valid_mask]
    rets = rets[~np.isnan(rets) & ~np.isinf(rets)]
    sharpe = (rets.mean() / rets.std()) * np.sqrt(BARS_PER_YEAR) if len(rets) > 1 and rets.std() > 0 else 0.0

    peak = np.maximum.accumulate(equity_arr)
    dd = (equity_arr - peak) / np.where(peak > 0, peak, 1)
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    years = (df.index[-1] - df.index[0]).days / 365.25
    ann = (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else -1.0
    calmar = ann / abs(max_dd) if max_dd < 0 else 0.0

    return {
        "n_trades": n_trades,
        "n_wins": n_wins,
        "win_rate": n_wins / n_trades if n_trades > 0 else 0.0,
        "n_trail_exits": n_trail_exits,
        "n_signal_exits": n_signal_exits,
        "final_equity": float(equity),
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "max_dd": float(max_dd),
    }


def buy_and_hold(df: pd.DataFrame, starting_capital: float) -> tuple[float, float, float]:
    warmup_bars = WARMUP_DAYS * BARS_PER_DAY
    entry_price = float(df["open"].iat[warmup_bars])
    entry_commission = starting_capital * COMMISSION_RATE
    btc_units = (starting_capital - entry_commission) / entry_price
    final = btc_units * float(df["close"].iat[-1])
    final -= final * COMMISSION_RATE
    tot = (final - starting_capital) / starting_capital
    return final, float(tot), 0.0  # sharpe not needed here


def ensure_table(engine) -> None:
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS exp4a_v2_grid (
                id SERIAL PRIMARY KEY,
                run_id TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now(),
                activation_mfe DOUBLE PRECISION,
                trail_distance DOUBLE PRECISION,
                n_trades INT,
                n_trail_exits INT,
                n_signal_exits INT,
                win_rate DOUBLE PRECISION,
                final_equity DOUBLE PRECISION,
                total_return DOUBLE PRECISION,
                sharpe DOUBLE PRECISION,
                calmar DOUBLE PRECISION,
                max_drawdown DOUBLE PRECISION
            )
        """))
        conn.commit()


def main() -> None:
    engine = get_engine()
    ensure_table(engine)
    log.info("loading_data")
    df = load_data(engine)
    log.info("data_loaded", bars=len(df))

    _, bh_ret, _ = buy_and_hold(df, 10000.0)

    # Include a reference baseline: Exp 3 winner with no trailing stop (activation=inf)
    print("\nRunning Exp 4a grid...")
    baseline = simulate_with_trailing(df, 999.0, 999.0, 10000.0)  # unreachable activation
    print(f"\nExp 3 winner baseline (no trail): ret={baseline['total_return']:+.1%}, "
          f"Sharpe={baseline['sharpe']:.2f}, trades={baseline['n_trades']}, "
          f"MaxDD={baseline['max_dd']:.1%}")

    results = []
    for act in ACTIVATION_MFE_VALUES:
        for td in TRAIL_DISTANCE_VALUES:
            m = simulate_with_trailing(df, act, td, 10000.0)
            alpha = m["total_return"] - bh_ret
            row = {"activation_mfe": act, "trail_distance": td, "alpha": alpha, **m}
            results.append(row)
            log.info("combo_done",
                     act=f"{act:.3f}", td=f"{td:.3f}",
                     trades=m["n_trades"],
                     trail=m["n_trail_exits"],
                     sig=m["n_signal_exits"],
                     ret=f"{m['total_return']:+.1%}",
                     sharpe=f"{m['sharpe']:.2f}")

    print("\n" + "=" * 110)
    print("EXPERIMENT 4a — TRAILING STOP ON EXP 3 WINNER (Ke=3, Ne=2, Kx=0)")
    print("=" * 110)
    print(f"Period: {df.index[0]} → {df.index[-1]}  ({len(df):,} bars)")
    print(f"Buy&Hold return: {bh_ret:+.1%}")
    print(f"Baseline (no trail): ret={baseline['total_return']:+.1%}, Sharpe={baseline['sharpe']:.2f}, "
          f"trades={baseline['n_trades']}, MaxDD={baseline['max_dd']:.1%}")
    print()
    print(f"{'act_mfe':>8} {'trail':>7} {'trades':>7} {'trail_ex':>9} {'sig_ex':>7} "
          f"{'win%':>6} {'return':>10} {'Sharpe':>7} {'Calmar':>7} {'MaxDD':>8} {'alpha':>10}")
    print("-" * 100)
    for r in results:
        print(f"{r['activation_mfe']:>7.3f}  {r['trail_distance']:>6.3f}  "
              f"{r['n_trades']:>7} {r['n_trail_exits']:>9} {r['n_signal_exits']:>7} "
              f"{r['win_rate']:>6.1%} {r['total_return']:>+10.1%} "
              f"{r['sharpe']:>7.2f} {r['calmar']:>7.2f} "
              f"{r['max_dd']:>8.1%} {r['alpha']:>+10.1%}")
    print()

    valid = [r for r in results if r["n_trades"] >= MIN_TRADES_VALID]
    if valid:
        best_sharpe = max(valid, key=lambda r: r["sharpe"])
        best_return = max(valid, key=lambda r: r["total_return"])
        best_calmar = max(valid, key=lambda r: r["calmar"])
        print("HIGHLIGHTS")
        print(f"  Best Sharpe: act={best_sharpe['activation_mfe']:.3f} "
              f"trail={best_sharpe['trail_distance']:.3f}  "
              f"ret={best_sharpe['total_return']:+.1%} Sharpe={best_sharpe['sharpe']:.2f} "
              f"DD={best_sharpe['max_dd']:.1%}")
        print(f"  Best return: act={best_return['activation_mfe']:.3f} "
              f"trail={best_return['trail_distance']:.3f}  "
              f"ret={best_return['total_return']:+.1%} Sharpe={best_return['sharpe']:.2f}")
        print(f"  Best Calmar: act={best_calmar['activation_mfe']:.3f} "
              f"trail={best_calmar['trail_distance']:.3f}  "
              f"ret={best_calmar['total_return']:+.1%} Calmar={best_calmar['calmar']:.2f} "
              f"DD={best_calmar['max_dd']:.1%}")

        # Compare to baseline
        best = best_sharpe
        delta_ret = best["total_return"] - baseline["total_return"]
        delta_sharpe = best["sharpe"] - baseline["sharpe"]
        delta_dd = best["max_dd"] - baseline["max_dd"]
        print(f"\n  Best-Sharpe combo vs no-trail baseline:")
        print(f"    return delta: {delta_ret:+.1%}")
        print(f"    sharpe delta: {delta_sharpe:+.2f}")
        print(f"    maxdd delta:  {delta_dd:+.1%}")

    with engine.connect() as conn:
        conn.execute(text("DELETE FROM exp4a_v2_grid WHERE run_id = :r"), {"r": "exp4a_v2"})
        conn.execute(text("""
            INSERT INTO exp4a_v2_grid
            (run_id, activation_mfe, trail_distance, n_trades, n_trail_exits, n_signal_exits,
             win_rate, final_equity, total_return, sharpe, calmar, max_drawdown)
            VALUES (:run_id, :activation_mfe, :trail_distance, :n_trades, :n_trail_exits,
                    :n_signal_exits, :win_rate, :final_equity, :total_return, :sharpe,
                    :calmar, :max_dd)
        """), [{"run_id": "exp4a_v2", **r} for r in results])
        conn.commit()
    log.info("results_saved", combos=len(results))


if __name__ == "__main__":
    main()
