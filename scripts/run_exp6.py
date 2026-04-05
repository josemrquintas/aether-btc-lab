#!/usr/bin/env python3
"""Experiment 6 — Strategy subset search (binary weighted ensemble).

Tests all 2^5 - 1 = 31 non-empty subsets of the 5 strategies, each with
the Exp 3 winner gate (Ne=2, Kx=0) and the Exp 4a-v2 winner trailing
stop (activation=5%, trail=0.5%). This is the simplest form of
weighted ensemble — weights are binary (0 = drop, 1 = include).

Ke (confluence threshold) scales with subset size:
  |S| = 1: Ke=1  (only option)
  |S| = 2: Ke=2  (unanimous)
  |S| = 3: Ke=2  (simple majority)
  |S| = 4: Ke=3  (3-of-4)
  |S| = 5: Ke=3  (3-of-5, matches Exp 3 winner)

Motivated by Exp 1 correlation finding: strategies are not equal.
Momentum is orthogonal, mean_reversion is strongly negatively correlated
with volatility_breakout and funding_volume. Some strategies may hurt
the ensemble more than help.

OVERFITTING CAVEAT: weights (subset selection) are being chosen in-sample.
A walk-forward follow-up is required before trusting any winning subset.

Writes to aether_btc_lab: exp6_subset.

Usage:
    python scripts/run_exp6.py
"""

import sys
from itertools import combinations
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

ALL_STRATEGIES = [
    "sig_momentum", "sig_mean_reversion", "sig_trend_following",
    "sig_volatility_breakout", "sig_funding_volume",
]
WARMUP_DAYS = 30
BARS_PER_DAY = 96
COMMISSION_RATE = 0.0004
BARS_PER_YEAR = 96 * 365.25

# Fixed from prior experiments
N_ENTRY = 2
K_EXIT = 0
TRAIL_ACTIVATION_MFE = 0.05
TRAIL_DISTANCE = 0.005
MIN_TRADES_VALID = 30


def ke_for_subset_size(size: int) -> int:
    """Majority rule: Ke = ceil(size / 2) + adjustment so |S|=5 gives Ke=3."""
    if size == 1:
        return 1
    if size == 2:
        return 2
    if size == 3:
        return 2
    if size == 4:
        return 3
    return 3  # size == 5


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
    for col in ALL_STRATEGIES:
        df[col] = df[col].fillna(0).astype(int)
    return df


def simulate_subset(df: pd.DataFrame, subset: tuple, starting_capital: float) -> dict:
    """Run asymmetric-gate + trailing-stop backtest using only the given strategy subset."""
    K_ENTRY = ke_for_subset_size(len(subset))
    bars = df.reset_index()
    n = len(bars)
    # Net vote uses ONLY subset columns
    net_vote = bars[list(subset)].sum(axis=1).astype(int).values
    opens = bars["open"].values.astype(float)
    highs = bars["high"].values.astype(float)
    lows = bars["low"].values.astype(float)
    closes = bars["close"].values.astype(float)

    warmup_bars = WARMUP_DAYS * BARS_PER_DAY

    equity = starting_capital
    pos_side = 0
    entry_price = 0.0
    trade_notional = 0.0
    n_trades = 0
    n_wins = 0
    mfe_price = 0.0
    trail_stop_price = 0.0
    trail_active = False
    equity_arr = np.empty(n)

    for i in range(n):
        equity_arr[i] = equity

        # Check trailing stop first
        stopped_out = False
        if pos_side != 0 and trail_active:
            if pos_side == 1 and lows[i] <= trail_stop_price:
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
                pos_side = 0
                trade_notional = 0.0
                trail_active = False
                stopped_out = True
            elif pos_side == -1 and highs[i] >= trail_stop_price:
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
                pos_side = 0
                trade_notional = 0.0
                trail_active = False
                stopped_out = True

        # Update MFE for next bar's stop
        if pos_side != 0 and not stopped_out:
            if pos_side == 1:
                if highs[i] > mfe_price:
                    mfe_price = highs[i]
                current_mfe_pct = (mfe_price - entry_price) / entry_price
            else:
                if lows[i] < mfe_price or mfe_price == 0.0:
                    mfe_price = lows[i]
                current_mfe_pct = (entry_price - mfe_price) / entry_price

            if current_mfe_pct >= TRAIL_ACTIVATION_MFE:
                if pos_side == 1:
                    raw_stop = mfe_price - TRAIL_DISTANCE * entry_price
                    trail_stop_price = max(entry_price, raw_stop)
                else:
                    raw_stop = mfe_price + TRAIL_DISTANCE * entry_price
                    trail_stop_price = min(entry_price, raw_stop)
                trail_active = True

        if i < warmup_bars or i < N_ENTRY:
            continue
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
        else:
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
                pos_side = 0
                trade_notional = 0.0
                trail_active = False
                mfe_price = 0.0
            elif action in ("enter_long", "enter_short") and equity > 0:
                entry_price = opens[i + 1]
                trade_notional = equity
                pos_side = 1 if action == "enter_long" else -1
                mfe_price = entry_price
                trail_active = False
                trail_stop_price = 0.0

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
    equity_arr[-1] = equity

    total_return = (equity - starting_capital) / starting_capital
    valid = equity_arr[:-1] > 0
    rets = np.zeros(len(equity_arr) - 1)
    rets[valid] = (equity_arr[1:][valid] - equity_arr[:-1][valid]) / equity_arr[:-1][valid]
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
        "win_rate": n_wins / n_trades if n_trades > 0 else 0.0,
        "final_equity": float(equity),
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "max_dd": float(max_dd),
        "K_entry": K_ENTRY,
    }


def buy_and_hold(df, starting_capital):
    warmup_bars = WARMUP_DAYS * BARS_PER_DAY
    entry_price = float(df["open"].iat[warmup_bars])
    entry_comm = starting_capital * COMMISSION_RATE
    btc_units = (starting_capital - entry_comm) / entry_price
    final = btc_units * float(df["close"].iat[-1])
    final -= final * COMMISSION_RATE
    return (final - starting_capital) / starting_capital


def ensure_table(engine):
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS exp6_subset (
                id SERIAL PRIMARY KEY,
                run_id TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now(),
                subset TEXT NOT NULL,
                subset_size INT,
                k_entry INT,
                n_trades INT,
                win_rate DOUBLE PRECISION,
                final_equity DOUBLE PRECISION,
                total_return DOUBLE PRECISION,
                sharpe DOUBLE PRECISION,
                calmar DOUBLE PRECISION,
                max_drawdown DOUBLE PRECISION,
                alpha_total DOUBLE PRECISION
            )
        """))
        conn.commit()


def main():
    engine = get_engine()
    ensure_table(engine)
    log.info("loading_data")
    df = load_data(engine)
    log.info("data_loaded", bars=len(df))
    bh_ret = buy_and_hold(df, 10000.0)

    # Enumerate all non-empty subsets
    results = []
    short_names = {s: s.replace("sig_", "")[:10] for s in ALL_STRATEGIES}

    print("\nRunning all 31 subset combinations...")
    for size in range(1, 6):
        for subset in combinations(ALL_STRATEGIES, size):
            m = simulate_subset(df, subset, 10000.0)
            alpha = m["total_return"] - bh_ret
            names = "+".join(short_names[s] for s in subset)
            row = {
                "subset": names,
                "size": size,
                "alpha": alpha,
                **m,
            }
            results.append(row)

    # Sort by Sharpe desc
    valid = [r for r in results if r["n_trades"] >= MIN_TRADES_VALID]
    valid.sort(key=lambda r: r["sharpe"], reverse=True)

    # Report
    print("\n" + "=" * 120)
    print("EXPERIMENT 6 — STRATEGY SUBSET SEARCH (binary weighted ensemble)")
    print("=" * 120)
    print(f"Base policy: Ne=2, Kx=0, trail act=5%, trail dist=0.5%")
    print(f"Buy&Hold: {bh_ret:+.1%}")
    print(f"Exp 4a-v2 winner (all 5 strategies): reference = +88.1%, Sharpe 0.93")
    print()
    print(f"Top 15 subsets by Sharpe (valid only, trades >= {MIN_TRADES_VALID}):")
    print()
    print(f"{'sz':>2} {'Ke':>3} {'subset':<60} {'trades':>7} {'ret':>8} {'Sharpe':>7} {'Calmar':>7} {'MaxDD':>8}")
    print("-" * 115)
    for r in valid[:15]:
        print(f"{r['size']:>2} {r['K_entry']:>3} {r['subset']:<60} "
              f"{r['n_trades']:>7} {r['total_return']:>+7.1%} "
              f"{r['sharpe']:>7.2f} {r['calmar']:>7.2f} {r['max_dd']:>8.1%}")
    print()

    # Leave-one-out attribution (size=4 subsets)
    size4 = [r for r in valid if r["size"] == 4]
    size4.sort(key=lambda r: r["sharpe"], reverse=True)
    full5 = next((r for r in results if r["size"] == 5), None)
    print("LEAVE-ONE-OUT (which strategy hurts the ensemble)")
    print("-" * 80)
    if full5:
        print(f"  Full 5 strategies: ret={full5['total_return']:+.1%} "
              f"Sharpe={full5['sharpe']:.2f} trades={full5['n_trades']}")
    for r in size4:
        removed = set(ALL_STRATEGIES) - set("sig_" + s for s in r["subset"].split("+"))
        delta_sharpe = r["sharpe"] - full5["sharpe"] if full5 else 0.0
        delta_ret = r["total_return"] - full5["total_return"] if full5 else 0.0
        print(f"  Removed: {r['subset']:<60} ret={r['total_return']:+7.1%} "
              f"Sharpe={r['sharpe']:+5.2f}  Δsharpe={delta_sharpe:+5.2f}  Δret={delta_ret:+6.1%}")
    print()

    # Persist
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM exp6_subset WHERE run_id = :r"), {"r": "exp6_subset_v1"})
        conn.execute(text("""
            INSERT INTO exp6_subset
            (run_id, subset, subset_size, k_entry, n_trades, win_rate,
             final_equity, total_return, sharpe, calmar, max_drawdown, alpha_total)
            VALUES (:run_id, :subset, :size, :k_entry, :n_trades, :win_rate,
                    :final_equity, :total_return, :sharpe, :calmar, :max_dd, :alpha)
        """), [{"run_id": "exp6_subset_v1", "k_entry": r["K_entry"], **r} for r in results])
        conn.commit()
    log.info("results_saved", combos=len(results))


if __name__ == "__main__":
    main()
