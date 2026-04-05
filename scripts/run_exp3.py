#!/usr/bin/env python3
"""Experiment 3 — Asymmetric entry/exit gated ensemble.

Adds a FLAT state and decouples entry/exit decisions:

  FLAT -> LONG/SHORT  (strict):
    requires |net_vote| >= K_entry for (N_entry + 1) consecutive bars
    in the same direction.

  LONG/SHORT -> FLAT  (loose):
    exit as soon as the current-direction net_vote drops below K_exit
    (no confirmation bars; immediate action on next bar open).

  Direct flips are NOT allowed — must pass through FLAT.

Grid:
  K_entry ∈ {2, 3, 4}
  N_entry ∈ {1, 2, 3, 4}
  K_exit  ∈ {1, 0, -1}           # higher = more sensitive (faster exit)
  N_exit  = 0                    # fixed

= 36 combos. Same equal weights, same full history, same costs as Exp 1/2.

K_exit interpretation for LONG position:
  K_exit =  1  -> exit when net_vote drops to 0 or below (signals split)
  K_exit =  0  -> exit when net_vote goes negative (signals oppose)
  K_exit = -1  -> exit when net_vote hits -1 or lower (mild opposition)

Symmetric for SHORT (exit if -net_vote < K_exit).

Writes to aether_btc_lab: exp3_grid (36 rows per run).

Usage:
    python scripts/run_exp3.py
"""

import argparse
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
BARS_PER_DAY = 96
COMMISSION_RATE = 0.0004
BARS_PER_YEAR = 96 * 365.25

K_ENTRY_VALUES = [2, 3, 4]
N_ENTRY_VALUES = [1, 2, 3, 4]
K_EXIT_VALUES = [1, 0, -1]
MIN_TRADES_FOR_VALID = 30  # combos with fewer trades flagged as insufficient


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


def simulate_asymmetric(df: pd.DataFrame, K_entry: int, N_entry: int,
                         K_exit: int, starting_capital: float) -> dict:
    """Run asymmetric-gate backtest. Returns metrics dict."""
    bars = df.reset_index()
    n = len(bars)
    net_vote = bars[STRATEGIES].sum(axis=1).astype(int).values
    opens = bars["open"].values.astype(float)

    warmup_bars = WARMUP_DAYS * BARS_PER_DAY

    equity = starting_capital
    pos_side = 0  # -1 SHORT, 0 FLAT, +1 LONG
    entry_price = 0.0
    trade_notional = 0.0
    n_trades = 0
    n_wins = 0
    equity_arr = np.empty(n)

    for i in range(n):
        equity_arr[i] = equity

        if i < warmup_bars or i < N_entry:
            continue

        action = None  # None, "enter_long", "enter_short", "exit"

        if pos_side == 0:
            # FLAT — check for strict entry condition
            window = net_vote[i - N_entry: i + 1]
            if np.all(window >= K_entry):
                action = "enter_long"
            elif np.all(window <= -K_entry):
                action = "enter_short"
        elif pos_side == 1:
            # LONG — check loose exit
            if net_vote[i] < K_exit:
                action = "exit"
        else:
            # SHORT — symmetric exit
            if net_vote[i] > -K_exit:
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

            elif action in ("enter_long", "enter_short") and equity > 0:
                entry_price = exec_price
                trade_notional = equity
                pos_side = 1 if action == "enter_long" else -1

    # Close hanging position at last bar
    if pos_side != 0 and trade_notional > 0:
        exec_price = float(bars["close"].iat[-1])
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
        "final_equity": float(equity),
        "total_return": float(total_return),
        "annualized": float(ann),
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "max_dd": float(max_dd),
    }


def buy_and_hold(df: pd.DataFrame, starting_capital: float) -> tuple[float, float, float, float]:
    warmup_bars = WARMUP_DAYS * BARS_PER_DAY
    entry_bar = warmup_bars
    entry_price = float(df["open"].iat[entry_bar])
    entry_commission = starting_capital * COMMISSION_RATE
    btc_units = (starting_capital - entry_commission) / entry_price

    eq_arr = np.empty(len(df))
    for i in range(len(df)):
        if i < entry_bar:
            eq_arr[i] = starting_capital
        else:
            eq_arr[i] = btc_units * float(df["close"].iat[i])
    eq_arr[-1] -= eq_arr[-1] * COMMISSION_RATE

    final = float(eq_arr[-1])
    total_return = (final - starting_capital) / starting_capital
    rets = np.diff(eq_arr) / eq_arr[:-1]
    rets = rets[~np.isnan(rets) & ~np.isinf(rets)]
    sharpe = (rets.mean() / rets.std()) * np.sqrt(BARS_PER_YEAR) if rets.std() > 0 else 0.0
    peak = np.maximum.accumulate(eq_arr)
    max_dd = float(((eq_arr - peak) / peak).min())
    return final, float(total_return), float(sharpe), max_dd


def ensure_table(engine) -> None:
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS exp3_grid (
                id SERIAL PRIMARY KEY,
                run_id TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now(),
                k_entry INT NOT NULL,
                n_entry INT NOT NULL,
                k_exit INT NOT NULL,
                n_trades INT,
                win_rate DOUBLE PRECISION,
                final_equity DOUBLE PRECISION,
                total_return DOUBLE PRECISION,
                sharpe DOUBLE PRECISION,
                calmar DOUBLE PRECISION,
                max_drawdown DOUBLE PRECISION,
                alpha_total DOUBLE PRECISION,
                insufficient_trades BOOLEAN
            )
        """))
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--run-id", default="exp3_asym_v1")
    args = parser.parse_args()

    engine = get_engine()
    ensure_table(engine)

    log.info("loading_data")
    df = load_data(engine)
    log.info("data_loaded", bars=len(df),
             start=str(df.index[0]), end=str(df.index[-1]))

    bh_final, bh_ret, bh_sharpe, bh_dd = buy_and_hold(df, args.capital)

    results = []
    print("\nRunning 36-combo grid...")
    for K_entry in K_ENTRY_VALUES:
        for N_entry in N_ENTRY_VALUES:
            for K_exit in K_EXIT_VALUES:
                m = simulate_asymmetric(df, K_entry, N_entry, K_exit, args.capital)
                alpha = m["total_return"] - bh_ret
                insufficient = m["n_trades"] < MIN_TRADES_FOR_VALID
                row = {
                    "K_entry": K_entry, "N_entry": N_entry, "K_exit": K_exit,
                    "alpha": alpha,
                    "insufficient": insufficient,
                    **m,
                }
                results.append(row)
                tag = " (!)" if insufficient else ""
                log.info("combo_done",
                         Ke=K_entry, Ne=N_entry, Kx=K_exit,
                         trades=m["n_trades"],
                         ret=f"{m['total_return']:+.1%}",
                         sharpe=f"{m['sharpe']:.2f}{tag}")

    # Report
    print("\n" + "=" * 110)
    print("EXPERIMENT 3 — ASYMMETRIC ENTRY/EXIT GATE (EQUAL WEIGHTS)")
    print("=" * 110)
    print(f"Period: {df.index[0]} → {df.index[-1]}  ({len(df):,} bars)")
    print(f"Buy&Hold: ret={bh_ret:+.1%}  Sharpe={bh_sharpe:.2f}  MaxDD={bh_dd:.1%}")
    print(f"(combos flagged '!' have < {MIN_TRADES_FOR_VALID} trades — statistically insufficient)")
    print()
    print(f"{'Ke':>3} {'Ne':>3} {'Kx':>3}  {'trades':>7} {'win%':>6} {'return':>10} "
          f"{'Sharpe':>7} {'Calmar':>7} {'MaxDD':>8} {'alpha':>10}")
    print("-" * 90)
    for r in results:
        flag = " !" if r["insufficient"] else "  "
        print(f"{r['K_entry']:>3} {r['N_entry']:>3} {r['K_exit']:>3}  "
              f"{r['n_trades']:>7} {r['win_rate']:>6.1%} {r['total_return']:>+10.1%} "
              f"{r['sharpe']:>7.2f} {r['calmar']:>7.2f} "
              f"{r['max_dd']:>8.1%} {r['alpha']:>+10.1%}{flag}")
    print()

    # Highlights — only consider statistically valid combos
    valid = [r for r in results if not r["insufficient"]]
    if valid:
        best_sharpe = max(valid, key=lambda r: r["sharpe"])
        best_return = max(valid, key=lambda r: r["total_return"])
        positive_alpha = [r for r in valid if r["alpha"] > 0]

        print("HIGHLIGHTS (valid combos only, trades >= 30)")
        print(f"  Best Sharpe:  Ke={best_sharpe['K_entry']} Ne={best_sharpe['N_entry']} "
              f"Kx={best_sharpe['K_exit']}  "
              f"ret={best_sharpe['total_return']:+.1%} "
              f"Sharpe={best_sharpe['sharpe']:.2f} "
              f"MaxDD={best_sharpe['max_dd']:.1%} "
              f"trades={best_sharpe['n_trades']}")
        print(f"  Best return:  Ke={best_return['K_entry']} Ne={best_return['N_entry']} "
              f"Kx={best_return['K_exit']}  "
              f"ret={best_return['total_return']:+.1%} "
              f"Sharpe={best_return['sharpe']:.2f} "
              f"alpha={best_return['alpha']:+.1%} "
              f"trades={best_return['n_trades']}")
        print(f"  Combos with positive alpha vs buy&hold: {len(positive_alpha)}/{len(valid)}")
    else:
        print("ALL combos produced insufficient trades — grid may be too strict.")
    print()

    # Persist
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM exp3_grid WHERE run_id = :r"), {"r": args.run_id})
        conn.execute(text("""
            INSERT INTO exp3_grid
            (run_id, k_entry, n_entry, k_exit, n_trades, win_rate,
             final_equity, total_return, sharpe, calmar, max_drawdown,
             alpha_total, insufficient_trades)
            VALUES (:run_id, :k_entry, :n_entry, :k_exit, :n_trades, :win_rate,
                    :final_equity, :total_return, :sharpe, :calmar, :max_dd,
                    :alpha, :insufficient)
        """), [{
            "run_id": args.run_id,
            "k_entry": r["K_entry"], "n_entry": r["N_entry"], "k_exit": r["K_exit"],
            "n_trades": r["n_trades"], "win_rate": r["win_rate"],
            "final_equity": r["final_equity"], "total_return": r["total_return"],
            "sharpe": r["sharpe"], "calmar": r["calmar"], "max_dd": r["max_dd"],
            "alpha": r["alpha"], "insufficient": r["insufficient"],
        } for r in results])
        conn.commit()
    log.info("results_saved", run_id=args.run_id, combos=len(results))


if __name__ == "__main__":
    main()
