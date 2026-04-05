#!/usr/bin/env python3
"""Experiment 2 — Confluence + Confirmation gated ensemble.

Builds on Experiment 1 by adding a two-parameter confidence gate:
  K (confluence threshold): require |net_vote| >= K to enter/flip
  N (confirmation bars):    require the same |net_vote| >= K condition
                             to have held for N previous bars (strict)

Runs a grid over K × N, same equal-weight ensemble and full history.

The position rule:
  - Compute net_vote at every bar (sum of persistent strategy states).
  - A direction d is "confirmed" at bar i if sign(net_vote[j]) == d AND
    |net_vote[j]| >= K for j in [i-N, i].
  - If a new direction is confirmed and differs from current position,
    flip at next bar open (100% equity, 1x leverage).
  - Otherwise hold current position.
  - Final bar closes any open position.

No SL, no TP, no time stop, no slippage.
Commission 0.04% per side. 30-day warmup.

Writes to aether_btc_lab: exp2_grid (one row per K×N combo).

Usage:
    python scripts/run_exp2.py
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
BARS_PER_DAY = 96
COMMISSION_RATE = 0.0004
BARS_PER_YEAR = 96 * 365.25

K_VALUES = [1, 2, 3, 4]
N_VALUES = [0, 1, 2, 4]


# ---------------------------------------------------------------------------
# Data loading (copy of Exp 1 loader)
# ---------------------------------------------------------------------------
def load_exp2_data(engine, pair: str = "BTCUSDT") -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# Gated simulator
# ---------------------------------------------------------------------------
def simulate_gated(df: pd.DataFrame, K: int, N: int,
                   starting_capital: float) -> tuple[int, float, float, float, float, float, int]:
    """Return (n_trades, final_equity, total_return, sharpe, calmar, max_dd, n_wins)."""
    bars = df.reset_index()
    n = len(bars)
    net_vote = bars[STRATEGIES].sum(axis=1).astype(int).values
    opens = bars["open"].values.astype(float)
    timestamps = bars["timestamp"].values

    warmup_bars = WARMUP_DAYS * BARS_PER_DAY

    equity = starting_capital
    pos_side = 0
    entry_price = 0.0
    trade_notional = 0.0
    n_trades = 0
    n_wins = 0
    equity_path = [starting_capital]

    for i in range(n):
        if i < warmup_bars or i < N:
            equity_path.append(equity)
            continue

        # Check confirmation: for each bar in window [i-N, i], net_vote sign same and |vote|>=K
        window = net_vote[i - N: i + 1]
        # Direction candidates
        target = 0
        if np.all(window >= K):
            target = 1
        elif np.all(window <= -K):
            target = -1

        if target != 0 and target != pos_side and i + 1 < n:
            exec_price = opens[i + 1]

            # Close existing (if any)
            if pos_side != 0 and trade_notional > 0:
                price_ret = (exec_price - entry_price) / entry_price * pos_side
                gross = price_ret * trade_notional
                pnl = gross - 2 * trade_notional * COMMISSION_RATE
                equity += pnl
                if equity < 0:
                    equity = 0
                if pnl > 0:
                    n_wins += 1
                n_trades += 1

            # Open new
            if equity > 0:
                entry_price = exec_price
                trade_notional = equity
                pos_side = target

        equity_path.append(equity)

    # Close hanging
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

    # Metrics
    eq_arr = np.array(equity_path[1:])  # skip the prepended starting_capital
    total_return = (equity - starting_capital) / starting_capital

    # Sharpe on bar returns
    if len(eq_arr) > 1:
        valid = eq_arr[:-1] > 0
        rets = np.zeros(len(eq_arr) - 1)
        rets[valid] = (eq_arr[1:][valid] - eq_arr[:-1][valid]) / eq_arr[:-1][valid]
        rets = rets[~np.isnan(rets) & ~np.isinf(rets)]
        if len(rets) > 1 and rets.std() > 0:
            sharpe = (rets.mean() / rets.std()) * np.sqrt(BARS_PER_YEAR)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Max DD
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / np.where(peak > 0, peak, 1)
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    # Annualized + Calmar
    years = (df.index[-1] - df.index[0]).days / 365.25
    ann = (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else -1.0
    calmar = ann / abs(max_dd) if max_dd < 0 else 0.0

    return n_trades, float(equity), float(total_return), float(sharpe), float(calmar), float(max_dd), n_wins


def buy_and_hold(df: pd.DataFrame, starting_capital: float) -> tuple[float, float, float, float]:
    """Return (final_equity, total_return, sharpe, max_dd)."""
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
    eq_arr[-1] -= eq_arr[-1] * COMMISSION_RATE  # exit commission

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
            CREATE TABLE IF NOT EXISTS exp2_grid (
                id SERIAL PRIMARY KEY,
                run_id TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now(),
                k_confluence INT NOT NULL,
                n_confirmation INT NOT NULL,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2 — confluence + confirmation grid")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--run-id", default="exp2_grid_v1")
    args = parser.parse_args()

    engine = get_engine()
    ensure_table(engine)

    log.info("loading_data")
    df = load_exp2_data(engine)
    log.info("data_loaded", bars=len(df),
             start=str(df.index[0]), end=str(df.index[-1]))

    # Buy-and-hold benchmark
    bh_final, bh_ret, bh_sharpe, bh_dd = buy_and_hold(df, args.capital)

    # Run the grid
    results = []
    print("\nRunning grid...")
    for K in K_VALUES:
        for N in N_VALUES:
            n_tr, final, tot, sh, cal, dd, wins = simulate_gated(df, K, N, args.capital)
            win_rate = wins / n_tr if n_tr > 0 else 0.0
            alpha = tot - bh_ret
            results.append({
                "K": K, "N": N,
                "trades": n_tr,
                "win_rate": win_rate,
                "final_equity": final,
                "total_return": tot,
                "sharpe": sh,
                "calmar": cal,
                "max_dd": dd,
                "alpha": alpha,
            })
            log.info("combo_done", K=K, N=N, trades=n_tr,
                     ret=f"{tot:+.1%}", sharpe=f"{sh:.2f}", alpha=f"{alpha:+.1%}")

    # Report
    print("\n" + "=" * 100)
    print("EXPERIMENT 2 — CONFLUENCE + CONFIRMATION GATE (EQUAL WEIGHTS)")
    print("=" * 100)
    print(f"Period: {df.index[0]} → {df.index[-1]}  ({len(df):,} bars)")
    print(f"Buy&Hold: ret={bh_ret:+.1%}  Sharpe={bh_sharpe:.2f}  MaxDD={bh_dd:.1%}")
    print()
    print(f"{'K':>2} {'N':>2} {'trades':>7} {'win%':>6} {'return':>10} {'Sharpe':>7} {'Calmar':>7} {'MaxDD':>8} {'alpha':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['K']:>2} {r['N']:>2} {r['trades']:>7} "
              f"{r['win_rate']:>6.1%} {r['total_return']:>+10.1%} "
              f"{r['sharpe']:>7.2f} {r['calmar']:>7.2f} "
              f"{r['max_dd']:>8.1%} {r['alpha']:>+10.1%}")
    print()

    # Best by Sharpe and by total return
    best_sharpe = max(results, key=lambda r: r["sharpe"])
    best_return = max(results, key=lambda r: r["total_return"])
    positive_alpha = [r for r in results if r["alpha"] > 0]

    print("HIGHLIGHTS")
    print(f"  Best Sharpe:  K={best_sharpe['K']} N={best_sharpe['N']} "
          f"ret={best_sharpe['total_return']:+.1%} "
          f"Sharpe={best_sharpe['sharpe']:.2f} "
          f"trades={best_sharpe['trades']}")
    print(f"  Best return:  K={best_return['K']} N={best_return['N']} "
          f"ret={best_return['total_return']:+.1%} "
          f"alpha={best_return['alpha']:+.1%} "
          f"trades={best_return['trades']}")
    print(f"  Combos with positive alpha vs buy&hold: {len(positive_alpha)}/{len(results)}")
    print()

    # Persist
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM exp2_grid WHERE run_id = :r"), {"r": args.run_id})
        conn.execute(text("""
            INSERT INTO exp2_grid
            (run_id, k_confluence, n_confirmation, n_trades, win_rate,
             final_equity, total_return, sharpe, calmar, max_drawdown, alpha_total)
            VALUES (:run_id, :K, :N, :trades, :win_rate, :final_equity,
                    :total_return, :sharpe, :calmar, :max_dd, :alpha)
        """), [{**r, "run_id": args.run_id} for r in results])
        conn.commit()
    log.info("results_saved", run_id=args.run_id, combos=len(results))


if __name__ == "__main__":
    main()
