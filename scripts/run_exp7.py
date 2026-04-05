#!/usr/bin/env python3
"""Experiment 7 — Walk-forward validation of Exp 6 subset search.

Splits the full history in half:
  Training period: 2019-09-01 → 2022-12-31 (~3.3 years)
  Test period:     2023-01-01 → 2026-04-05 (~3.3 years)

Phase 1: Run all 31 subsets on training data, find training winner.
Phase 2: Run all 31 subsets on test data (unseen during Phase 1 decision).
Phase 3: Compare rankings. Does the training winner also win on test?
         Are the Exp 6 favorites (4-strat drop-MR, 2-strat trend+funding)
         robust across both periods?
Phase 4: Leave-one-out in each period independently. Is mean_reversion
         the worst in BOTH periods (structural) or only one (overfit)?

VERDICT logic:
  GOOD:   training winner's test Sharpe >= 0.7 * training Sharpe
  OK:     training winner's test Sharpe >= 0.4 * training Sharpe
  BAD:    test Sharpe collapses (< 0.4 * training) — subset was overfit

Writes to aether_btc_lab: exp7_walkforward.

Usage:
    python scripts/run_exp7.py
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
SHORT_NAMES = {s: s.replace("sig_", "")[:10] for s in ALL_STRATEGIES}

WARMUP_DAYS = 30
BARS_PER_DAY = 96
COMMISSION_RATE = 0.0004
BARS_PER_YEAR = 96 * 365.25

# Fixed policy from prior experiments
N_ENTRY = 2
K_EXIT = 0
TRAIL_ACTIVATION_MFE = 0.05
TRAIL_DISTANCE = 0.005
MIN_TRADES_VALID = 20  # lower than Exp 6 since each period is half the data

# Walk-forward split boundary
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"

# Subsets of interest to always highlight
HIGHLIGHT_SUBSETS = {
    "full_5": tuple(ALL_STRATEGIES),
    "drop_mean_rev": tuple(s for s in ALL_STRATEGIES if s != "sig_mean_reversion"),
    "trend_funding_2": ("sig_trend_following", "sig_funding_volume"),
}


def ke_for_subset_size(size: int) -> int:
    if size == 1: return 1
    if size == 2: return 2
    if size == 3: return 2
    if size == 4: return 3
    return 3


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
    """Run asymmetric gate + trailing stop on a given strategy subset.

    IMPORTANT: reset warmup based on the passed-in df so each period can
    be simulated independently.
    """
    if len(df) < WARMUP_DAYS * BARS_PER_DAY + 100:
        return {"n_trades": 0, "win_rate": 0.0, "final_equity": starting_capital,
                "total_return": 0.0, "sharpe": 0.0, "calmar": 0.0, "max_dd": 0.0,
                "K_entry": 0, "error": "insufficient_data"}

    K_ENTRY = ke_for_subset_size(len(subset))
    bars = df.reset_index()
    n = len(bars)
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

        stopped_out = False
        if pos_side != 0 and trail_active:
            if pos_side == 1 and lows[i] <= trail_stop_price:
                exec_price = trail_stop_price
                price_ret = (exec_price - entry_price) / entry_price
                gross = price_ret * trade_notional
                pnl = gross - 2 * trade_notional * COMMISSION_RATE
                equity += pnl
                if equity < 0: equity = 0
                if pnl > 0: n_wins += 1
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
                if equity < 0: equity = 0
                if pnl > 0: n_wins += 1
                n_trades += 1
                pos_side = 0
                trade_notional = 0.0
                trail_active = False
                stopped_out = True

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
                if equity < 0: equity = 0
                if pnl > 0: n_wins += 1
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
        if equity < 0: equity = 0
        if pnl > 0: n_wins += 1
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
    if len(df) <= warmup_bars:
        return 0.0
    entry_price = float(df["open"].iat[warmup_bars])
    entry_comm = starting_capital * COMMISSION_RATE
    btc_units = (starting_capital - entry_comm) / entry_price
    final = btc_units * float(df["close"].iat[-1])
    final -= final * COMMISSION_RATE
    return (final - starting_capital) / starting_capital


def subset_name(subset: tuple) -> str:
    return "+".join(SHORT_NAMES[s] for s in subset)


def run_period(df_period: pd.DataFrame, period_name: str) -> list[dict]:
    """Run all 31 subsets on a given period, return results list."""
    results = []
    for size in range(1, 6):
        for subset in combinations(ALL_STRATEGIES, size):
            m = simulate_subset(df_period, subset, 10000.0)
            results.append({
                "period": period_name,
                "subset": subset_name(subset),
                "subset_tuple": subset,
                "size": size,
                **m,
            })
    return results


def ensure_table(engine):
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS exp7_walkforward (
                id SERIAL PRIMARY KEY,
                run_id TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now(),
                period TEXT NOT NULL,
                subset TEXT NOT NULL,
                subset_size INT,
                k_entry INT,
                n_trades INT,
                win_rate DOUBLE PRECISION,
                final_equity DOUBLE PRECISION,
                total_return DOUBLE PRECISION,
                sharpe DOUBLE PRECISION,
                calmar DOUBLE PRECISION,
                max_drawdown DOUBLE PRECISION
            )
        """))
        conn.commit()


def main():
    engine = get_engine()
    ensure_table(engine)
    log.info("loading_data")
    df = load_data(engine)
    log.info("data_loaded", bars=len(df),
             start=str(df.index[0]), end=str(df.index[-1]))

    # Split
    train_df = df[df.index <= TRAIN_END]
    test_df = df[df.index >= TEST_START]
    log.info("split", train_bars=len(train_df), test_bars=len(test_df),
             train_start=str(train_df.index[0]), train_end=str(train_df.index[-1]),
             test_start=str(test_df.index[0]), test_end=str(test_df.index[-1]))

    bh_train = buy_and_hold(train_df, 10000.0)
    bh_test = buy_and_hold(test_df, 10000.0)
    log.info("buy_and_hold",
             train=f"{bh_train:+.1%}",
             test=f"{bh_test:+.1%}")

    # Phase 1: Training period
    log.info("phase_1_training")
    train_results = run_period(train_df, "train")

    # Phase 2: Test period
    log.info("phase_2_test")
    test_results = run_period(test_df, "test")

    # Join by subset
    train_by_name = {r["subset"]: r for r in train_results}
    test_by_name = {r["subset"]: r for r in test_results}

    # Training winners (by Sharpe, minimum trade count)
    train_valid = [r for r in train_results if r["n_trades"] >= MIN_TRADES_VALID]
    train_valid.sort(key=lambda r: r["sharpe"], reverse=True)

    test_valid = [r for r in test_results if r["n_trades"] >= MIN_TRADES_VALID]
    test_valid.sort(key=lambda r: r["sharpe"], reverse=True)

    # ---- Report ----
    print("\n" + "=" * 118)
    print("EXPERIMENT 7 — WALK-FORWARD VALIDATION OF SUBSET SEARCH")
    print("=" * 118)
    print(f"Train: {train_df.index[0]} → {train_df.index[-1]}  ({len(train_df):,} bars)")
    print(f"Test:  {test_df.index[0]} → {test_df.index[-1]}  ({len(test_df):,} bars)")
    print(f"Train Buy&Hold: {bh_train:+.1%}   |   Test Buy&Hold: {bh_test:+.1%}")
    print()

    # -- Top 10 training winners with their test-period performance --
    print("TOP 10 TRAINING WINNERS (ranked by training Sharpe)")
    print(f"{'rank':>4}  {'subset':<55} {'train_sh':>9} {'train_ret':>10} {'test_sh':>9} {'test_ret':>10} {'test_dd':>8}")
    print("-" * 118)
    for rank, r in enumerate(train_valid[:10], 1):
        t = test_by_name.get(r["subset"])
        if t:
            print(f"{rank:>4}  {r['subset']:<55} "
                  f"{r['sharpe']:>9.2f} {r['total_return']:>+9.1%} "
                  f"{t['sharpe']:>9.2f} {t['total_return']:>+9.1%} "
                  f"{t['max_dd']:>8.1%}")
    print()

    # -- Top 10 test winners (independent ranking, to check overlap) --
    print("TOP 10 TEST WINNERS (ranked by test Sharpe — ex-post, for rank-stability comparison)")
    print(f"{'rank':>4}  {'subset':<55} {'test_sh':>9} {'test_ret':>10} {'train_sh':>9} {'train_ret':>10}")
    print("-" * 118)
    for rank, r in enumerate(test_valid[:10], 1):
        tr = train_by_name.get(r["subset"])
        if tr:
            print(f"{rank:>4}  {r['subset']:<55} "
                  f"{r['sharpe']:>9.2f} {r['total_return']:>+9.1%} "
                  f"{tr['sharpe']:>9.2f} {tr['total_return']:>+9.1%}")
    print()

    # -- Highlight candidates --
    print("HIGHLIGHT CANDIDATES (stability of our named policies)")
    print(f"{'policy':<25} {'train_sh':>9} {'train_ret':>10} {'train_dd':>9} "
          f"{'test_sh':>9} {'test_ret':>10} {'test_dd':>9}")
    print("-" * 100)
    for label, subset in HIGHLIGHT_SUBSETS.items():
        name = subset_name(subset)
        tr = train_by_name.get(name)
        te = test_by_name.get(name)
        if tr and te:
            print(f"{label:<25} "
                  f"{tr['sharpe']:>9.2f} {tr['total_return']:>+9.1%} {tr['max_dd']:>+9.1%} "
                  f"{te['sharpe']:>9.2f} {te['total_return']:>+9.1%} {te['max_dd']:>+9.1%}")
    print()

    # -- Leave-one-out in each period independently --
    print("LEAVE-ONE-OUT — is mean_reversion the worst in BOTH periods?")
    full5 = tuple(ALL_STRATEGIES)
    print(f"  Baseline (all 5):")
    tr_full = train_by_name.get(subset_name(full5))
    te_full = test_by_name.get(subset_name(full5))
    if tr_full: print(f"    train: Sharpe {tr_full['sharpe']:+.2f}, ret {tr_full['total_return']:+.1%}")
    if te_full: print(f"    test:  Sharpe {te_full['sharpe']:+.2f}, ret {te_full['total_return']:+.1%}")
    print()
    print(f"  {'removed':<20} {'train Δret':>12} {'train Δsh':>11} {'test Δret':>12} {'test Δsh':>11}")
    for removed_strategy in ALL_STRATEGIES:
        subset_4 = tuple(s for s in ALL_STRATEGIES if s != removed_strategy)
        name_4 = subset_name(subset_4)
        tr = train_by_name.get(name_4)
        te = test_by_name.get(name_4)
        if tr and te and tr_full and te_full:
            dr_tr = tr["total_return"] - tr_full["total_return"]
            ds_tr = tr["sharpe"] - tr_full["sharpe"]
            dr_te = te["total_return"] - te_full["total_return"]
            ds_te = te["sharpe"] - te_full["sharpe"]
            print(f"  {removed_strategy.replace('sig_',''):<20} "
                  f"{dr_tr:>+11.1%} {ds_tr:>+11.2f} "
                  f"{dr_te:>+11.1%} {ds_te:>+11.2f}")
    print()

    # -- The big verdict --
    if train_valid:
        train_winner = train_valid[0]
        test_of_winner = test_by_name.get(train_winner["subset"])
        if test_of_winner:
            train_sh = train_winner["sharpe"]
            test_sh = test_of_winner["sharpe"]
            if train_sh > 0:
                ratio = test_sh / train_sh
            else:
                ratio = 0.0

            print("VERDICT")
            print(f"  Training winner: {train_winner['subset']}")
            print(f"    train Sharpe: {train_sh:.2f}  train return: {train_winner['total_return']:+.1%}")
            print(f"    test  Sharpe: {test_sh:.2f}  test  return: {test_of_winner['total_return']:+.1%}  "
                  f"MaxDD {test_of_winner['max_dd']:+.1%}")
            print(f"    test/train Sharpe ratio: {ratio:.2f}")
            print()
            if ratio >= 0.7:
                print("  ✓ GOOD — test-period Sharpe retains >= 70% of training Sharpe.")
                print("    The subset selection appears robust. Candidate suitable for next phase.")
            elif ratio >= 0.4:
                print("  ~ OK — test-period Sharpe retains 40-70% of training.")
                print("    Edge may be real but partially in-sample. Treat with caution.")
            else:
                print("  ✗ BAD — test-period Sharpe collapsed to < 40% of training.")
                print("    The subset was almost certainly overfit. Cannot trust the Exp 6 numbers.")
        print()

    # Persist
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM exp7_walkforward WHERE run_id = :r"), {"r": "exp7_v1"})
        for r in train_results + test_results:
            conn.execute(text("""
                INSERT INTO exp7_walkforward
                (run_id, period, subset, subset_size, k_entry, n_trades, win_rate,
                 final_equity, total_return, sharpe, calmar, max_drawdown)
                VALUES (:run_id, :period, :subset, :size, :k_entry, :n_trades, :win_rate,
                        :final_equity, :total_return, :sharpe, :calmar, :max_dd)
            """), {
                "run_id": "exp7_v1",
                "period": r["period"],
                "subset": r["subset"],
                "size": r["size"],
                "k_entry": r.get("K_entry", 0),
                "n_trades": r["n_trades"],
                "win_rate": r["win_rate"],
                "final_equity": r["final_equity"],
                "total_return": r["total_return"],
                "sharpe": r["sharpe"],
                "calmar": r["calmar"],
                "max_dd": r["max_dd"],
            })
        conn.commit()
    log.info("results_saved", train_rows=len(train_results), test_rows=len(test_results))


if __name__ == "__main__":
    main()
