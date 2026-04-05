# Signal Experiments — From Null Result to Breakthrough

**Status:** In-sample results only. Walk-forward validation required before any live use.
**Period tested:** 2019-09-01 → 2026-04-05 (6.59 years, 231,081 × 15m candles)
**Benchmark:** Buy & hold BTC: +713.3% return, Sharpe 0.82, MaxDD -77.2%
**Database:** `aether_btc_lab` (cloned from prod, isolated, own refresh loop)
**Branch:** `feat/exp1-ensemble-baseline`

---

## TL;DR

Starting from 5 precomputed signal strategies (momentum, mean_reversion, trend_following, volatility_breakout, funding_volume), a six-experiment sequence established that:

1. **Equal-weight voting is catastrophic** (-100% return) due to commission churn.
2. **Gate-based churn reduction** (Exp 2-3) makes the strategy tradable (Sharpe 0.62-0.93, MaxDD -10-12%).
3. **Trailing stops at BTC-appropriate scale** (5% MFE activation) add +37% return and +0.31 Sharpe.
4. **Mean_reversion was poisoning the ensemble.** Dropping it unlocks +654% additional return.

Two candidate policies beat buy-and-hold on both return AND risk metrics for the first time:

| Policy | Strategies | Return | Sharpe | MaxDD | Trades |
|---|---|---|---|---|---|
| **4-strat (best risk-adjusted)** | drop mean_reversion | **+742.5%** | **1.18** | **-29.2%** | 992 |
| **2-strat (best absolute return)** | trend_following + funding_volume | **+1,048.3%** | **1.19** | -46.8% | 1,206 |
| Buy & hold | — | +713.3% | 0.82 | -77.2% | — |

---

## The Sequence

### Experiment 1 — Equal-weight ensemble baseline

**Question:** Do the 5 raw signals have tradable edge with a minimal execution policy?

**Setup:** Net vote = sum of persistent strategy states (-1/0/+1). Enter/flip on any majority. 100% equity, 1x leverage, no SL/TP, 0.04% commission. Full history.

**Result:** **Catastrophic -100% return, 13,339 trades, Sharpe -2.52.**

**Diagnosis:** 13,339 trades × 0.08% round-trip = ~1,067% cumulative commission drag. Fees alone wiped the account regardless of any signal edge. The bottleneck was not signal quality — it was trade frequency.

**Correlation matrix (recorded for later use):**

| | momentum | mean_rev | trend | vol_break | funding |
|---|---|---|---|---|---|
| momentum | 1.00 | -0.06 | -0.03 | +0.03 | +0.12 |
| mean_rev | -0.06 | 1.00 | -0.27 | **-0.52** | **-0.51** |
| trend | -0.03 | -0.27 | 1.00 | +0.40 | +0.19 |
| vol_break | +0.03 | -0.52 | +0.40 | 1.00 | +0.39 |
| funding | +0.12 | -0.51 | +0.19 | +0.39 | 1.00 |

**mean_reversion was strongly negatively correlated with vol_break (-0.52) and funding (-0.51).** This clue would become central in Exp 6.

---

### Experiment 2 — Symmetric confluence + confirmation gate

**Question:** Can trade-frequency reduction alone turn the ensemble profitable?

**Setup:** 16-combo grid over K (vote threshold) × N (confirmation bars). Flip position only when `|net_vote| ≥ K` for `N+1` consecutive bars.

**Result:** **K=2, N=2** wins with **+136.4% return, Sharpe 0.50, MaxDD -80.3%, 1,380 trades.**

**Lesson:** Churn reduction works — the signal edge was real, just drowned in commissions. But the winning combo still had -575% alpha vs buy-and-hold, and a -80% drawdown is not tradable with real money.

---

### Experiment 3 — Asymmetric gate with FLAT state

**Question:** Does decoupling entry and exit conservatism improve the policy?

**Setup:** 3-state machine (FLAT/LONG/SHORT). Strict entry (`K_entry`, `N_entry` confirmation). Loose exit (`K_exit`, no confirmation delay). Direct flips disallowed — must pass through FLAT. 36-combo grid.

**Result:** **K_entry=3, N_entry=2, K_exit=0** wins with **+50.8% return, Sharpe 0.62, MaxDD -12.3%, 117 trades.**

**Lesson:** Asymmetric gating is a clean risk reducer — 6x smaller drawdown than Exp 2 winner for a modest return cost. First result that could survive contact with real money (from a psychological drawdown standpoint). Still -660% alpha vs buy-and-hold.

---

### Trade Analysis (pre-Exp 4)

**Question:** What are the losing trades actually doing?

**Method:** Rerun Exp 3 winner with full trade capture. Compute per-trade MAE/MFE/regime.

**Findings:**

- **Direction hypothesis wrong:** Both LONG (+$3,702) and SHORT (+$1,377) profitable. Shorts aren't bleeding.
- **Regime matrix surprising:** Only "bull SHORT" cell is dead ($174 on 43 trades — basically zero). Other three cells carry the strategy.
- **Loser diagnostic (key finding):** 45.9% of losing trades had MFE > +0.5% before reversing to loss. Winners held 3x longer than losers (69 median bars vs 22). **Strategy is giving back gains** — not "winners cut short" but "losers allowed to bleed back from positive to negative."

**Hypothesis for Exp 4:** Trailing stop can "rescue" the ~34 losers that had positive MFE by converting them to scratches or small wins.

---

### Experiment 4a — Trailing stop (version 1, failed)

**Setup:** 9-combo grid. activation_mfe ∈ {0.5%, 1%, 2%}, trail_distance ∈ {0.3%, 0.5%, 1%}. On top of Exp 3 winner.

**Result:** Winner activation=2%, trail=0.3% gives only +51.8% (essentially baseline), Sharpe 0.81 (+0.19), MaxDD -8.7%. Marginal.

**Lesson (from user pushback):** **0.5% MFE is noise-scale for BTC.** BTC's 15m ATR is 0.3-0.8% normally, 1-2% in volatile conditions. A 0.5% move is not meaningful — it's normal intrabar wiggle. The grid was boxed too tight; the winner was at the edge, which almost always means the real optimum is outside the grid.

---

### Experiment 4a-v2 — Trailing stop (rescaled to BTC volatility)

**Setup:** 12-combo grid. activation_mfe ∈ {2%, 3%, 5%, 10%}, trail_distance ∈ {0.5%, 1%, 2%}.

**Result:** **activation=5%, trail=0.5%** wins with **+88.1% return, Sharpe 0.93, MaxDD -10.1%, 119 trades.**

**Deltas over Exp 3 baseline:**
- Return: +50.8% → +88.1% (+37.3% — real jump)
- Sharpe: 0.62 → 0.93 (+0.31 — meaningful)
- MaxDD: -12.3% → -10.1% (better)

**Alternative point in the grid:** activation=10%, trail=0.5% gives **+108.5% return, Calmar 1.04** — highest absolute return but slight Sharpe cost (0.86).

**Lesson:** Trailing stops work beautifully when sized to the instrument's real volatility. The 5% activation only triggers on trades with real directional conviction, then locks in a tight 0.5% trail from peak. Trades that never reach 5% MFE follow normal signal-based exits. Trade management at this scale adds genuine alpha, not just risk reduction.

---

### Experiment 5 — Trade frequency relaxation

**Question:** Can we get "day trader" frequency (hundreds of trades/year) by relaxing the entry threshold?

**Setup:** 18-combo grid. K_entry ∈ {2, 3}, N_entry ∈ {1, 2, 3}, K_exit ∈ {0, -1, -2}.

**Result:** **K_entry=3, N_entry=2, K_exit=0 STILL wins.** Every K_entry=2 combo underperforms — at 1,000-3,000 trades per combo, commission drag dominates whatever extra edge the relaxed entry captured. Best K_entry=2 combo produced 1,667 trades and +39.4% (worse than the 117-trade Exp 3 winner).

**Lesson:** With 5 equal-weight signals and 0.08% round-trip fees, **more trades is not profitable** — the math doesn't work unless per-trade edge increases. Exp 5 appeared to close the door on the "more frequency" hypothesis. Exp 6 would reopen it.

---

### Experiment 6 — Strategy subset search 🎯

**Question:** Are all 5 strategies actually helping? What if some are dragging the ensemble down?

**Setup:** Brute-force 31 non-empty subsets of the 5 strategies. Each subset uses `Ke = majority(|S|)`, Ne=2, Kx=0, trailing stop act=5%, trail=0.5%. Ke scaling: |S|=1→1, |S|=2→2, |S|=3→2, |S|=4→3, |S|=5→3.

**Results (top by Sharpe):**

```
sz   subset                                         trades    return  Sharpe   MaxDD
 2   trend_following + funding_volume                1,206  +1048.3%   1.19  -46.8%
 4   drop mean_reversion                               992   +742.5%   1.18  -29.2%
 2   vol_breakout + funding_volume                  1,506   +640.4%   1.06  -31.5%
 3   momentum + trend + funding_volume              1,284   +526.0%   0.94  -49.7%
 5   ALL 5 (Exp 4a-v2 baseline)                        119    +88.1%   0.93  -10.1%
     BUY & HOLD                                          —   +713.3%   0.82  -77.2%
```

**Leave-one-out:**

```
Full 5:                               +88.1%   Sharpe 0.93
Remove mean_reversion:                +742.5%  Sharpe 1.18   Δ +654%, +0.25 Sharpe
Remove momentum:                       +75.5%  Sharpe 0.83   Δ  -13%, -0.10 Sharpe
```

**The diagnostic is unambiguous: mean_reversion is a net drag.** Removing it:
- **8.4x increases return** (+88% → +742%)
- Improves Sharpe by 0.25
- Generates 8x more tradable opportunities (119 → 992 trades)
- First time beating buy-and-hold on both dimensions

**Why it matters (vindication of Exp 1 correlation matrix):** mean_reversion was -0.52 correlated with vol_breakout and -0.51 with funding_volume. It was systematically voting AGAINST trades the trend-followers wanted to take, producing whipsaws that the asymmetric gate could filter but not eliminate. Exp 6 is effectively the first time we acted on the correlation matrix finding from Exp 1 — 5 experiments later.

**Strategy ranking (inferred from subset frequency in top 4):**
- **funding_volume** — MVP, appears in all top 4
- **trend_following** — essential, in 3 of top 4
- **volatility_breakout** — solid contributor, in 3 of top 4
- **momentum** — helpful but not essential
- **mean_reversion** — NET DRAG on the ensemble

**Also answers Exp 5's question:** Exp 5 concluded "more trades isn't possible" with all 5 strategies. Exp 6 shows the issue wasn't trade count — it was that mean_reversion was generating most of the bad trades. The 4-strategy subset produces 992 trades (8x more than Exp 3's winner) with better metrics everywhere.

---

## Candidate policies for production

Two finalists, selection depends on risk tolerance:

### Candidate A — Best risk-adjusted (4-strategy)

**Subset:** momentum + trend_following + volatility_breakout + funding_volume (drop mean_reversion)

**Policy parameters:**
- Asymmetric gate: K_entry=3 (majority of 4), N_entry=2, K_exit=0
- Trailing stop: activation_mfe=5%, trail_distance=0.5%
- Position: 100% equity, 1x leverage
- Costs: 0.04% commission per side, no funding/slippage modeled yet

**Metrics:**
- Return: **+742.5%**
- Sharpe: **1.18**
- MaxDD: **-29.2%**
- Trades: 992 (~150/year, or 3/week — real day-trader tempo)
- Alpha vs BH: +29% on return, +0.36 on Sharpe, +48% on MaxDD

### Candidate B — Best absolute return (2-strategy)

**Subset:** trend_following + funding_volume only

**Policy parameters:**
- Asymmetric gate: K_entry=2 (unanimous of 2), N_entry=2, K_exit=0
- Trailing stop: activation_mfe=5%, trail_distance=0.5%
- Position: 100% equity, 1x leverage

**Metrics:**
- Return: **+1,048.3%**
- Sharpe: **1.19**
- MaxDD: **-46.8%**
- Trades: 1,206 (~180/year)
- Alpha vs BH: +335% on return, +0.37 on Sharpe, +30% on MaxDD

### Trade-off

| Criterion | 4-strat | 2-strat |
|---|---|---|
| Absolute return | +742.5% | **+1,048.3%** |
| Sharpe ratio | 1.18 | **1.19** |
| MaxDD | **-29.2%** | -46.8% |
| Trade frequency | 992 | 1,206 |
| Simplicity | 4 moving parts | **2 moving parts** |
| Psychological tradability | **Easier** | Harder (-47% hurts) |
| Return ceiling | Lower | **Higher** |

**Both are shippable candidates.** 4-strat is safer, 2-strat has higher ceiling. A production system could run either — or both in a split-capital setup.

---

## Critical caveats

1. **In-sample selection bias.** Exp 6's winner was chosen by scanning 31 subsets on the full history. With 31 options, some will look good by chance even if none have real edge. **Walk-forward validation is mandatory** before trusting these numbers with live capital.

2. **No funding cost modeling.** Binance perpetual funding runs +0.01% to +0.05% per 8h in bull markets (longs pay). Our backtests assume zero funding. At 1x leverage holding longs, this is probably a 10-30% annualized drag not captured in our +742% / +1048% numbers. **Real returns will be meaningfully lower.**

3. **No slippage modeling.** Market orders at exact next-bar open. Real fills on BTC/USDT perp are usually 1-2 bps off. Small effect on a per-trade basis, accumulates over 992-1206 trades.

4. **No liquidation modeling.** At 1x it's impossible, so it's irrelevant to these specific runs. Will matter if leverage is added.

5. **Commission assumption is taker (0.04%).** Maker rebates (0.02%) would significantly improve economics if limit-order entries are feasible.

6. **Single timeframe (15m).** All signals generated on 15m bars. No higher-timeframe trend filter, no lower-timeframe execution refinement.

7. **Survivorship: BTC only.** The entire experiment is on a single asset that went up 8x during the period. Results are not transferable to other crypto or to different BTC regimes without re-testing.

---

## Next steps (in priority order)

### 1. Walk-forward validation of Exp 6 winner (MANDATORY before live)

Split data 2019-2022 (training) vs 2023-2026 (test). Run the 31-subset search ONLY on training, pick the winner, apply it to the untouched test period. If training winner also wins on test with Sharpe within ~30% of training, the edge is real. If it collapses, the subset was overfit.

**Also run leave-one-out separately on both periods.** If mean_reversion is the worst in BOTH, the finding is structural. If only in training, overfitting.

### 2. Add funding cost modeling

Extend the simulator to apply funding every 32 bars (8h) on open positions using the `funding_rates` table already in the DB. Rerun the walk-forward winner with funding included. Expect returns to drop meaningfully (likely 15-30%).

### 3. Add leverage as a separate experiment

Once a walk-forward-validated policy with funding exists, sweep leverage ∈ {1x, 2x, 3x, 5x} on the winning subset. Must include liquidation modeling. Find the leverage that maximizes risk-adjusted return subject to an acceptable DD cap (e.g., -50% max).

### 4. Refactor to shared backtester class

Extract the simulator into `src/aether_btc_lab/backtest/ensemble.py` so Exp 1-6 and future experiments share one code path. Add funding, leverage, slippage as optional features. Verify numerical parity with existing results before adopting.

### 5. Explore confidence weighting

Current ensemble treats each signal as binary (-1/0/+1). Each signal also has a confidence score (`sig_*_conf`) which we've never used. Could be a source of additional edge. Single-variable experiment after walk-forward is done.

### 6. Explore timeframe aggregation

Test whether running the gate on 1h or 4h bars (with 15m execution) changes the results. Higher-timeframe gates are structurally less noisy, which might improve trade quality.

---

## Files produced

- `scripts/backfill_indicators.py` — one-shot backfill of indicators+signals across full candle history
- `scripts/run_exp1.py` — Exp 1 baseline ensemble
- `scripts/run_exp2.py` — Exp 2 symmetric gate grid
- `scripts/run_exp3.py` — Exp 3 asymmetric gate grid
- `scripts/analyze_exp3_trades.py` — Exp 3 winner trade analysis
- `scripts/run_exp4a.py` — Exp 4a v1 trailing stop (wrong grid)
- `scripts/run_exp4a_v2.py` — Exp 4a v2 trailing stop (BTC-scale)
- `scripts/run_exp5.py` — Exp 5 frequency relaxation
- `scripts/run_exp6.py` — Exp 6 subset search (breakthrough)

## Database tables

- `exp1_trades`, `exp1_summary`, `exp1_equity_curve` — Exp 1 results
- `exp2_grid` — Exp 2 grid results
- `exp3_grid`, `exp3_trades` — Exp 3 grid + winner trade list
- `exp4a_grid`, `exp4a_v2_grid` — Trailing stop grids
- `exp5_grid` — Exp 5 frequency grid
- `exp6_subset` — Exp 6 subset search results

All tables in `aether_btc_lab` database on VPS (217.65.146.184:5432).

---

**Document last updated:** 2026-04-05
