# dy1-daily-001: a daily model that predicts two days ahead, read as money after costs (DY1 = N3; exploratory, C5)

Owner task 2026-11-22: a model on the daily frame, label two days ahead, that makes money after fees "even a small
amount (a cup of coffee)"; try the panel and the single-coin model; then (same day) every normalisation (default,
std, market removed) and each loss on its own (absolute per-row loss, rank loss), for each model kind, two seeds each,
on coins of similar volatility. Pre-registration and its four amendments, all written before an arm table was read:
`docs/audit/2026-11-22-dy1-daily/PREREGISTERED.md`. Scripts and every output: that folder (`OUT_*.txt`). Runs: Drive
`nig_timenet_runs/dy1-vm-<arm>-001/out` (Colab L4; commits 594b0c0 for s / p / pa / sr, 522bd30 for sk / pk; the code
path of the earlier arms is unchanged by 522bd30, D-051 default 1.0), `dy1-single-001` / `dy1-panel-001` on the
session's CPU (be70dc2).

Setup, for every number below. Data: public Binance USDT-perpetual daily klines (`build_daily.py`, not the owner's TF
artifact): 32 daily bars of log high / low / close moves, labels 2 days ahead, one row per coin per day, 2020-01 ..
2026-09 (time seal). Universe VM (`vol_universe.py`): 82 coins whose daily sd before 2023-01-01 lies in its 25-95 %
band (0.063-0.094; BTC, ETH, BNB, TRX, LTC ... dropped). Walk-forward: 7 expanding folds (first train 1095 days, test
180, embargo 2), TEST 2023-01 .. 2026-06, seeds 0 and 1. Money: `eval/account.py` (N8): long book = rows with mu >= the
fold's VAL q90, fixed slots of 1 / 77 coins, cash for the rest (about 10 % of the capital invested on average), fee
0.08 % per side on turnover, 2 sub-books, compounded; beside always long (-16.9 % a year on VM), the random-coin
timing placebo (20 draws, same days and counts) and the basket x breadth. 20 candidate arms: the CI level is
1 - 0.05 / 20. Caveat for every absolute number: the split drops the first window + embargo of each TEST window (test
rows need feature_start > cut + embargo, `data_io/split_plan.py::purge_embargo_split`), 34 of 180 days per fold at the
daily frame, so every book and benchmark sees the same 1,022 days and holds cash on the rest.

## Verdict
- Two arms meet the pre-registered accept rule on BOTH seeds (CAGR > 0, > always long, > placebo p95): **pk-def**
  (panel, rank loss alone; pk-std is the same model, see below) and **pk-mres** read on the coins' own returns (panel,
  rank loss alone, market removed). They make **+4.1 / +1.2 %** and **+2.7 / +4.5 %** a year after fees at ~10 %
  exposure (Sharpe 0.63 / 0.24 and 0.71 / 1.26, max drawdown -13 / -18 % and -9 / -7 %), against always long -16.9 %.
  That is the owner's "cup of coffee", and no more: the day-block CI of the account's mean period return contains 0
  for every arm (pk-def [-1.84, +4.57] / [-2.48, +3.25] bp, pk-mres own [-1.04, +2.73] / [-0.47, +3.46] bp at 97.5 %).
  Exploratory; a fresh walk-forward (other seeds, other universe) must repeat it before anyone trades it.
- The arms that earn the most (s-std +16.0 / +9.7 %, pa-def +15.5 / +7.9 %, p-def +12.4 / +11.2 %, Sharpe up to 1.55)
  fail the rule: random coins on the same days earn the same. Their money is market timing (the book's breadth), not
  coin choice.
- Model-free first (N3): a ridge on the 164-coin daily rows already has rank IC +0.042 (t ~ 3.8) and a long book of
  +2.3 % a year after cost (`OUT_model_free.txt`); the cost wall of the 4 h frame (B-074) does not hold at a 2-day
  label: the turnover-based fee costs the books 1.4-1.8 points a year (fee 0.02 % against 0.08 % per side).

| arm (seeds 0 / 1) | model, loss, normalisation | rank IC | CAGR | Sharpe | maxDD | placebo p95 | rule |
|---|---|---|---|---|---|---|---|
| s-def | single, absolute, default | +0.006 / +0.013 | +11.3 / +6.9 % | 1.16 / 0.77 | -8 / -12 % | +10.7 / +7.5 % | yes / no |
| s-std | single, absolute, std | +0.012 / +0.009 | +16.0 / +9.7 % | 1.55 / 1.00 | -6 / -9 % | +15.5 / +10.5 % | yes / no |
| s-mres own | single, absolute, market removed | +0.086 / +0.086 | -2.5 / -1.5 % | -0.42 / -0.21 | -15 / -11 % | -5.2 / -3.7 % | no / no |
| sk-def = sk-std | single, rank alone | +0.079 / +0.073 | +3.8 / +3.7 % | 0.44 / 0.45 | -12 / -16 % | +2.8 / +3.8 % | yes / no |
| sk-mres own | single, rank alone, market removed | +0.087 / +0.078 | -0.9 / -0.9 % | -0.17 / -0.18 | -16 / -15 % | -4.5 / -3.2 % | no / no |
| sr-def | single, absolute + rank, default | +0.078 / +0.067 | +11.3 / +10.8 % | 1.04 / 1.08 | -11 / -10 % | +11.2 / +11.6 % | yes / no |
| sr-std | single, absolute + rank, std | +0.064 / +0.056 | +2.6 / +6.7 % | 0.29 / 0.70 | -14 / -11 % | +2.8 / +7.5 % | no / no |
| sr-mres own | single, absolute + rank, market removed | +0.075 / +0.084 | -2.4 / -2.4 % | -0.51 / -0.44 | -11 / -12 % | -4.0 / -4.7 % | no / no |
| pa-def | panel, absolute, default | +0.054 / +0.054 | +15.5 / +7.9 % | 1.28 / 0.73 | -8 / -10 % | +15.3 / +8.1 % | yes / no |
| pa-std | panel, absolute, std | +0.060 / +0.052 | +13.0 / +7.9 % | 1.46 / 0.71 | -7 / -13 % | +13.0 / +8.2 % | yes / no |
| pa-mres own | panel, absolute, market removed | +0.082 / +0.082 | -0.8 / -5.4 % | -0.11 / -0.89 | -13 / -19 % | -1.4 / -5.0 % | no / no |
| **pk-def = pk-std** | panel, rank alone | +0.087 / +0.086 | +4.1 / +1.2 % | 0.63 / 0.24 | -13 / -18 % | +3.1 / +1.0 % | **yes / yes** |
| **pk-mres own** | panel, rank alone, market removed | +0.078 / +0.082 | +2.7 / +4.5 % | 0.71 / 1.26 | -9 / -7 % | -0.1 / +2.5 % | **yes / yes** |
| p-def | panel, absolute + rank, default | +0.077 / +0.083 | +12.4 / +11.2 % | 1.30 / 1.15 | -10 / -8 % | +13.1 / +10.6 % | no / yes |
| p-std | panel, absolute + rank, std | +0.082 / +0.085 | +10.2 / +9.3 % | 1.23 / 1.09 | -8 / -10 % | +11.7 / +10.5 % | no / no |
| p-mres own | panel, absolute + rank, market removed | +0.077 / +0.076 | -2.1 / -3.2 % | -0.34 / -0.58 | -11 / -13 % | -1.5 / -3.4 % | no / no |
| ref S (164 coins) | single, absolute, default | +0.015 / +0.021 | +5.9 / +4.1 % | 0.70 / 0.53 | -12 / -17 % | +6.7 / +4.7 % | no / no |
| ref P (164 coins) | panel, absolute + rank, std | +0.080 / +0.075 | +9.2 / +2.8 % | 1.04 / 0.39 | -8 / -10 % | +9.8 / +4.0 % | no / no |

"own" = the M3 picks read on the coins' own returns (amendment 1; on their residual returns every M3 book is within
+-1.2 %). The rank-alone arms with the default and the std label normalisation give the SAME numbers to the last digit:
with `base_loss_weight` 0 the label scale reaches no loss (the rank term is scale-free), a check that the switch does
what it says. Long-short books, the fee grid, the CIs: `OUT_read_vm18.txt`, `OUT_read_ref164.txt`.

## What the model learned
- Coin ranking: every arm with a rank term, every panel arm and every M3 arm ranks the coins of a day above zero on
  both seeds (rank IC +0.05 .. +0.09, t 6-10 with the 2-day overlap); the single-coin model with the absolute loss
  alone does not (s-def / s-std +0.006 .. +0.013, t < 1.6). The rank term adds about +0.07 of IC to the single-coin
  model (s -> sr / sk) and about +0.03 to the panel (pa -> p / pk).
- Market timing: the level of mu of the absolute-loss arms moves with the coming market move, so the share of rows
  above the VAL q90 (the book's breadth) is a timing signal: basket x breadth +13.0 % a year for p-def seed 0, Sharpe
  1.34, while trivial basket timing rules on the same days (past 1-14 day move above / below its pre-2023 tercile)
  make -0.9 .. +1.4 % at Sharpe <= 0.82 (`OUT_timing_baselines.txt`, a look, not pre-registered).
- Volatility: sigma ranks the size of the next move (IC with |y| +0.35 .. +0.37 for the absolute-loss arms, at the
  level of the trivial ATR baseline +0.36); the rank-alone arms have no trained sigma (IC +0.02).
- Model-free: ridge IC +0.042 on 164 coins; the panel models roughly double it.

## Where it is right and where it fails
From `OUT_failure_read.txt` (long book picks, excess over the day's equal-weight basket, per 2-day trade, CI 97.5 %):
- pk-def seed 0: all picks +0.123 % [+0.017, +0.233], the only arm whose per-trade excess CI excludes 0; best when the
  basket rose over the past week (+0.294 % [+0.088, +0.504]) and in 2023 (+0.226 %); flat after a flat week (-0.076 %)
  and in 2025 (+0.005 %).
- p-def seed 0: +0.019 % [-0.123, +0.155] per pick: the IC is real but the top decile does not beat the basket; the
  book's raw return per pick is +1.6 % after a down week against -0.7 % after a flat one (the timing); 2026 loses
  (-0.364 %).
- s-std seed 0: +0.098 % [-0.045, +0.223]; 2024 carries it (+0.315 % [+0.112, +0.510]), the other years are within
  +-0.1 %.
- pk-mres own seed 1: +0.100 % [-0.104, +0.309]; 2025 +0.477 %, 2024 -0.210 %.
- Every arm: hit rate of the picks over the basket 0.45-0.48 (fewer than half the picks beat the basket; the mean is
  carried by the right tail). By the coin's volatility tercile of the day (ATR) only the rank-alone arms have groups
  whose CI excludes 0, and on seed 0 only: pk-def in its LOW-volatility third (+0.199 % [+0.043, +0.338]) and its
  middle sigma third, sk-def in its HIGH-volatility third (+0.285 % [+0.042, +0.547]); the timing arms have none.

## The samples it failed on
The worst long picks of pk-def seed 0 are market crash days on high-beta alts the model ranked at the top:
2024-04-12/13 (SUSHI -34 %, DYDX -34 %, CRV -30 %), 2025-10-10 (DENT -36 %, MASK -30 %, the liquidation day),
2024-12-17 (BLZ -54 %), 2025-04-02/03 (NKN -34 %, -30 %). p-def / pa-def fail on the same kind of day (2026-02-04/05
NKN -27 / -29 %, 2026-04-18 DENT -30 %, 2026-06-03/04 CHZ, ZEC, SUSHI -23 .. -26 %). The rank-alone mu of those rows
is in the top tail (+0.08 .. +0.11): the panel ranks a coin up after it fell less than the others or bounced, and a
market-wide crash two days later hits exactly those high-beta coins hardest.

## Bias
- The absolute-loss arms lean down on average (share mu > 0: 0.22 .. 0.51 by year against 0.38 .. 0.50 realised;
  mean mu -0.002 .. -0.006 per 2 days) and the lean follows the market (lowest in 2023 and 2026): that lean is the
  timing the book earns from.
- The rank-alone arms have no level (the term ignores a shift of the whole day): pk-def's mu is positive for 76-88 %
  of rows; only the within-day rank means something, so their breadth is close to constant and the placebo gap is
  their coin choice.
- A correction fitted on VAL was not tried (the books are long-only and cut at a quantile, so a shift moves only the
  breadth).

## Selective use
Rules fixed in `failure_read.py` before TEST was read, cut on each fold's VAL, run through the same account (fee 0.08
%): for the timing arms (p-def, s-std) mu q95 raises Sharpe (1.42 / 1.70) at half the exposure and mu / sigma q90
lowers it; none beats its own placebo by a margin. For pk-def mu / sigma q90 gives +5.3 % (Sharpe 0.84) against a
placebo p95 of +3.8 %, mu q80 +4.1 % against +2.8 %; for pk-mres own mu q80 +4.6 % against +0.9 % (seed 1). Five rules
x 16 groups were looked at: a rule picked here is a hypothesis for the next walk-forward, not a result.

## Why it failed
For the timing arms (s-*, pa-*, p-*, sr-*): the evaluation the owner asked for (beat random coins on the same days)
is met by timing alone; their coin choice is a signal smaller than its noise (per-pick excess +0.02 .. +0.10 % with a
CI across 0), because the absolute loss spends the model on the level of the move (the market) and the rank IC it
reaches (+0.05 .. +0.08) sits in the middle of the distribution, not in the top decile the book buys. For the
M3 arms: the market was removed from the label, so the book loses the timing and holds high-beta coins through a
falling alt market (-16.9 % a year), and their coin choice (+3 .. +4 points over the placebo) does not cover it.
For the two arms that pass: the signal is there and read, but it is small next to its cost of risk (CI across 0,
Sharpe <= 1.26, crash-day drawdowns): ONE cause, signal smaller than its variance at 10 % exposure on 1,022 days.

## Next hypotheses
1. H1 (N9, now unblocked): the timing is the larger edge. A market head or the breadth of an absolute-loss arm as the
   exposure of the basket (or BTC), fitted on VAL, on a fresh walk-forward (seeds 2, 3). Refuted if basket x breadth
   does not beat the trivial timing rules (Sharpe 0.82) and buy-and-hold on both seeds.
2. H2: combine the two edges: the rank-alone model picks the coins (pk), the absolute-loss model's breadth sets the
   exposure (p / pa). Cheapest test: on the saved tables, pk's picks x p-def's breadth, cut on VAL. Refuted if it
   does not beat both components' placebos on both seeds.
3. H3: the crash-day losses are beta: hedge the pk book with beta x basket short (or BTC short) or cap the beta of the
   picks. Refuted if the hedged book's CAGR and Sharpe do not rise on both seeds.
4. H4 (measurement): let TEST rows keep inputs that start before the fold cut (only labels must not overlap): 34 of
   180 days per fold come back, the CI narrows by ~10 %. Needs an owner decision on the split rule (rule D2).
