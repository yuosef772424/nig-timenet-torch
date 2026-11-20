# m3-full-001: method 3 (market residual) on the full artifact, GRU, 2 seeds (owner 2026-11-19/20; exploratory, C5)

Job `docs/jobs/rolling-mres-001.json`, Colab L4, commit aa738f3 (`colab_chain.sh`, checkout `/content/nig_t`), after
its smoke (rule E5); the runtime was recycled once and both seeds resumed from their Drive checkpoints on the same
commit. Data `/content/art_mres` = `data_io/market_residual.py --cache` on art1 (535 coins; the market = their
equal-weight close log move per bar; beta per coin from its last 180 bars; 2,520,913 rows, 2,519,933 with channels,
2,520,077 with a residual label; label variance own 0.00152 -> residual 0.00092). Arm M3 = arm R of
`rolling-ind-001` with only the artifact changed (GRU 64 x 2, lr 3e-4, 8 epochs, patience 3, 400k rows, top 100
coins, 7 expanding folds, TEST 2023-01 .. 2026-06, seeds 0 and 1). Reference A = arm R (`rolling-ind-001/out_R_s{0,1}`;
the model / trainer / evaluation code is the same as on aa738f3, only `data_io` tools changed in between).
Pre-registration `docs/audit/2026-11-19-arch-indicators/PREREGISTERED_M3_FULL.md` (+ amendment 1, the V1 sceptic
reading). Readings on Drive `nig_timenet_runs/analysis-ind-001/m3_full*`; in M3's own tables `y_ret` is the residual
return, in A's the coin's own return.

## Verdict

| pre-registered rule (both seeds) | result | |
|---|---|---|
| Feasible: M's IC against the residual label above A's, and M's top-decile residual gross above A's | **holds** | IC 0.093 / 0.094 vs 0.015 / 0.012; gross +0.051 / +0.049 % vs +0.022 / +0.018 % per trade |
| Pays: M's top-decile residual gross per trade above 0.16 % | **fails** | +0.05 % |
| Market back better: V1 or V2 above V0 on the mean day AND without the best 5 % of days | **holds (V1)** | V1 +0.289 / +0.272 % vs V0 +0.173 / +0.215 %; without best 5 %: +0.026 / +0.035 % vs -0.072 / -0.022 % |

V1 (the residual forecast plus beta x the market forecast, the market forecast = the mean of A's mu at the
timestamp) is the first long-only book of this study that stays above 0 without its best 5 % of days on both seeds,
and is positive in every fold and every year. The sceptic reading (amendment 1) shows what it is: **market timing
with a small coin-choice edge**, not a coin edge. It is a hypothesis for live use, not a result, until it repeats on
other seeds.

| V1 (seed 0 / 1) | value |
|---|---|
| own net, mean day [CI 95 %] | +0.289 % [+0.124, +0.452] / +0.272 % [+0.142, +0.404] |
| without the best 5 % of days | +0.026 % / +0.035 % |
| per fold (day mean, folds 0-6) | +0.37 +0.33 +0.11 +0.69 +0.21 +0.15 +0.16 / +0.24 +0.39 +0.24 +0.63 +0.13 +0.13 +0.14 % |
| per year 2023 / 2024 / 2025 / 2026 (to 06) | +0.35 / +0.38 / +0.17 / +0.17 % (s0); +0.30 / +0.41 / +0.12 / +0.16 % (s1) |
| minus the equal-weight market (1 cost) | -0.112 % [-0.144, -0.079] / -0.123 % [-0.159, -0.086] |
| minus BTC (2 costs) | -0.084 % [-0.191, +0.026] / -0.107 % [-0.196, -0.018] |
| beta to BTC | 1.38 / 1.37 |
| days with a trade | 756 / 803 of 1225; median 62 trades on a traded day |
| timing only (every row of the timestamps V1 trades, 1 cost) | +0.173 % / +0.126 % (without best 5 %: -0.060 / -0.084 %) |

## What the model learned

- **The residual is predictable where the own return is not.** Close rank IC by timestamp against the residual
  label: M 0.123 0.074 0.093 0.087 0.085 0.087 0.103 (seed 0), 0.121 0.071 0.098 0.089 0.086 0.085 0.106 (seed 1);
  A, the same rows, 0.015 / 0.012 (one fold below 0). Six times the IC of the own-return model, in every fold
  (the local look on 60 coins said the same: 0.077).
- The top decile's residual gross is small but robust: day mean +0.067 % [+0.052, +0.082] / +0.072 % [+0.055,
  +0.088], +0.031 / +0.030 % without the best 5 % of days.
- It buys other coins: only 7 % / 5 % of M3's long trades are A's; half of them are calm coins (34k of 68k).
- It trains normally: best epoch 4-7 of 7-8 (early stopping).

## Where it is right and where it fails

M3's own long book (residual return, one cost; "minus market" = minus the mean residual of the timestamp):

| group | net s0 / s1 | minus market s0 / s1 | baseline (always long, same rows) |
|---|---|---|---|
| sigma quintile 1 in book (narrowest) | -0.16 / -0.16 % (hit 0.23 / 0.26) | -0.20 / -0.18 % | (market) |
| sigma quintile 5 in book (widest) | -0.05 / -0.06 % (hit 0.48 / 0.47) | -0.11 / -0.12 % | (market) |
| mu quintile 1 / 5 in book | -0.09 / -0.11 % (s0) | -0.12 / -0.15 % | (market) |
| calm / mid / wild coins | -0.13 / -0.09 / -0.10 % (s0) | -0.13 .. -0.16 % | -0.13 / -0.12 / -0.11 % |
| market up / down (realised) | +0.04 / -0.30 % | -0.20 / -0.09 % | +0.11 / -0.39 % |
| 2023 .. 2026 | -0.13 / -0.12 / -0.08 / -0.13 % (s0) | -0.15 .. -0.16 % | -0.13 / -0.12 / -0.11 / -0.13 % |

The residual edge is about the same size in every group (no group carries it); inside the book a wider sigma is
better (the residual moves of calm coins are too small to pay), and a stronger mu does NOT help (mu quintile 5 is not
above quintile 1): the mean forecast's ordering at the top is weak, the IC comes from the whole distribution.

## The samples it failed on

The confident residual losses are market-wide days the residual does not remove: 2023-06-09 16:00 .. 06-10 04:00
(EOS -13 %, SHIB -9 %, SUSHI -9 %, CTK -8 %, DASH -7 %, ADA -7 %, MATIC -6 %, AVAX -6 %, SOL -6 %, TRX -7 %: the
alt-market sell-off of that weekend; the mean residual of those timestamps is -0.9 % to -1.5 %, which suggests,
not verified, that the 180-bar beta under-stated the coins' falls), DYDX
2024-04-12 -10 % (twice), ALPHA 2023-03-04 -9 % (twice), ZEC 2024-02-06 -7 %, FTM 2024-06-17 -7 %. The worst days of
the book are small (-1.5 % on 2026-02-04, -1.1 % on 2026-06-02): the residual book has no crash days, its losses are
the cost.

## Bias

Extreme down-lean: the share of rows with mu > 0 is 0.001-0.27 per fold against a realised share of positive
residuals of 0.44-0.48; mean mu -0.09 .. -0.16 % against a realised mean residual +0.02 .. +0.06 %. The residual is
right-skewed (median below 0, mean above 0) and the NIG mean follows the bulk, not the mean (idea M3e). The book uses
a VAL quantile, so the level does not change which rows are bought; it matters for any rule that uses mu's sign or
size (V1 adds mu to beta x the market forecast, so the down-lean shifts V1's score by a constant per fold, absorbed
by its VAL threshold).

## Selective use

On M3's own residual book, the four VAL-fitted rules of `read_detail.py` (4 tried) all lose about one cost: top 5 %
of mu -0.12 %, top 10 % with p_dir > 0.5 -0.09 %, mu / sigma -0.11 %, low sigma -0.11 % per trade (minus market
-0.14 .. -0.15 %). The two pre-registered "market back" variants: V1 (above) passes, V2 (only timestamps whose market
forecast is in its VAL top 20 %, then M's top decile) does not (+0.106 / +0.124 % per day, negative without the
best 5 %). Variants tried on these tables: 4 rules + V0 / V1 / V2 + the V1 sceptic views (multiple testing: V1 is
one of 7 books read).

## Why it failed

**Signal smaller than the cost** (for the residual alone): method 3 makes the coin's part of the move readable (IC
0.09 vs 0.015, every fold) but the top decile's residual is +0.05 % per 8 h against a 0.16 % round trip. V1 pays
only by adding back market timing (beta 1.38), so its after-cost profit is directional, not a coin edge.

## Next hypotheses

| # | Hypothesis | Refuted if | Cheapest test |
|---|---|---|---|
| V1-rep | V1 is not a seed accident: it stays above 0 without the best 5 % of days | on seeds 2 and 3 of R and M3 (same commit), V1's day mean without the best 5 % of days is at or below 0 on either seed | 4 GPU runs (R and M3 x seeds 2, 3), about 1 h with split folds; then `m3_readd.py` + `m3_v1_read.py` |
| M3-H | Holding V1's coins while their score stays in the top 25-30 % cuts the turnover and raises the net per round trip | V1's net per round trip with the hysteresis exit is not above V1's on both seeds | `hold_exit.py` on a V1 score table (no training) |
| M3e | A target that follows the right tail (the mean simple residual, or an upper-quantile head) puts the top of the ranking where the residual P&L is (mu quintile 5 is not above quintile 1 today) | the top decile's residual gross per trade is not above M3's 0.05 % on both seeds | a head / target switch on the residual artifact: E1 first, then the same walk-forward |
| M3-B | A better beta (shorter window, or the market's downside beta) removes the crash days (2023-06-10) from the residual losses | the residual book's worst 1 % of rows do not shrink on VAL | rebuild `art_mres` with another beta window (minutes), re-read the tables of a short look |
