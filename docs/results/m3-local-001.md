# m3-local-001: market residual (method 3), a local CPU look (owner 2026-11-19; exploratory, rule C5)

NOT the owner's artifact: public Binance futures klines (`data_io/vision_slice.py`), the 60 default coins, 4h,
2022-01 .. 2025-12, stride 2, 252,562 rows. Residual artifact by `data_io/market_residual.py` (commit d4009cc):
equal-weight market of the 60 coins, beta per coin from its last 180 bars (past only), 10 input channels, close label
= own label - beta_t x market move over the label span. The residual label keeps 34 % of the close label's variance
(0.00087 -> 0.00029): beta x market is two thirds of an 8 h move. Pre-registration
`docs/audit/2026-11-19-arch-indicators/PREREGISTERED_M3.md` (before any number); script `m3_local.py` (commit before
the runs). Runs: `jobs.rolling` on this sandbox's CPU, single-coin GRU 64 x 2, lr 3e-4, 5 epochs, patience 2, 60k
train rows, 4 expanding folds (TEST 2024-01 .. 2025-12), seeds 0 and 1; arm A = raw inputs / own label, arm M =
residual inputs / residual label. Outputs stay in the session scratchpad (not on Drive).

## Verdict

- **Feasible by the pre-registered rule: YES** (both conditions on both seeds).
  1. Close rank IC by timestamp against the residual label: M 0.076 / 0.077, positive in all 4 folds on both seeds
     (+0.066 .. +0.087); A 0.002 / 0.000 (folds between -0.012 and +0.017).
  2. Top decile (VAL 90th percentile of mu), residual gross per trade: M +0.046 % / +0.033 % against A +0.016 % /
     +0.027 %; M's day mean CI is above 0 on both seeds (+0.062 % [+0.032, +0.091], +0.046 % [+0.007, +0.084]).
- **Pays: NO.** +0.03 to +0.05 % gross per trade is far under one round trip (0.16 %).
- Model-free bound (ridge, same rows): the SAME raw features fitted on the OWN label rank coins backwards on test
  (IC -0.030, negative in all 4 folds); fitted on the RESIDUAL label they rank right (IC +0.062 against the residual,
  +0.064 against the own label's ranks, positive in all 4 folds). The residual feature set fitted on the residual
  label gives +0.051.

- **M3a (longer horizons, model-free, amendment 1): cannot pay either.** Ridge on the residual features and label,
  per-timestamp top 10 %, residual gross per trade (mean of the 4 folds): 8 h +0.039 %, 24 h +0.063 %, 48 h
  +0.079 % (48 h: above 0.16 % in 2 of 4 folds only). Rank IC stays positive (24 h +0.041 .. +0.056 in every fold).
  And the BOTTOM decile earns more in mean simple return than the top at every horizon (24 h +0.154 %, 48 h +0.270 %):
  the ranking is right in log / median terms, the mean P&L is carried by the right tail of volatile coins.

## What the model learned

M learned to rank coins inside a timestamp (IC 0.077 in every fold), and the ridge shows the label is what matters:
the own label teaches the model the market's move, which reverses the cross-sectional order.

## Where it is right and where it fails

Residual return per trade by decile of M's mu within the timestamp (seed 0; seed 1 has the same shape):
+0.046, -0.010, -0.008, -0.021, +0.002, +0.011, +0.000, +0.063, +0.032, +0.036 %, while M's predicted sigma falls
from 2.36 % (decile 1) to 0.99 % (decile 10). The middle deciles rise; the lowest decile (the most volatile coins)
has a POSITIVE mean simple return. Per-timestamp top 10 % (information, not pre-registered): M +0.036 / +0.048 %
per trade, A +0.044 / +0.057 %; top minus bottom: M -0.012 / -0.032 % per day (CI contains 0).

## The samples it failed on

The lowest-mu decile: high-volatility coins whose median residual move is negative (so they rank low) but whose
mean simple return is pulled up by a few large up moves. Row-level tables are in the scratchpad, not listed here.

## Bias

M's mu is tied to its predicted sigma (Spearman across deciles about -1): it ranks partly by volatility, low
volatility = high mu.

## Selective use

One pre-registered selection (global VAL threshold). The per-timestamp top 10 % and the decile curve were read after
the numbers (information only, two extra variants).

## Why it failed

**Signal smaller than the cost, and part of it is a volatility ranking:** the residual ranking is real (model and
ridge agree, every fold), but it lives in the median of log moves; the mean simple return the top decile earns over
8 h is about a quarter of the cost, and the most volatile coins' right tail undoes the bottom of the ranking.

## Next hypotheses

| # | Hypothesis | Refuted if | Cheapest test |
|---|---|---|---|
| M3a | (tested, amendment 1: REFUTED) residual drift adds up to more than 0.16 % at 24-48 h | top decile +0.063 % / +0.079 % | done |
| M3e | The P&L lives in the right tail: a target that is the MEAN simple residual return (not the log move) or a quantile head on the upper tail ranks the coins that pay | the per-timestamp top decile's mean residual return is not above the bottom decile's | local run, residual label as exp(r) - 1 |
| M3b | A volatility-scaled residual target (residual / predicted sd) stops mu from ranking by volatility and makes the deciles monotone | the top-minus-bottom spread stays at or below 0 | the same local run with a scaled label |
| M3c | Use the residual model as a FILTER for the long-only book: do not buy coins in M's lowest 3 deciles | the filtered long book's mean day without the best 5 % is not above the unfiltered one | the full artifact (GPU) or this local slice with arm A's long book |
| M3d | The full artifact (535 coins) gives the same IC gain on its own folds | IC against the residual label below 0.03 on the 7 folds | `market_residual` on Colab (`--cache`), 2 seeds, 7 folds |
