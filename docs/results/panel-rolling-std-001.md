# panel-rolling-std-001: the panel model with std target normalisation, walk-forward (owner task 2026-11-18)

Exploratory (rule C5). Pre-registration: `docs/audit/2026-11-18-panel-std/PREREGISTERED.md` (written before any
number). Jobs on commit `6c928bb`, data `c572b1b36960`, L4: `panel-rolling-std-001` (candidate: relative inputs +
rank term + `target_norm: logret_std`) and `panel-rolling-ref-002` (the same without `target_norm`), 7 expanding
folds x seeds 0 and 1, 1225 pooled TEST days per seed. Sources: `MyDrive/nig_timenet_runs/<job>/out/rolling_folds.csv`
and `pooled_days_seed_<s>/panel_report.md`, read through the Drive connector (the runtime was gone).

**Verdict: refuted on the close, a loss on high / low ranking; not profitable on all rows.** Per the pre-registered
rule the close rank IC is not above the reference in at least 4 of 7 folds on seed 0 (3 / 7; seed 1: 4 / 7; mean
difference -0.0001). The std arm is kept as a switch, off by default (it already is: D-033).

| per fold, mean over the 14 fold x seed runs | std (candidate) | reference | panel-rolling-001 (same arm, `ebbd336`) |
|---|---|---|---|
| close rank IC by timestamp | 0.0858 | 0.0860 | 0.0860 |
| high rank IC | **0.179** | 0.360 | 0.360 |
| low rank IC | **0.333** | 0.441 | n/r |
| high / low `share_std_below_move` (T-sharp share) | 0.95-0.99 | 0.02-0.53 | 0.04-0.46 |
| best epoch = last epoch (8 / 8) | 12 of 14 runs | 4 of 14 | |

The reference re-run on a new commit reproduces `panel-rolling-001` to four decimals (close 0.0860, high 0.3596):
the walk-forward is deterministic, so differences between arms here are the arms, not noise of the pipeline.

Pooled TEST days (seed 0 / seed 1), cost 0.0008 per side:

| | std | reference |
|---|---|---|
| metric 1 (sign book) mean day gross | +0.093 % / +0.061 % | +0.071 % / +0.070 % |
| metric 1 mean day net | -0.067 % / -0.099 % | -0.089 % / -0.090 % |
| metric 1 beats every baseline | no / no (beats yesterday_market_sign only) | no / no (+ last_bar_asymmetry) |
| metric 2 (top / bottom 20 %) mean day net | -0.116 % / -0.114 % | -0.118 % / -0.118 % |
| metric 2 beats every baseline | no / no (momentum only) | no / no |
| relative rank IC (pooled) | 0.0855 / 0.0862 | 0.0853 / 0.0867 |
| predicted up (realised 0.486) | 0.455 / 0.421 | 0.524 / 0.500 |
| top_confidence rows (~20 %, VAL thresholds): metric 1 mean day net | **+0.069 % / +0.047 %** | **+0.045 % / +0.050 %** |
| top_abs_mu rows (~20 %): metric 1 mean day net | +0.037 % / +0.003 % | +0.038 % / +0.031 % |

## What the model learned
The close head reads the same cross-sectional ranking with or without the label standardisation (rank IC 0.086 in
every fold, 0.06-0.11 per fold, every seed; relative rank IC 0.085-0.087 against momentum / reversal 0.048 and the
last-bar asymmetry 0.038): the rank term carries it and is invariant to the per-head affine map, as expected. The
sd of every head still ranks its own error (pooled correlation sd vs |error| 0.32 on the close, both arms). On E4
(planted 0.15) both arms read 0.125 of the close signal (known answer, `docs/handoff/2026-11-18-panel-std-rolling.md`).

## Where it is right and where it fails
From the pooled reports: the close sign book wins 56-58 % of days gross in both arms and loses after cost on all
rows; the rows the model is most confident on (top ~20 % by confidence, thresholds fitted on VAL) are the only
group with a positive net mean day, in all four runs. The high / low heads fail in every fold (IC down by 0.18 /
0.11 against the reference, no fold where std is better). The error map by sd / move / side / regime / fold groups
from the prediction tables is NOT yet computed: `read_std_rolling.py` (the first step of the chain) writes it to
`MyDrive/nig_timenet_runs/panel-rolling-std-001/reading_vs_panel-rolling-ref-002/reading.json`.

## The samples it failed on
Pending the same reading (`worst_rows_cand`, `worst_days_cand`: the confident wrong-sign rows and the worst neutral
days with the market's move). Earlier audit of the same model family: the large losses are squeezes of thin coins
from quiet bases (B-077, `docs/audit/2026-11-17-failure-charts/`).

## Bias
The std arm leans short (predicted up 0.455 / 0.421 against a realised 0.486); the reference leans slightly long
(0.524 / 0.500). Per-coin share predicted up 0.10-0.72 and uncorrelated with the realised share (-0.02 / +0.13) in
both. A VAL re-centring of the close mean is in the reading (`recenter_cand`), not yet run.

## Selective use
The panel report's four rules (fixed before this run, thresholds on VAL, applied to TEST): `top_confidence` is
positive after cost in all 4 runs (+0.045 % to +0.069 % a day on about 20 % of rows), `top_abs_mu` in 4 of 4 but
near 0 on one, `low_sd` negative everywhere. No CI is printed for these rows and 4 rules x 2 arms were looked at:
this is a hypothesis for the next walk-forward, not a result (multiple testing). The reading adds 16 more rules x
kept shares with a day-block CI on the kept rows.

## Why it failed
Named cause (hypothesis with evidence, not yet isolated): **the ordered-means coupling is applied in the normalised
space.** With `target_norm`, `mu_low <= mu_close <= mu_high` holds on z = (y - mean_h) / std_h
(`model/target_norm.py` PITFALLS; in code the order is applied at `model/builder.py:315`, the inverse map at
`:321`, and `enforce_order` defaults to True), where the three heads are centred on 0 and the order has no meaning: a confident
close forces the high / low means with it. Evidence: the loss is on high / low only (close unchanged), in every
fold and seed, and on E4 too (0.27 -> 0.23); the high / low "T-sharp pass" (share 0.95-0.99) is not sharpness: the
means sit near the train mean move (+-3 %), so |pred - close| is large by construction, while the sd ranks the
error worse (pooled 0.21 vs 0.38 on the high head) and the 80 % coverage is 0.91-0.92 (outside 75-85 %). Slower
convergence (12 of 14 runs at their last epoch) is a second, smaller effect.

## Next hypotheses
1. Ordering in label units: apply the ordered-means constraint AFTER the inverse map (or turn it off with
   `target_norm`). Refuted if the high / low rank IC stays more than 0.02 below the reference on E4 and on two folds.
   Cheapest test: E4 look, 2 seeds, minutes. **E4 done 2026-11-18 (`label-order-e4-look`, commit `ed2d3b4`, switch
   `order_in_label_units`): not refuted.** High / low rank IC seed 0 / 1: reference 0.263 / 0.253 and 0.269 / 0.255;
   std 0.232 / 0.222 and 0.228 / 0.211; std + order on prices 0.263 / 0.252 and 0.269 / 0.256 (close 0.127 / 0.128 in
   all). The walk-forward on real data is `panel-rolling-std-order-001`.
2. Standardise the close head only (`target_norm` per head). Refuted if the close bias / coverage do not move while
   high / low stay at the reference. Cheapest test: E4 then two folds.
3. `top_confidence` is a real edge after cost: pre-register it (20 % share, VAL thresholds) and read it with a
   day-block CI on the pooled days of `panel-rolling-ref-003` / `-2tf-001` (the next walk-forwards). Refuted if the
   CI of the net mean day includes 0 on either seed.
