# The H3 test again without the normalisation layer (owner 2026-11-18) — refuted, the same partial gain as H

**Exploratory (rule C5).** Same data and split as `docs/results/har-range-h3-001.md` (60-coin public-kline slice,
train 2023 / test 2024, 61,980 test rows), two versions x two seeds as the owner asked: A0 = run default; N = A0 with
`use_instance_norm = False` (`InstanceNorm(normalize=False)`: no per-window centring / scaling, the train-scaled
input reaches the encoder; stats path, soft clip, affine unchanged; NOT combined with `har_range`). Pre-registration
`docs/audit/2026-11-17-batch-by/PREREGISTERED_4.md` and the switch at commit f168097 (before any number). Reading
`h3_check.py <artifact> <study> N`, `failure_reading.py <artifact> <study> N`. Colab L4, about 67 s per model. The A0
models reproduce study 3 exactly (same per-epoch losses, same test loss -5.6486 / -5.6798): the runs are
deterministic. Gates of rule C6 not run (deviation, as in study 3).

## Verdict against the pre-registered rule

Rank IC by timestamp per quarter of 2024 (mean of seeds 0 and 1); bound = the better of the 6-bar mean range and the
ridge on the HAR features fitted on 2023:

| head | quarter | ridge (bound) | A0 | N | N - A0 | H of study 3 |
|---|---|---|---|---|---|---|
| high | Q1 | 0.327 | 0.293 | 0.290 | -0.004 | 0.292 |
| high | Q2 | 0.363 | 0.331 | 0.329 | -0.002 | 0.330 |
| high | Q3 | 0.331 | 0.302 | 0.301 | -0.002 | 0.300 |
| high | Q4 | 0.352 | 0.325 | 0.323 | -0.002 | 0.324 |
| low | Q1 | 0.437 | 0.383 | 0.392 | +0.009 | 0.393 |
| low | Q2 | 0.425 | 0.365 | 0.371 | +0.006 | 0.377 |
| low | Q3 | 0.388 | 0.325 | 0.332 | +0.007 | 0.334 |
| low | Q4 | 0.393 | 0.341 | 0.352 | +0.011 | 0.348 |

Whole year per seed: high A0 0.310 / 0.316, N 0.305 / 0.316; low A0 0.346 / 0.360, N 0.354 / 0.369.

- Rule (a), N reaches the bound in >= 3 of 4 quarters on both heads: **0 / 4 on both** -> fails.
- Rule (b), N beats A0 in >= 3 of 4 quarters on both heads and on every seed: low 4 / 4 (both seeds), high 0 / 4 ->
  fails.
- **Refuted.** Read: N closes 15 % of the A0-to-bound gap on the low head (H: 17 %) and -8 % on the high head (H:
  -6 %).

## What the model learned

Removing the per-window normalisation does what adding the range terms did, almost number for number: the low head
ranks better in every quarter and on both seeds (+0.006 .. +0.011; calm third 0.364 -> 0.368, volatile third 0.196 ->
0.209), the high head does not move (-0.002 .. -0.004). Two different ways of giving the model the window's
volatility level give the same small gain and stop at the same place, well under the ridge.

## Where it is right and where it fails

Low-head rank IC by volatility tercile of the last input bar: calm A0 0.364 / N 0.368, middle 0.211 / 0.216, volatile
0.196 / 0.209 (the gain is largest in the volatile third, the opposite of where the models rank best). Per coin the
low-head error moves by at most 0.0001 (TRX, SUI, COMP gain; EGLD, DYDX, ONE unchanged). Test raw loss: A0 -5.649 /
-5.680, N -5.653 / -5.683.

## The samples it failed on

The same rows as studies 2 and 3 (A0 seed 0, high head): UNI 2024-02-23 (+57 % from a last-bar range of 0.016), XTZ
2024-11-18 (+44 %), CELO 2024-03-11 (+42 %), HBAR 2024-04-23 (+40 %), and the November-December 2024 rally (SAND, TRX,
XLM): squeezes out of a quiet base that no volatility input announces (B-077).

## Bias

Close head: A0 predicts up on 47.3 % of the rows, N on 46.7 %, realised 49.8 % (mean prediction -0.00004 / -0.00003
against a mean label of +0.00047). Unlike H (49.3 %), N does not reduce the short lean.

## Selective use

Not measured (battery off; T-profit was not the question). Rules tried on TEST: none.

## Why it failed

ONE cause, now seen on two different input changes: **the model memorises after the first epoch**. In all four models
the best epoch is epoch 1: the training loss keeps falling (N seed 0: -6.34, -6.58, -6.65, -6.72) while the
validation loss rises (-5.69, -5.65, -5.68, -5.64); A0 the same as in study 3. Whatever form the volatility level
reaches the model in (range terms in the readout, H; no normalisation, N), one useful epoch lifts the low head by
about 0.01 and leaves the high head where it was, while a ridge on the same inputs gets 0.03-0.06 more. The limit is
the training (one epoch before memorisation), not the representation of the input.

## Next hypotheses

1. **H5 (now the first): stop the memorisation, then re-test H / N.** A lower learning rate (3e-4) or stronger weight
   decay should move the best epoch past 1; with more useful epochs H or N should close more of the gap. Refuted if
   the best epoch stays at 1, or if with a later best epoch neither reaches the bound in 3 of 4 quarters on both heads.
   Cheapest test: A0 / N (or H) x 2 seeds with lr 3e-4 on this split (about 5 minutes on the L4).
2. **H6: the high head loses the signal after the readout.** H and N both moved only the low head; the doctor's card A
   showed the high label's probe at the trunk under the simple features (0.150 vs 0.204) while the low label passed.
   Refuted if card A on an N or H model shows the high-label probe at the trunk at or above the simple baseline.
   Cheapest test: the doctor card A with `--model '{"use_instance_norm": false}'` on the Colab CPU (minutes).
