# H3: the HAR range terms as a readout input (owner 2026-11-18) — refuted, a partial gain on the low head

**Exploratory (rule C5).** 60-coin public-kline slice of study 2 (`vision_slice`, 129,900 rows), ONE split: train
2023 (54,180 rows; inner validation 8,699), test 2024 (61,980 rows). Two versions x two seeds as the owner asked: A0
= run default; H = A0 + `har_range = [0, 1]` (`model/har_range.py`: mean bar range over the last 1 / 6 / 32 raw bars
and their logs -> BatchNorm -> Linear -> GELU, into the readout). Pre-registration
`docs/audit/2026-11-17-batch-by/PREREGISTERED_3.md` and the switch at commit 44249ee (before any number); reading
`h3_check.py`, `failure_reading.py <artifact> <study> H`. Colab L4, about 70 s per model; gates of rule C6 not run
(deviation recorded in the pre-registration).

## Verdict against the pre-registered rule

Rank IC by timestamp per quarter of 2024 (versions: mean of seeds 0 and 1); bound = the better of the 6-bar mean
range and the ridge on the HAR features fitted on the 2023 rows:

| head | quarter | 6-bar range | ridge | A0 | H | H - A0 |
|---|---|---|---|---|---|---|
| high | Q1 | 0.319 | **0.327** | 0.293 | 0.292 | -0.002 |
| high | Q2 | 0.354 | **0.363** | 0.331 | 0.330 | -0.001 |
| high | Q3 | 0.322 | **0.331** | 0.302 | 0.300 | -0.003 |
| high | Q4 | 0.337 | **0.352** | 0.325 | 0.324 | -0.002 |
| low | Q1 | 0.423 | **0.437** | 0.383 | 0.393 | +0.010 |
| low | Q2 | 0.419 | **0.425** | 0.365 | 0.377 | +0.012 |
| low | Q3 | 0.380 | **0.388** | 0.325 | 0.334 | +0.008 |
| low | Q4 | 0.389 | **0.393** | 0.341 | 0.348 | +0.007 |

Whole year per seed: high A0 0.310 / 0.316, H 0.306 / 0.316; low A0 0.346 / 0.360, H 0.356 / 0.369.

- Rule (a), H reaches the bound in >= 3 of 4 quarters on both heads: **0 / 4 on both** -> fails.
- Rule (b), H beats A0 in >= 3 of 4 quarters on both heads and on every seed: low 4 / 4 (both seeds), high 0 / 4
  -> fails.
- **H3 refuted.** Read: H closes 17 % of the A0-to-bound gap on the low head and -6 % on the high head. The ridge's
  penalty sat at the top of its grid on the high head (100,000) again: the bound is a lower bound.

## What the model learned

- With the range terms in its readout, the model ranks the LOW move better in every quarter and on both seeds
  (+0.007 .. +0.012), in the calm and in the volatile third (low IC calm 0.364 -> 0.377, volatile 0.196 -> 0.210),
  and its close head loses most of its short lean (predicted up 47.3 % -> 49.3 % against 49.8 % realised).
- The HIGH move does not change (-0.001 .. -0.003 in every quarter). Neither version reaches what a ridge on the HAR
  features (these range terms plus per-channel means and sds) gets (high 0.327-0.363, low 0.388-0.437).

## Where it is right and where it fails

Low-head rank IC (seeds pooled) by volatility tercile of the last input bar: calm A0 0.364 / H 0.377, middle 0.211 /
0.213, volatile 0.196 / 0.210. Per coin the low-head error moves by at most 0.0002 between the versions (TRX, BTC, ARB
gain; DYDX, RUNE, DOGE lose): the gain is spread, not carried by a few coins. Test raw loss: A0 -5.649 / -5.680, H
-5.660 / -5.683 (H slightly better on both seeds).

## The samples it failed on

The same rows as study 2 (A0 seed 0, high head): upward jumps put at 1-4 % from a quiet last bar — UNI 2024-02-23
(+57 %, last-bar range 0.016), XTZ 2024-11-18 (+44 %), CELO 2024-03-11 (+42 %), HBAR 2024-04-23 (+40 %, 0.014) — and
the November-December 2024 alt-coin rally (SAND, TRX, XLM). The range terms cannot announce a squeeze out of a quiet
base: these rows have a SMALL recent range.

## Bias

Close head: A0 predicts up on 47.3 % of the rows, H on 49.3 %, realised 49.8 %; mean prediction -0.00004 (A0) and
+0.00013 (H) against a mean label of +0.00047. H is the less biased version.

## Selective use

Not measured (battery off; T-profit was not the question). Rules tried on TEST: none.

## Why it failed

ONE cause: **the model memorises after the first epoch, so the readout never gets to learn the range terms**. In all
four models the best epoch (by the inner validation loss) is epoch 1: the training loss keeps falling (A0 seed 0:
-6.33, -6.59, -6.68, -6.75) while the validation loss rises (-5.70, -5.64, -5.67, -5.61); the same in H (-5.72 ->
-5.59 and -5.76 -> -5.64). With one useful epoch the new 6-term input moves the low head a little and the high head
not at all, while a ridge on the HAR features (a superset of the six terms), fitted in one solve, sets the bound. The train - validation gap is
already 0.6 at epoch 1 (the memorisation of B-038 / M1, now seen on the 60-coin slice).

## Next hypotheses

1. **H5: the range terms help once the model stops memorising first.** Train H (and A0) with fewer free parameters
   between the terms and the heads, e.g. the HAR terms feeding the NIG means through a linear skip with no hidden
   layer (the ridge's form), or with a lower learning rate / stronger weight decay so that the best epoch is later
   than 1. Refuted if the best epoch stays at 1 or H still does not reach the bound in 3 of 4 quarters. Cheapest test:
   the same 2 x 2 runs with lr 3e-4 (about 5 minutes on the L4).
2. **H6: the high head needs the terms more directly than the readout gives them.** The doctor showed the trunk
   holding less of the high label than the simple features (0.150 vs 0.204) while the low label passed (0.220 vs
   0.214); H moved only the low head, consistent with the high head losing the signal after the readout. Refuted if
   card A on an H model shows the high-label probe at the trunk at or above the simple baseline. Cheapest test: the
   doctor card A with `--model '{"har_range": [0, 1]}'` (minutes on the Colab CPU).
3. Squeezes from a quiet base are not in any range term; they stay the model's worst rows whatever the batching or
   the range input (B-077).
