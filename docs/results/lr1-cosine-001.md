# lr1-cosine-001: a cosine learning-rate decay in the walk-forward (exploratory, rule C5)

Owner question 2026-11-19: "do you notice a gap in the settings or in the training that can be improved, then train the
model". Gap found: the run path never passed a learning-rate schedule (`flat_run.fit_settings_of` left `FitSettings.lr`
None), so every run so far trained at a constant lr; in `rolling-ls1-a0-001` the best epoch was the last of 8 in 4 of 7
folds. New switch `TrainSettings.lr_schedule` / `lr_min_ratio` (D-040, "constant" by default). E1 gate passed: cosine
beats constant on mu ranking on 3 of 3 seeds (`docs/audit/2026-11-19-loss-side-heads/E1_LR_COSINE.txt`). Arm C =
`rolling-lr1-cos-001` (A0 + cosine from 3e-4 to 3e-5 over the same 8 epochs), seeds 2 and 3, Colab L4, commit 2764907,
after the smoke `rolling-lr1-smoke-001` (rule E5). Reference A0 = `rolling-ls1-a0-001` (same configuration and seeds;
the default path is byte-equal). Accept rule written before the numbers: `PREREGISTERED_LR1.md`; output
`LR1_OUTPUT.txt` (same folder).

## Verdict

**C fails its pre-registered rule.** (1) close rank IC (fold mean): seed 2 0.0221 against 0.0219 (above), seed 3
0.0199 against 0.0205 (below); (2) lower selected validation loss in 4 / 7 (seed 2) and 2 / 7 (seed 3) folds, the rule
needs 5 / 7; (3) no book loses more than 0.05 % per trade (passes). The constant lr was not the bottleneck.

## What the model learned

- The volatility heads gain a little on both seeds: high / low rank IC 0.346 / 0.399 and 0.344 / 0.397 against A0's
  0.342 / 0.392 (+0.002 .. +0.007), in the direction E1 showed.
- The close head does not change beyond the seed spread (0.022 / 0.020 against 0.022 / 0.021).

## Where it is right and where it fails

- By fold, the selected validation loss is lower with C in the early folds (0-3: 4 / 4 on seed 2, 2 / 4 on seed 3) and
  higher in the late folds (4-6: 0 / 3 on both seeds). The differences are 0.003-0.02 nats, the size of the seed spread.
- With C the best epoch is the last one in 6 / 7 folds: expected, the lr is lowest there; it does not mean more
  training would help.

## The samples it failed on

The books are the A0 books within a few basis points: SHORT top 5 % by -mu -0.124 / -0.024 % per trade (A0 -0.126 /
-0.032 %), LONG top 5 % by mu +0.398 / +0.353 % (A0 +0.335 / +0.353 %); their worst trades are the squeezes of
`ls1-side-heads-001`.

## Bias

No change of lean measured beyond the seed spread; the long side carries the gross edge, as in A0.

## Selective use

The same pre-registered books as LS1 (top 5 % by mu and E per side): none moves by more than 0.07 % per trade; no
short book above 0 per trade; the long books stay negative without their best 5 % of days.

## Why it failed

**Hidden by the size of the remaining signal, not by the optimiser.** Between epochs 5 and 8 the validation loss of A0
moves by ~0.005 nats; the schedule changes where in that flat region training stops, not what the model can read. The
close head's skill (IC ~0.02) is limited by the inputs, as every earlier arm showed.

## Next hypotheses

| # | Hypothesis | Refuted if | Cheapest test |
|---|---|---|---|
| R1 | cosine helps the volatility heads (+0.002 .. +0.007 IC on both seeds) | a third seed shows no gain | the same job, seed 4 |
| R2 | the close head is limited by the inputs, not the training: volume / liquidity features (B-055) raise its IC | a volume artifact leaves the close IC at ~0.02 | a volume artifact |
