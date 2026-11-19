# Pre-registration: rolling-single-002 (written 2026-11-19, before the job starts)

## Question
`docs/results/short-side-001.md` found, on `rolling-single-001` (seeds 0 and 1), one selection that came close to the
scorecard verdict: **LONG top 20 % confidence** (`|p_dir - 0.5|` at or above the VAL 0.8 quantile, side long). Its
mean day was +0.195 % [+0.002, +0.420] (seed 0) and +0.200 % [-0.035, +0.380] (seed 1), with 56 % winning days and a
positive busy-day mean. It was negative without its best 5 % of days. That row was picked by looking at those two
seeds' TEST days. Does it hold on two NEW seeds of the same configuration?

## Job
`docs/jobs/rolling-single-002.json`: the configuration of `rolling-single-001` unchanged (GRU, lr 3e-4, 100
longest-listed coins per fold, 400k train rows per fold, 7 expanding folds, 8 epochs, patience 3, selection
`val_raw_loss`), seeds **2 and 3**, on this branch's commit (the battery prints the scorecard sections per fold and
saves the VAL tables). Smoke first: `rolling-single-smoke-001` on the same commit (rule E5).

## Accept rule (fixed now)
The candidate passes only if, on the pooled TEST days of the 7 folds, **for BOTH seeds 2 and 3**, all of these hold:
- mean day net above 0, with the lower bound of the 5-day-block CI above 0 at level 1 - 0.05 / 27 (the scorecard's
  Bonferroni over its 27 rows);
- winning days above 50 %;
- mean day above 0 without the best 5 % of days;
- mean day above 0 on busy days (at least the median number of trades).

Thresholds are taken from each fold's own VAL table, as the battery does.

## What does NOT count
- Any other scorecard row that passes on seeds 2 and 3 is a new hypothesis, not a result: it was not pre-registered.
- New seeds test robustness to the model's randomness, NOT to data snooping. The data period is the same as
  `rolling-single-001`, so a pass is still exploratory (rule C5) and needs data the candidate has not seen.

## Also read (no accept rule)
The log / exp split of the short side on seeds 2 and 3 (`LONG vs SHORT PAYOFF`). Expected from short-side-001: the
short's exp() part is negative in every fold and close to its predicted value.
