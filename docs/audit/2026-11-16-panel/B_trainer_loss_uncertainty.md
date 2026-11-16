# Report B: trainer, loss and uncertainty of the panel model (panel audit 2026-11-16)

Metrics and accept rule: `PREREGISTERED.md` (written before these numbers). Every number here is **a look, not a
result** (rule C5). Arms, data and devices: report A. Tool: `eval/uncertainty_use.py` (written into every arm's
`panel_report.md`); thresholds are fitted on VAL and applied to TEST.

## What was read

- `trainer/tasks.py` `evidential_task_loss`: the NIG negative log-likelihood + evidence regulariser + the
  confidence-calibration term, a mean over ALL rows of the batch.
- `trainer/trainer.py`: whole-timestamp batches (`group_batches`), evaluation on whole-timestamp chunks, selection
  by a registered metric (`val_rank_ic_reg` on the panel arms so far).
- `eval/battery_adapter.py`: the predictive sd = `nig_predictive_std(nu, alpha, beta)` decoded to return units.

## The questions, with the known answer first (E4)

The ORACLE of E4 (the exact conditional mean and sd of every row), read by the same code, TEST 116 days:

| question | oracle's answer on E4 |
|---|---|
| sd ranks the absolute error across coins / across time / pooled | +0.22 / +0.34 / +0.38 |
| T-sharp of the close head (share of rows with sd < predicted move) | 0.000 |
| coverage of mean +- 1.28 sd; sd of the standardised error | 0.82; 0.96 |
| share of rows predicted up (realised) | 0.50 (0.49); per coin 0.43 .. 0.57 |
| winning days, metric 1 / metric 2, all rows | 87.1 % / 91.4 % |
| the same, only the top 20 % of rows by confidence (mean / sd) | 87.1 % / 78.4 % |
| the same, only the 20 % of rows with the lowest sd | 73.8 % / 80.3 % |

Two things follow before any model is read: **a close T-sharp of 0 is the right answer** when the predictable part
is small against the noise (the true sd is above the true mean move on every row), and **dropping rows by
confidence makes even the oracle's book worse** (it gives up breadth, which is what turns a small IC into winning
days).

## Uncertainty and direction per arm (TEST)

E4:

<!-- TABLE:e4:unc -->

Real long slice:

<!-- TABLE:real:unc -->

Selection by confidence: winning days of metric 1 / metric 2 (gross), thresholds from VAL.

E4:

<!-- TABLE:e4:sel -->

Real long slice:

<!-- TABLE:real:sel -->

## Findings

1. **The NIG sd ranks the absolute error, across coins and across time, on every arm.** Real slice, close head:
   about +0.28 across coins and +0.22 across time (high / low the same size); E4: +0.21 / +0.31 against the oracle's
   +0.22 / +0.34. It is a volatility forecast and a good one. The cross-section term does not damage it.
2. **Calibration is acceptable, slightly wide**: coverage of the 80 % interval 0.84-0.89 on real data, sd of the
   standardised error 1.0-1.2. Nothing to fix before the full run; the interval scale can be fitted later
   (`eval/calibrator.py`).
3. **T-sharp of the close head is 0 on every arm, and it should be** (the oracle's is 0 too). It stays information
   only (O-1 closed).
4. **Direction bias: the close mean carries a level.** On E4 the arms without `cross_center` predict "up" for only
   14-30 % of the rows of a coin (realised 49 %), and a fifth of the timestamps get ONE sign for all 50 coins: the
   sign portfolio is then a market bet (47-66 % winning days) although the ranking is at the oracle's level. With
   the mean centred inside its timestamp the same book wins 81-86 %. On the real slice the share of predicted-up
   rows per coin runs from about 0.26 to 0.97: the model has learnt each coin's training-era drift. Cause: 40-60 %
   of the label's variance is the common move, and the pooled NIG loss fits the train period's level into the mean.
5. **Confidence does not select trades.** Keeping the top 20 % of rows by mean / sd, or the calmest 20 %, does not
   raise the share of winning days on real data on either metric, and on E4 it LOWERS metric 2 (as it does for the
   oracle). Ranking the whole cross-section by mean / sd instead of the mean changes little. The sd is useful for
   SIZE (it ranks the error), not for choosing which rows to trade; a sized book was not built here (the battery's
   `sigma_sizing` section covers the per-row case).
6. **The cross-section rank term raises the rank IC of every head on real data, on both seeds where both exist**,
   and costs nothing on E4. 60,000-row slice (CPU): close 0.066 / 0.060 -> 0.089 / 0.097, low 0.31 / 0.33 -> 0.345 /
   0.353, high 0.23 / 0.235 -> 0.252 / 0.266. Long slice: relative rank IC of the per-coin arm 0.065 -> 0.092, of the
   relative arm 0.082 -> 0.095 (seed 0). E4: the per-coin model with the term alone reaches the relative-input level
   (IC 0.089 -> 0.123, metric 2 81 % -> 87 %). The `ic` kind (values instead of ranks) is level with `rank_ic` or
   below it; ranks are the safer target for 8 h returns (report C, finding 4).
7. **The higher rank IC does not show in the winning days at 115 days** (metric 2 stays 55-64 % for every arm). It is
   the one consistent gain of this audit and it is small; whether it is worth anything is for `panel-001` to say.
8. **Selecting the epoch by the long / short book (`val_ls_sharpe_close`)**: on E4 it picks a good model after one
   epoch (88.8 %); on the real slice the validation value is about 0 at every epoch, so it selects on noise there.
   Kept as a registered metric, not used in the arms of `panel-001`.

## Proposals

| # | Proposal | Evidence | Built as |
|---|---|---|---|
| B1 | Add the rank term to panel runs | finding 6 | `TrainSettings.cross_section_loss = "rank_ic"` (off by default); arms `relative_rank_loss`, `relative_rank_loss_center` |
| B2 | Centre the close mean inside its timestamp | finding 4 (E4: 47-66 % -> 81-86 % in metric 1) | model option `cross_center` (off by default) |
| B3 | Do not gate trades by confidence; keep the sd for sizing | finding 5 | nothing built; the report rows stay in every `panel_report.md` |
| B4 | Keep `val_rank_ic_reg` as the selection metric of panel runs; `val_ls_sharpe_close` is there to try on the full data | finding 8, report C finding 5 | registered, not a default |
| B5 | Leave T-sharp of the close head as information | finding 3 | no change |

Not verified: beta-NLL, EMA and gradient clipping on the panel route (TR1 switches, untouched here); the rank term
with a weight other than 1.0; the gates of rule C6 on any of the new arms (a winner must pass them before it becomes
a default).
