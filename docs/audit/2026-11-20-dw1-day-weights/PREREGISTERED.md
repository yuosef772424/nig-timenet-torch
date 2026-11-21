# Pre-registration: DW1, day-balanced and failure-weighted loss (owner 2026-11-20)

Written before any number of these runs. Exploratory (rule C5). Owner message 2026-11-20: "update the loss function so
the loss weights are updated per day, so that every day in the training data is balanced, and any day it fails on is
penalised more (or in the suitable way) so it tries to generalise on it; then run an experiment on Colab".
Mechanism: `trainer/day_weights.py`, decision D-039.

## Question
Does weighting the training loss by UTC day (every day the same total weight; days with a high loss, measured at the
start of each epoch, weighted up) make the close model generalise better across TEST days (fewer days where it fails),
and does that reach the long book (T-profit)?

## Arms (one commit, one artifact `art1`, configuration of `rolling-r23-001` = arm R of `rolling-ind-001`)
| arm | job | train.day_weights |
|---|---|---|
| R (control) | `rolling-dw1-r-001` | None (every row weight 1) |
| DW | `rolling-dw1-dw-001` | `{}` = balance True, strength 1.0, max_ratio 5, momentum 0.5, shrink_rows 20 |
| BAL | `rolling-dw1-bal-001` | `{"strength": 0.0}` = balance only (separates the two parts of DW) |

Seeds 0 and 1, walk-forward 7 folds (expanding, 180-day test), folds split over processes (`chain.sh`). Smoke
`rolling-dw1-smoke-001` (arm DW, 2 epochs: the measurement path runs on the GPU) first (rule E5).

## Metrics (TEST of every fold pooled per seed; `dw1_read.py` in this folder; cost 0.08 % per side)
- **P1 day-loss tail**: per TEST day, the mean Gaussian NLL of the close return under the model's predictive mean and
  sd (`mu`, `sigma` of the battery table); CVaR10 = the mean of the worst 10 % of days. Lower is better.
- **P2 day-IC tail**: per TEST day, the mean over its timestamps of the close rank IC (mu against y_ret, timestamps with
  >= 5 coins); the mean of the lowest 10 % of days. Higher is better. Also the mean day IC and the share of days > 0.
- **P3 long book** (as `read_arms.py`: VAL 90th percentile of mu per fold, buy every TEST row at or above it): mean day
  net, mean without the best 5 % of days, CVaR5 of the day net (mean of the worst 5 % of days), winning days.
- Information: the paired per-day difference DW - R of the day NLL and the day IC (block-bootstrap CI over days), per
  fold; the training-day table `day_weights.csv` of each fold (which training days were weighted up).

Trivial baselines on the same rows: always-long (every row, P3), IC 0 (P2); P1 is read only between arms.

## Accept rules (both seeds)
1. **"DW generalises better over days"**: on BOTH seeds DW beats R on P1 (lower CVaR10 day NLL) AND on P2 (higher
   lowest-decile day IC), AND DW's mean day IC is not below R's by more than 0.005.
2. **"DW pays more"**: on BOTH seeds DW's long-book mean without the best 5 % of days is above R's, AND DW's CVaR5 of
   the day net is not worse than R's.
3. The same two rules for BAL against R (information: which part carries a gain).

DW is a candidate for the next walk-forward (M3 / V1 with day weights) only if rules 1 and 2 hold. Every rule that
fails gets the failure reading of `docs/PHILOSOPHY.md` section 3 in `docs/results/dw1-001.md`.
