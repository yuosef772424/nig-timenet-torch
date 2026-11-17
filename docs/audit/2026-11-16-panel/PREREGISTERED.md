# Panel audit 2026-11-16: metrics and accept rule, written BEFORE any number on real data was read

Owner task 2026-11-16 ("improve the panel (multi-coin) model and prepare the full-data Colab training"). One session
did the three audits itself (the owner cancelled the subagent campaign); the reports are `A_architecture.md`,
`B_trainer_loss_uncertainty.md`, `C_normalisation.md` in this folder. This file was committed before the arms and the
looks ran; the code is `eval/day_metrics.py` (definitions below are its docstrings).

Every CPU number in this folder is **a look, not a result** (rule C5); >= 2 seeds per arm (rule C4).

## Metrics

A day = the UTC calendar day of the prediction time (the close of the last input bar); a row = one coin at one
timestamp; `realised_i` = close(t + horizon) / close(t) - 1; `pred_i` = the model's predicted close return (NIG mean,
decoded). Rows of one day are averaged; with a label longer than the row step (real artifact: horizon 2 bars, step 1)
the day is the mean over the overlapping tranches.

1. **Absolute direction.** Per day, the equal-weight portfolio of `sign(pred_i) * realised_i` over all coins. Reported:
   the share of winning days, gross of fees.
2. **Market-neutral.** Per timestamp, long the top 20 % and short the bottom 20 % of the coins by
   `pred_i - mean_j pred_j` (half a unit each side); per day the mean over its timestamps. Reported: the share of
   winning days, gross of fees, and the cross-sectional rank IC of the relative move
   (Spearman of `pred_i` with `realised_i - mean_j realised_j` inside the timestamp, mean over timestamps).
3. **Baselines both must beat.** Metric 1: always-long; yesterday's market sign (sign of the mean trailing 24 h return
   of the coins of the timestamp); the last-bar asymmetry of DS1 (`log(H/C) + log(L/C)` of the last bar, sign fitted
   on VAL). Metric 2: the previous day's relative-move ranking, as momentum and as reversal; the last-bar asymmetry.
   High / low heads: the HAR volatility baseline (`eval/vol_baseline.py`, already in the test report).
4. **Owner's aspiration: 80 % winning days, gross.** Reported: the measured share with a CI over days (block bootstrap,
   5-day blocks, 95 %; the Wilson interval is in the JSON). **Accept rule**: on >= 2 seeds, the model's day P&L AND
   its winning-day share are above EVERY baseline's with a paired 5-day-block bootstrap CI that excludes 0
   (`day_metrics.paired_days`, `accept`). Also reported: **T-profit** = the mean day P&L after 0.0008 per side (a row
   = one round trip = 2 sides; metric 2 = 2 sides per unit of gross exposure), the main goal (owner 2026-11-07).

## What is fitted where (no leakage)

- Splits are by time with whole timestamps on one side; purged by label end (`data_io/split_plan.py`).
- Statistics (input scalers, label scale, beta_i, sigma_i, ridge weights) are fitted on TRAIN rows only.
- Baseline orientation and confidence thresholds are fitted on VAL and applied to TEST.
- The real CPU slice lies inside the FULL run's training era (it ends at the full run's train end,
  `environments.panel_slice`): nothing chosen here has seen the full run's validation or test rows. The model-free
  look on the full artifact (report C) reads the full run's TRAIN and VAL rows only, never TEST.

## Known answer (E4)

`data_io/synthetic_panel.py`: one market factor, a beta per coin, a coin-specific drift of planted IC 0.15. The oracle
(the exact conditional mean) is scored with the same code, in the same tables as the arms: an arm is judged against
the oracle's share of winning days, not against 100 %.

## Questions per auditor

- A (architecture): do `cross_input='relative'`, `cross_coin_layers` (kinds `mean` / `attention`) move metric 1, metric 2
  and the relative rank IC on E4 and on the real slice? Is the encoder used (B-070, N-20)? What is missing?
- B (trainer, loss, uncertainty): does the NIG sd rank |error| across coins and across time? T-sharp, coverage,
  direction bias (mean predicted sign vs realised, per coin and per day). Does selecting by confidence raise the
  share of winning days? Does a cross-sectional loss term help?
- C (normalisation): raw; minus the timestamp mean; minus beta_i x market; (r - beta x market) / sigma_i;
  cross-sectional ranks; a top-coin market. For each: cross-coin correlation, effective number of coins
  (`doctor.data_checks.overlap` formula), and what a ridge on simple features reaches.
