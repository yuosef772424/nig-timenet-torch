# PREREGISTERED (3) — H3: give the model the range terms it under-reads (owner 2026-11-18)

Written before any number of this study was read. Owner's request (Arabic, 2026-11-18): "test the hypothesis [H3],
but there is no need to train many models: two versions are enough, each with two seeds". Exploratory (rule C5).

## Hypothesis

H3 (`docs/results/batch-by-wf-002.md`, next hypothesis 1): the model under-reads the volatility level of its input
window (a ridge on the HAR range terms beat it on high and low in every fold; the instance norm erases the level).
Handing the HAR range terms of the RAW window to the readout (switch `har_range = [0, 1]`, `model/har_range.py`,
off by default) lets the model reach the model-free bound.

## Design (4 trainings: 2 versions x 2 seeds)

- Data: the 60-coin slice of study 2 (`vision_slice`, 60 default coins, 4h, 2023-01 .. 2024-12, stride 2, 129,900
  rows; Drive `nig_timenet_runs/batch_by_002/real60`).
- ONE split instead of walk-forward folds (the owner asked for few models): `rolling_run` with one expanding fold,
  train window 365 days (2023), embargo 1 day, test window 350 days (2024); the last 15 % of the train window is the
  inner validation (epoch selection). Test blocks for the reading: the four calendar quarters of 2024.
- Versions: **A0** = the run default (rows, batch 256); **H** = A0 + `har_range = [0, 1]` (input channels high, low).
  Seeds 0 and 1. 4 epochs, AdamW lr 1e-3, best epoch by `val_raw_loss`, battery off, Colab L4.
- The gates of rule C6 are NOT run first (owner: few models; the switch only adds a readout input); recorded as a
  deviation.

## Metric and bound

Rank IC by timestamp of the high and low heads, per quarter of the test year, from the saved test tables (labels
joined from the artifact), mean over the 2 seeds. Bound: the better of the 6-bar mean range and the ridge on
`doctor.common.simple_features` fitted on the TRAINING rows (penalty chosen on the inner validation, grid {1, 10,
100, 1000, 10000, 100000}), on the same rows and quarters.

## Accept rule (H3 supported if BOTH)

(a) **H reaches the bound**: H's IC >= the bound's IC in at least 3 of the 4 quarters, on BOTH heads;
(b) **H beats A0**: H's IC > A0's IC in at least 3 of the 4 quarters on BOTH heads, AND on each seed separately H's
whole-year IC > A0's on both heads.
H3 refuted otherwise. Read, not judged: the gap closed (H - A0) / (bound - A0) per head, the test raw loss, the close
head, coverage.
