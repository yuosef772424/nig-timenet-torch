# Pre-registration: the panel model with std normalisation, walk-forward (owner task 2026-11-18)

Written BEFORE any number of the jobs below is read (`docs/PHILOSOPHY.md` section 2, step 1). Exploratory (rule C5).

## The question
Owner message 2026-11-18 (Arabic): "test the panel model, but using std normalisation of the batch data; use the
rolling training". Read as: the panel arm `relative_rank_loss_logret_std` of `workflow/panel_study.py` (D-033: the
TF `logret_std` normalisation in full, i.e. train-standardised inputs, which is already the run default
`input_scaler: train_std`, PLUS per-head train-standardised labels `model.target_norm: logret_std`), with the
relative inputs (each coin against the cross-section of its timestamp batch) and the cross-section rank term,
walked forward. Does the per-head label standardisation change what the panel model reads, and does it pay?

## Arms, data, folds (one commit, one data fingerprint `c572b1b36960`)
| job | arm | model overrides |
|---|---|---|
| `panel-rolling-std-001` | candidate | `encoder gru`, `cross_input relative`, `target_norm logret_std` |
| `panel-rolling-ref-002` | reference (baseline arm) | `encoder gru`, `cross_input relative` (= `panel-rolling-001`, re-run on this commit) |

Both: train `cross_section_loss rank_ic`, selection `val_rank_ic_reg`, 8 epochs, patience 3, batch 512, lr 3e-4,
`split.cross_section` with the 100 longest-listed coins per fold, at most 400000 train rows per fold, 7 expanding
folds (first train 1095 days, test and step 180 days, embargo 2 days), seeds 0 and 1, day metrics on (test days
pooled per seed, about 1260 days). Smoke first: `panel-rolling-std-smoke-001` (rule E5).
`panel-rolling-001` (commit `ebbd336`, the same reference arm) is a reproducibility check of the reference.

## Metrics and baselines
1. T-profit (the goal): the pooled-days panel report of each seed (`pooled_days_seed_<s>/panel_report.md`): the
   absolute and the market-neutral day books net of 0.0008 per side, next to every baseline of
   `eval/day_metrics.py`, day-block CI.
2. Paired against the reference: per fold and seed, the close / high / low rank IC by timestamp
   (`rolling_folds.csv`); the paired day difference of the neutral book on the pooled days.
3. Diagnostics the change is expected to move: high / low / close coverage of the 80 % interval and
   `share_std_below_move`, the bias of the close mean (share predicted up vs realised).

## Accept rules (fixed now)
- **Profit (T-profit):** the candidate's book beats every day-metric baseline on the pooled days with a CI that
  excludes 0, on BOTH seeds.
- **Diagnostic gain (kept as a switch, not a default):** close rank IC above the reference in >= 6 of 7 folds on
  both seeds, and the pooled mean difference larger than twice the seed spread, with no high / low rank IC loss
  over 0.01.
- **Refuted:** close rank IC not above the reference in at least 4 of 7 folds on either seed.

## Expectation (written before the run)
The rank term is invariant to the per-head affine map, so the close ranking should move little; the change is
expected mostly on the NIG heads' starting point (centred high / low), i.e. on calibration (coverage,
`share_std_below_move`) and on the close bias. A profit win is not expected: the reference's gross edge is under
half of its cost (B-074).

## Known answer
E4 (the synthetic panel, planted specific IC 0.15) through the `panel` entry, arms `relative_rank_loss` and
`relative_rank_loss_logret_std`, seeds 0 and 1, run on Colab beside the walk-forward; read against the oracle's
0.15, not against 1.
