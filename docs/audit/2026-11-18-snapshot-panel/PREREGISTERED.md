# Snapshot panel model (the owner's "100 x 3" design) — pre-registration (written 2026-11-18, before any number)

## Question
Is a model that reads the WHOLE moment at once (every coin of the timestamp, its 3 channels over the 32 bars, as one
input; coin-specific input weights and coin-specific output weights; NIG outputs for every coin) more accurate than
today's panel model (each coin a row through shared weights, relative inputs, cross-section rank term)?

Owner (2026-11-18, Arabic): "design the model structure to match the size of the inputs and the outputs; this is not
about simplifying for speed but about the best accuracy; test it on a small sample for errors; we will test it
directly on Colab on the full data with rolling training."

## Arms (one commit, one data fingerprint, the same folds, seeds 0 and 1)
| arm | model | why |
|---|---|---|
| `base` (baseline arm) | today's panel model: `nig_timenet_v2`, encoder gru, `cross_input: relative`, `cross_section_loss: rank_ic`, selection `val_rank_ic_reg` (= `panel-rolling-001`) | the best panel configuration so far (`docs/results/panel-001.md`) |
| `joint` | `snapshot_panel` with `shared_path: false`: the owner's design as stated (the moment as one vector, per-coin input and output weights) | the hypothesis itself |
| `hybrid` | `snapshot_panel` with both paths (joint moment encoder + the shared per-coin encoder, joined per coin) | the design the lead expects to be the most accurate: it keeps what the shared model already reads and adds what only the joint view can read |

All three train with the same loss (NIG + class heads + the cross-section rank term), the same selection metric
(`val_rank_ic_reg`) and the same rows (the 100 coins with the most TRAIN rows of each fold, whole timestamps).

## Metrics (in this order)
1. Primary: the close head's rank IC WITHIN the timestamp (relative ranking), pooled over the test days of all folds,
   per seed, with its day-block CI (`rolling_summary`, `panel_report`).
2. The same for high and low (volatility ranking; their trivial baseline is the last bar's range).
3. Day metrics of the top / bottom book after the 0.0008 per-side cost (`day_metrics`): share of winning days and
   mean net per day, next to always-long, market sign, momentum and reversal.
4. Diagnostics (not evidence): VAL NIG NLL of each head, 80 % coverage, train-minus-val gap (memorisation: the joint
   arm sees about 100 x fewer independent rows than its parameter count suggests).

## Known answer first (E4, CPU, before any GPU minute)
E4 plants a coin-specific move that a reader sees only after removing beta_i x market from the window (planted IC
0.15; `summary['readers']` gives the IC of the three readers). A model with coin-specific input weights can learn
beta_i per coin, the shared per-coin model cannot (it removes the mean, not beta_i x market). Expected on E4: joint >=
base on the close relative IC. If the joint arm does NOT beat base on E4, the 100 x 3 view has no advantage even where
one is built into the data, and the real-data result is read in that light.

## Accept rule (written now)
An arm X replaces `base` as the panel candidate only if, on the pooled test days of the walk-forward, on BOTH seeds:
(a) its close relative rank IC exceeds base's by more than max(0.01, base's seed spread), AND (b) high and low rank IC
are not lower than base's by more than 0.01, AND (c) its net day P&L of the top / bottom book is not below base's CI.
Otherwise X is read with the failure reading of `docs/PHILOSOPHY.md` section 3 before anything else is tried. The
number of variants tried so far (3 arms) widens the CIs (`n_variants`).

## What would refute the owner's argument ("the 300 outputs compensate for the few samples")
The joint arm's train-minus-val gap grows epoch by epoch while its val IC stays below base on both seeds, on E4 and on
the real folds.
