# rolling-memorisation-001: does the walk-forward make the model memorise the first fold? + a per-coin cap proposal

Owner question 2026-11-19 (Arabic): "does the rolling training increase memorisation, since the first folder repeats
more than all the others?", and an external AI proposal (smoothing filters, ternary labels, frequency gating, a
maximum weight per coin). Read-only diagnosis of the finished walk-forwards `panel-rolling-ref-002`, `-std-001`,
`-ls-001` (13 of 14 folds) and `-base-001`; scripts in `docs/audit/2026-11-19-rolling-memorisation/`, outputs on
Drive `nig_timenet_runs/memorisation-check-001/` (`memorisation.json`, `curves.json`, `coins.json`).

Failure reading: not applicable (a diagnosis of existing runs, no new arm).

## 1. Is the first fold repeated? No, not in the weights
- Every trained fold builds a NEW model from its seed (`workflow/rolling_run.py:_train_fold` -> `steps.build`) with
  its own checkpoint folder; weights pass to a later fold only when `retrain_every_k > 1` (`_reuse`), and the jobs
  used 1. A model sees each of its rows at most `epochs` (8) times, whatever the fold.
- The data of the first era IS in every fold's window (expanding), but its share falls: rows before 2021-07 are 42 %
  of fold 0's train rows, 34 / 29 / 24 / 21 / 18 / 17 % in folds 1-6 (the 400000-row cap strides whole timestamps
  over the whole window).

## 2. Memorisation in the learning curves: none seen
In all 55 fold x seed runs (4 jobs), the validation loss improves or flattens over the 8 epochs and never turns up;
the training loss stays ABOVE the validation loss (e.g. ref-002 fold 6: train -5.30 -> -5.78, val -6.17 -> -6.39)
and the cross-section rank loss is lower on validation (0.68-0.70) than on train (0.71-0.72). Fold 6 looks like fold
0: no growth of a train / val gap with the fold index. The models under-fit rather than memorise (the selection
metric is flat after epoch 2-4). `panel-rolling-ls-001` trains exactly like ref-002 (same seed and data: identical
curves); it only picks another epoch (close rank IC equal on 13 folds, mean difference +0.0002).

## 3. What the window does instead (two facts that matter more)
- **The most recent data is never trained on.** The inner validation tail is the last 15 % of each window by time
  (`RollingSettings.val_share` 0.15, D-031): 5.5 months before fold 0's test, 11 months before fold 6's. Those months
  only choose the epoch; the model trades the test half-year with a training end 5-11 months old.
- **The same old coins every fold.** `split.top_coins` 100 = the longest-listed coins of the train rows: 61-65 of
  the 100 were listed before 2021, the coin set of fold 6 shares 82 % with fold 0 (Jaccard), and coins listed later
  never enter, however liquid.

## 4. The per-coin cap proposal, measured
- A coin's weight in a day's equal-weight book is at most 1.2 % (median 1.1 %): a cap of 20 % (or 5 %) never binds.
  The losses are not concentrated: the 10 worst coins hold 18-23 % of all losses (101 coins), the 10 most frequent
  coins 18 % of the worst 1 % rows.
- The volatile coins hold more of the tail losses (worst 1 % rows by volatility quintile, calm -> volatile: 10 / 17 /
  21 / 25 / 27 %), but dropping the most volatile fifth leaves the net unchanged (ref-002 s0 -0.0885 % -> -0.0883 %
  per row). The weak group is the CALMEST fifth: hit 0.488-0.494 (below 50 %) and the smallest gross (0.05-0.08 %
  per row against 0.16 % of cost).
- The rest of the proposal against this project: time split, purge and embargo exist (rule D2); Ledoit-Wolf and box
  constraints belong to a portfolio optimiser this project does not have; the model already has a trend / seasonal
  decomposition (frequency gating); the inputs are 3 channels (already parsimonious); a centred Savitzky-Golay filter
  looks ahead and any smoothing removes the last-bar shape that is a measured predictor (DS1); no order book in the
  artifact. Ternary labels (a dead zone of +-cost) are the one idea that meets a measured problem (the cost wall,
  B-074), and the selective rule that keeps the top-confidence rows is its closest relative here.

## Next hypotheses
1. Refit on train + validation with the chosen epoch count (or `val_share` 0.05), so the last months are trained on.
   Refuted if the close rank IC and the pooled net do not move beyond the seed spread on the same 7 folds.
2. Pick the panel's coins by recent liquidity instead of listing age. Refuted if the IC per fold does not move.
3. A dead-zone (ternary) close label or a trade filter at |predicted move| > cost, read on the calm coins.
