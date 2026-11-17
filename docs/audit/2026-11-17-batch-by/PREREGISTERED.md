# PREREGISTERED — batch_by walk-forward on a small real sample (owner 2026-11-17)

Written before any number of this study was read. Owner's request (Arabic, 2026-11-17): "run a walk-forward comparison
between rows, timestamp and day on a small sample of the real data". Exploratory (rule C5): a look on a CPU, not a
result; the owner's Drive artifact and the GPU are not used.

## Question

Does making one training batch = all rows of one timestamp (`timestamp`) or of one UTC day (`day`), with the batches
shuffled instead of the rows (D-034), change what the single-coin model learns, compared with shuffled row batches
(`rows`, the default)? And is any difference due to the GROUPING or only to the smaller batch size that grouping
brings with it?

## Data

Public Binance USDT-perp 4h klines through `data_io/vision_slice.py` (the real artifact's definition: log-move inputs,
log-return labels clipped at 1.0, window 32, horizon 2), 20 liquid unsealed coins (BTC ETH BNB SOL XRP ADA DOGE AVAX
LINK DOT LTC BCH TRX ATOM NEAR UNI ETC FIL XLM AAVE), 2023-01 .. 2024-12, stride 2 (one row per coin every 8 h):
43,520 rows. NOT the owner's artifact (535 coins, TF pipeline): numbers describe the method on real candles.

## Design

- Walk-forward (`workflow/rolling_run.py`, entry of ROADMAP F2): expanding folds, first train window 365 days, test
  window 90 days, step 90 days, embargo 1 day, 4 folds (test quarters of 2024), inner validation = last 15 % of each
  train window (selects the epoch; the test window is read only by the report). Seeds 0 and 1 (rule C4).
- Model and training: the run default (`default_model_settings`, GRU preset + D-023 / D-026 switches), AdamW lr 1e-3,
  4 epochs, best epoch by `val_raw_loss`, CPU. Battery OFF (T-profit is not judged in this look: 20 coins x 90 days
  give too few trades per fold for the battery's CIs).
- Arms (one commit, one data fingerprint):

| arm | batch_by | batch_size | rows per batch (about) | role |
|---|---|---|---|---|
| A0 | rows | 256 | 256 | baseline (project default) |
| A1 | day | (unused) | 60 (20 coins x 3 rows per day) | candidate |
| A2 | timestamp | (unused) | 20 (20 coins) | candidate |
| C60 | rows | 60 | 60 | size control of A1 |
| C20 | rows | 20 | 20 | size control of A2 |

## Metrics (per fold x seed, on the fold's TEST window)

1. **Primary: the trainer's own raw loss on the test rows** (`Trainer.evaluate`; the objective every arm optimises,
   same loss schedule for every arm since all run 4 epochs). Trivial baseline: the A0 arm (what the default gives).
2. Rank IC by timestamp per head (`targets.<head>.skill.rank_ic_by_timestamp`), trivial baseline 0 (shuffled); for
   high / low the volatility baselines of rule C1 are NOT recomputed here, so their IC is read only RELATIVE to A0.
3. 80 % interval coverage per head and `share_std_below_move` (T-sharp), read relative to A0.
4. Training cost: wall time per arm, steps per epoch.

## Accept rule (fixed now)

An arm X is **better than Y** if BOTH:
(a) its test raw loss is lower than Y's in at least 7 of the 8 fold x seed pairs, AND the mean over the 4 folds of
the paired difference (seeds averaged per fold) has a 95 % t-interval (3 df) below 0;
(b) its high and low rank IC (mean over folds and seeds) are not below Y's by more than twice Y's seed spread
(mean over folds of |seed 0 - seed 1|).
Comparisons that are read: A1 vs A0, A2 vs A0 (does it help?), A1 vs C60, A2 vs C20 (is it the grouping or the batch
size?). Four comparisons: a single pass of the rule is reported as "a look", and nothing becomes a default from this
study (a default change needs the owner and a GPU walk-forward on the real artifact).

## What would refute "grouping helps"

A1 / A2 not better than A0, or better than A0 but not better than their size controls (then the gain is the batch
size, not the date grouping).

## If no arm wins

The failure reading of PHILOSOPHY section 3 is written in the results note from the saved test tables (per-fold
error, per-head, per-coin), with one named cause and two next hypotheses.
