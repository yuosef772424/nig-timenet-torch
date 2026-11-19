# more-data-flat-001: does more training data help? All train rows against 400k on one flat split (owner 2026-11-19)

Pre-registered: `docs/audit/2026-11-18-panel-std/PREREGISTERED.md` addendum 5. Jobs `panel-std-order-full-001` and
`panel-std-order-400k-001` (commit `c0f1219`, data `c572b1b36960`, L4), arm `relative_rank_loss_logret_std_label_order`
(relative inputs + rank term + std targets + the order on prices), the artifact's default split (train to 2026-02-24,
val to 2026-06-15, test to 2026-09), the 100 longest-listed coins, 8 epochs, patience 3, seeds 0 and 1. The same val
(54717) and test (49791) rows in both; train 1113463 rows against 399945. Exploratory (rule C5): one split, about 100
test days, a look.

Failure reading: not applicable (a data-size look on one split; the arm's own reading is in
`docs/results/panel-rolling-std-001.md` and its walk-forward `panel-rolling-std-order-001`).

| seed 0 / 1 | all rows (1.11 M) | 400k |
|---|---|---|
| close rank IC on TEST | 0.1050 / 0.1073 | 0.1042 / 0.0981 |
| high rank IC | 0.316 / 0.319 | 0.318 / 0.318 |
| low rank IC | 0.424 / 0.423 | 0.419 / 0.421 |
| relative rank IC (pooled day frame) | +0.105 / +0.107 | +0.104 / +0.098 |
| best epoch / epochs run | 1 / 4 (early stop) and 7 / 8 | 8 / 8 and 6 / 8 |
| metric 2 (neutral book) net mean day | -0.099 % / -0.093 % | -0.069 % / -0.063 % |
| metric 1 winning days (CI) | 62.6 % [50, 74] / 43.4 % [34, 54] | 51.5 % [43, 60] / 58.6 % [50, 68] |
| seconds per epoch (six jobs on one L4) | 171 / 181 | 64 / 72 |

## Reading
1. **Not accepted, not refuted.** The close rank IC is higher with all rows on both seeds (+0.0008 and +0.0092, mean
   +0.005), but on seed 0 the gain is below the capped job's seed spread (0.006), so the pre-registered "more data
   helps" rule is not met; the refutation (not higher on either seed) is not met either. High / low: no loss (low
   +0.003 .. +0.005).
2. **No gain in the books.** The neutral book's net is worse with all rows (-0.099 / -0.093 % against -0.069 /
   -0.063 % a day) and the absolute book swings between seeds (62.6 % vs 43.4 % winning days): with about 100 test days
   the day metrics cannot separate the two (their CIs overlap widely).
3. **More data per epoch reaches its best sooner:** seed 0 peaked at epoch 1 and stopped at 4; at 2.7x the time per
   epoch, the extra 700k older rows buy little. Together with rolling-memorisation-001 (the newest months are left out
   of training, the oldest era is a large share), the next data lever is RECENCY (train on the last months, or weight
   them), not volume.

## Next hypotheses
1. Train on train + validation (the newest months) with the chosen epoch count; refuted if the close IC on the same
   test rows does not rise beyond the seed spread.
2. A time-decay sample weight (recent rows count more) at equal row count; refuted likewise.
