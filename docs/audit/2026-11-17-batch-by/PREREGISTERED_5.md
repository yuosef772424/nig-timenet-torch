# PREREGISTERED (5) — H5: a lower learning rate (1e-5), then A0 vs N again (owner 2026-11-18)

Written before any number of this study was read. Owner's request (Arabic, 2026-11-18): "with a lower learning rate,
e.g. 1e-5". Exploratory (rule C5).

## Hypothesis (H5 of `docs/results/no-instance-norm-001.md`)

At lr 1e-3 every model peaks at epoch 1 and then memorises (training loss falls, validation loss rises), so an input
change (H: range terms; N: no per-window normalisation) gets one useful epoch and closes only ~15 % of the gap to the
model-free bound. **With lr 1e-5 the best epoch moves past 1, and with more useful epochs N reaches the bound.**

## Design (4 trainings: 2 versions x 2 seeds)

Same data and split as PREREGISTERED_3 / _4 (60 coins, train 2023 with the last 15 % as inner validation, test 2024).
Versions **A0** (run default) and **N** (`use_instance_norm = False`), both with **lr 1e-5** (AdamW, weight decay as
the run default) and **20 epochs** (at 1e-5 four epochs are about 850 steps: too few to learn anything); best epoch by
`val_raw_loss`, early stopping as the run default (patience 15, smoothed over a window of 3). Seeds 0 and 1, battery
off, Colab L4, one commit. Gates of rule C6 not run (deviation, as in studies 3-4).

## Accept rules (read in this order)

1. **Memorisation stops**: the best epoch is > 1 in all 4 models. If not, H5 is refuted at once.
2. **N reaches the bound and beats A0** (the rules of PREREGISTERED_4 at lr 1e-5): (a) N's rank IC by timestamp
   (mean of the 2 seeds) >= the bound (the better of the 6-bar range and the ridge on the HAR features, fitted on 2023)
   in >= 3 of 4 quarters of 2024 on both heads; (b) N > A0 in >= 3 of 4 quarters on both heads and on every seed over
   the whole year.
H5 supported only if 1 AND 2 hold. Read, not judged: A0 and N at lr 1e-5 against the same versions at lr 1e-3
(study 4, `batch_by_004`), the gap closed, the train-validation curves, the test raw loss, the close head's bias.
