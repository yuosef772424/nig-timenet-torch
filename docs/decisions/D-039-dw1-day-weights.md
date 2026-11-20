# D-039 — Day-balanced, failure-weighted loss + walk-forward on Colab (owner 2026-11-20)

Date: 2026-11-20
Owner decision: owner message 2026-11-20: "update the loss function so the loss weights are updated per day, so that every day in the training data is balanced, and any day it fails on is penalised more (or in the suitable way) so it tries to generalise on it; then run an experiment on Colab"
Why: the readings of 2026-11-19 / 20 (`docs/results/m3-full-001.md`, `v1rep-i2-001.md`, `ir-001.md`) show the P&L concentrated in a few days and the losses in others (the crash days, the 2023-06-10 alt sell-off); the rows of the busy recent days (more coins) also outweigh the early days in a plain row mean. The new option is off by default (`day_weights` None = every row weighs 1, the run identity unchanged), so no existing run changes.

## What changes behaviour
- `TrainSettings.day_weights` / `FitSettings.day_weights` None = the loss as before (every row weight 1, no
  measurement pass, the checkpoint keys and the run identity unchanged: `FIT_SWITCH_DEFAULTS.day_weights` None).
  `RunSettings(train)` changes only by that new field.
- `DayWeightSettings` (read only when the option is on; `trainer/day_weights.py`): `balance` True = every UTC day
  the same total weight (the owner's "every day balanced"); `strength` 1.0 = the exponent on the robust z-score of
  the day loss (a day 1 MAD-sigma worse weighs e = 2.7 x as much before the cap); `max_ratio` 5.0 = a row weighs at
  most 5 x and at least 1/5 x the mean (a day the model cannot learn does not take over); `momentum` 0.5 = the day
  loss is an EMA of the measurements (the weights do not jump each epoch); `shrink_rows` 20 = a day with few rows is
  pulled toward the overall mean loss (noise of small days); `every` 1 = measured at the start of every epoch from
  epoch 1 (epoch 0 = balance only); `measure_chunk` 8192 = rows per forward pass of the measurement (speed only).

- `nig_timenet_torch.trainer.day_weights.DayWeightSettings(balance)`: `<new>` -> `True`
- `nig_timenet_torch.trainer.day_weights.DayWeightSettings(every)`: `<new>` -> `1`
- `nig_timenet_torch.trainer.day_weights.DayWeightSettings(max_ratio)`: `<new>` -> `5.0`
- `nig_timenet_torch.trainer.day_weights.DayWeightSettings(measure_chunk)`: `<new>` -> `8192`
- `nig_timenet_torch.trainer.day_weights.DayWeightSettings(momentum)`: `<new>` -> `0.5`
- `nig_timenet_torch.trainer.day_weights.DayWeightSettings(shrink_rows)`: `<new>` -> `20`
- `nig_timenet_torch.trainer.day_weights.DayWeightSettings(strength)`: `<new>` -> `1.0`
- `nig_timenet_torch.trainer.identity.FIT_SWITCH_DEFAULTS.day_weights`: `<new>` -> `None`
- `nig_timenet_torch.trainer.trainer.FitSettings(day_weights)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.settings.RunSettings(train)`: `TrainSettings(epochs=5, batch_size=256, seed=0, lr=0.001, weight_decay=0.0001, label_smoothing=0.1, selection_metric='val_raw_loss', early_stop=EarlyStopSettings(patience=15, min_delta=0.0001, smoothing='window', window=3, ema_beta=0.7, restore_best_weights=True), device='auto', trace_representative=False, steps_per_execution=1, grad_clip_norm=None, ema_decay=None, init='torch', weight_decay_groups='all_params', uncertainty_weighting=True, class_loss_weight=1.0, lambda_reg_end=0.05, lambda_calib_end=0.1, beta_nll=0.0, beta_nll_normalize=False, loss_balance='none', kendall_init_logvar=0.0, staged=None, cross_section_loss='none', cross_section_weight=1.0, direction_balance=0.0, batch_by='rows', batch_order='rows', time_block_bars=1)` -> `TrainSettings(epochs=5, batch_size=256, seed=0, lr=0.001, weight_decay=0.0001, label_smoothing=0.1, selection_metric='val_raw_loss', early_stop=EarlyStopSettings(patience=15, min_delta=0.0001, smoothing='window', window=3, ema_beta=0.7, restore_best_weights=True), device='auto', trace_representative=False, steps_per_execution=1, grad_clip_norm=None, ema_decay=None, init='torch', weight_decay_groups='all_params', uncertainty_weighting=True, class_loss_weight=1.0, lambda_reg_end=0.05, lambda_calib_end=0.1, beta_nll=0.0, beta_nll_normalize=False, loss_balance='none', kendall_init_logvar=0.0, staged=None, cross_section_loss='none', cross_section_weight=1.0, direction_balance=0.0, batch_by='rows', batch_order='rows', time_block_bars=1, day_weights=None)`
- `nig_timenet_torch.workflow.settings.TrainSettings(day_weights)`: `<new>` -> `None`
