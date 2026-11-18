# D-034 — Alpha / beta floors of the Drive models, floor-report unit fix, speed arms + lite model, notebook (rolling both models, target_norm, plots, stop-and-keep-weights), time-block batches (owner 2026-11-18)

Date: 2026-11-18
Owner decision: owner messages 2026-11-18 (Arabic): (1) "take some models from Google Drive, check the alpha and beta outputs and why alpha and beta are at their lower bounds in 100 % of the samples, or the exact share"; (2) "launch a Sonnet agent to update the main notebook with the updates made: rolling training for both models, the target-normalisation change, the plot of predictions against the truth and the other additions"; (3) "search for ways to speed up training; try settings and design a model version that trains fast"; (4) "if I stop the model in the middle of training I want to use the last weights for the tests"; (5) "will the plot show like this" (screenshot of the TF price panels with the naive baseline and the lag scan); (6) "each batch = all coins of the same moment, so the gradients are computed over them together, like the panel but for each coin on its own".
Why: `docs/audit/2026-11-18-alpha-beta-speed/NOTES.md`: the health report printed beta 100 % on its floor because the floor was read in head units (fixed, no default involved); alpha is really frozen at 2.0; the notebook did not expose the merged features; a stopped notebook could not reach the tests; the owner asked for the TF price panels and for one-moment batches.

## What changes behaviour
Every line is a NEW default of a NEW option or function; no existing behaviour changes and the default run is unchanged (its identity hash too: `batch_order` / `time_block_bars` are in `FIT_SWITCH_DEFAULTS`, left out of the identity at their defaults):
- `legacy_plots.pred_vs_true` / `top_coins` / `price_panels`: plot tools off the run path; `last_n` 16 and `max_lag` 5 copy the TF figures the owner showed; 3 coins is a report size. The battery table gains a `last_close` column (additive; old tables stay readable; `price_panels` asks for a re-run of the battery on an old table).
- `trainer.groups` batch builders and `TrainSettings` / `FitSettings` `batch_order` "rows" (= `torch.randperm` batches exactly, tested) / `time_block_bars` 1: the experiment arm "time_blocks" (one batch = all rows of a block of consecutive timestamps, blocks shuffled) is off; with it, `cross_section_loss` works without the timestamp channel. `RunSettings(train)` changes only by these two fields.
- `run.train` / `train_staged` `on_interrupt` None = a KeyboardInterrupt propagates as before. The notebook (a runner, not a package default) sets `ON_STOP_WEIGHTS = "last"`: a stopped training keeps the weights of the last finished epoch and the later cells run (owner message 4).
- Not pinned (notebook settings): `ROLLING_MODEL` "single", `TARGET_NORM` "none", `LEGACY_REPORT` True, `PLOT_PRED_VS_TRUE` True, `PLOT_KIND` "both", `PLOT_N_COINS` 3, `PLOT_LAST_N` 16; the new cells only read the run's own outputs (battery tables) and write figures / reports.
- `eval/nig_floor.nig_floors` returns the beta floor in label units (beta_min x label scale^2 x target-norm std^2): a fix of a measurement, not a default; it changes the numbers of the health report's floor table only.

- `nig_timenet_torch.eval.legacy_plots.pred_vs_true(title)`: `<new>` -> `''`
- `nig_timenet_torch.eval.legacy_plots.price_panels(heads)`: `<new>` -> `('high', 'low', 'close')`
- `nig_timenet_torch.eval.legacy_plots.price_panels(last_n)`: `<new>` -> `16`
- `nig_timenet_torch.eval.legacy_plots.price_panels(max_lag)`: `<new>` -> `5`
- `nig_timenet_torch.eval.legacy_plots.price_panels(title)`: `<new>` -> `''`
- `nig_timenet_torch.eval.legacy_plots.top_coins(n)`: `<new>` -> `3`
- `nig_timenet_torch.trainer.groups.block_batches(shuffle)`: `<new>` -> `True`
- `nig_timenet_torch.trainer.groups.row_batches(shuffle)`: `<new>` -> `True`
- `nig_timenet_torch.trainer.groups.time_block_batches(shuffle)`: `<new>` -> `True`
- `nig_timenet_torch.trainer.identity.FIT_SWITCH_DEFAULTS.batch_order`: `<new>` -> `'rows'`
- `nig_timenet_torch.trainer.identity.FIT_SWITCH_DEFAULTS.time_block_bars`: `<new>` -> `1`
- `nig_timenet_torch.trainer.trainer.FitSettings(batch_order)`: `<new>` -> `'rows'`
- `nig_timenet_torch.trainer.trainer.FitSettings(time_block_bars)`: `<new>` -> `1`
- `nig_timenet_torch.workflow.run.train(on_interrupt)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run.train_staged(on_interrupt)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.settings.RunSettings(train)`: `TrainSettings(epochs=5, batch_size=256, seed=0, lr=0.001, weight_decay=0.0001, label_smoothing=0.1, selection_metric='val_raw_loss', early_stop=EarlyStopSettings(patience=15, min_delta=0.0001, smoothing='window', window=3, ema_beta=0.7, restore_best_weights=True), device='auto', trace_representative=False, steps_per_execution=1, grad_clip_norm=None, ema_decay=None, init='torch', weight_decay_groups='all_params', uncertainty_weighting=True, class_loss_weight=1.0, lambda_reg_end=0.05, lambda_calib_end=0.1, beta_nll=0.0, beta_nll_normalize=False, loss_balance='none', kendall_init_logvar=0.0, staged=None, cross_section_loss='none', cross_section_weight=1.0, direction_balance=0.0)` -> `TrainSettings(epochs=5, batch_size=256, seed=0, lr=0.001, weight_decay=0.0001, label_smoothing=0.1, selection_metric='val_raw_loss', early_stop=EarlyStopSettings(patience=15, min_delta=0.0001, smoothing='window', window=3, ema_beta=0.7, restore_best_weights=True), device='auto', trace_representative=False, steps_per_execution=1, grad_clip_norm=None, ema_decay=None, init='torch', weight_decay_groups='all_params', uncertainty_weighting=True, class_loss_weight=1.0, lambda_reg_end=0.05, lambda_calib_end=0.1, beta_nll=0.0, beta_nll_normalize=False, loss_balance='none', kendall_init_logvar=0.0, staged=None, cross_section_loss='none', cross_section_weight=1.0, direction_balance=0.0, batch_order='rows', time_block_bars=1)`
- `nig_timenet_torch.workflow.settings.TrainSettings(batch_order)`: `<new>` -> `'rows'`
- `nig_timenet_torch.workflow.settings.TrainSettings(time_block_bars)`: `<new>` -> `1`
