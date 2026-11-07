# D-013: run steps (flat single split) and the profit battery in the run (package I3)

Date: 2026-11-07
Owner decision: hardening package I3 of `docs/hardening/PLAN.md` (finding HF-23, bottleneck B-035) plus the planner
additions of the same session (owner goal: profit). No value of the baseline job changes: `jobs/baseline.py` is now a thin
entry over `workflow/run.py` and its summary / test report equal the pre-refactor outputs (`tests/jobs/test_baseline_pin.py`).

- Moved, same values: `jobs.baseline.ANTI_MEMORIZATION_CONFIG` (an import, now seen in `workflow.settings`), `evaluate_test(chunk)` ->
  `EvalSettings.chunk`, `trainer_config(label_smoothing)` -> `workflow.flat_run.trainer_config` (label smoothing is now a required argument,
  its default lives in `TrainSettings.label_smoothing`).
- `RunSettings` is the new strict settings tree; its defaults are NEW (the job overrides device, seed, epochs, batch size, run folder,
  artifact, split and health rows from its config, everything else keeps the old literal: lr 1e-3, weight decay 1e-4, label smoothing 0.1,
  80/10/10 time quantiles, chunk 20000, anti-memorization preset). `epochs=5` and `batch_size=256` are notebook defaults only.
- `RunSettings.rolling` is removed (it was read by nothing; rolling folds are package I4 and use `RollingSettings` directly).

- Battery behaviour change (no default value changes, so no key line): every battery section charges `cost` PER SIDE (a trade pays
  `2 x 0.0008`; the older sections paid it once, the market-neutral one `2 x cost` instead of `4 x cost`), and the direction of
  every section is the close class head `p_dir` (sanity, shock, weekly, per-coin and persistence used `sign(mu)` before).
  Ranking by `|mu|` stays. Fixture numbers are in the PR description.

- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.class_head_hidden`: `32` -> `<removed>`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.d_model`: `64` -> `<removed>`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.dropout`: `0.25` -> `<removed>`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.encoder`: `'gru'` -> `<removed>`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.feature_dropout`: `0.1` -> `<removed>`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.head_hidden`: `64` -> `<removed>`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.input_clip`: `4.0` -> `<removed>`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.input_noise_std`: `0.1` -> `<removed>`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.level_norm`: `'batch'` -> `<removed>`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.level_passthrough`: `True` -> `<removed>`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.linear_path_l2`: `0.001` -> `<removed>`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.num_layers`: `2` -> `<removed>`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.stats_mode`: `'symlog'` -> `<removed>`
- `nig_timenet_torch.jobs.baseline.evaluate_test(chunk)`: `20000` -> `<removed>`
- `nig_timenet_torch.jobs.baseline.trainer_config(label_smoothing)`: `0.1` -> `<removed>`
- `nig_timenet_torch.workflow.settings.RunSettings(rolling)`: `None` -> `<removed>`
- `nig_timenet_torch.jobs.baseline.run_settings(code_sha)`: `<new>` -> `''`
- `nig_timenet_torch.workflow.run.Run(extra)`: `<new>` -> `{}`
- `nig_timenet_torch.workflow.run.Run(fit_settings)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run.Run(ident)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run.Run(ms)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run.Run(out)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run.Run(result)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run.Run(trainer)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run.Run(x)`: `<new>` -> `{}`
- `nig_timenet_torch.workflow.run.Run(y)`: `<new>` -> `{}`
- `nig_timenet_torch.workflow.run.prepare(out)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run.run_all(deadline)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run.run_all(on_epoch)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run.run_all(out)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run.train(deadline)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run.train(on_epoch)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run.write_summary(extra)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.class_head_hidden`: `<new>` -> `32`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.d_model`: `<new>` -> `64`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.dropout`: `<new>` -> `0.25`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.encoder`: `<new>` -> `'gru'`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.feature_dropout`: `<new>` -> `0.1`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.head_hidden`: `<new>` -> `64`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.input_clip`: `<new>` -> `4.0`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.input_noise_std`: `<new>` -> `0.1`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.level_norm`: `<new>` -> `'batch'`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.level_passthrough`: `<new>` -> `True`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.linear_path_l2`: `<new>` -> `0.001`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.num_layers`: `<new>` -> `2`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.stats_mode`: `<new>` -> `'symlog'`
- `nig_timenet_torch.workflow.settings.EvalSettings(chunk)`: `<new>` -> `20000`
- `nig_timenet_torch.workflow.settings.EvalSettings(health_max_rows)`: `<new>` -> `100000`
- `nig_timenet_torch.workflow.settings.RunSettings(code_sha)`: `<new>` -> `''`
- `nig_timenet_torch.workflow.settings.RunSettings(data)`: `<new>` -> `LoaderSettings(directory='.', name='preprocessing_output', format='auto', mmap=True, local_dir='auto', allow_unchecked=False, verify_hashes=False)`
- `nig_timenet_torch.workflow.settings.RunSettings(model)`: `<new>` -> `ModelSettings(architecture='nig_timenet_v2', overrides={'d_model': 64, 'num_layers': 2, 'head_hidden': 64, 'class_head_hidden': 32, 'dropout': 0.25, 'input_clip': 4.0, 'stats_mode': 'symlog', 'input_noise_std': 0.1, 'feature_dropout': 0.1, 'linear_path_l2': 0.001, 'level_passthrough': True, 'level_norm': 'batch', 'encoder': 'gru'})`
- `nig_timenet_torch.workflow.settings.RunSettings(split)`: `<new>` -> `SplitSettings(train_end='', val_end='', train_share=0.8, val_share=0.1, embargo_hours=0.0, max_rows=None)`
- `nig_timenet_torch.workflow.settings.RunSettings(train)`: `<new>` -> `TrainSettings(epochs=5, batch_size=256, seed=0, lr=0.001, weight_decay=0.0001, label_smoothing=0.1, selection_metric='val_raw_loss', early_stop=EarlyStopSettings(patience=15, min_delta=0.0001, smoothing='window', window=3, ema_beta=0.7, restore_best_weights=True), device='auto')`
- `nig_timenet_torch.workflow.settings.SplitSettings(embargo_hours)`: `<new>` -> `0.0`
- `nig_timenet_torch.workflow.settings.SplitSettings(max_rows)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.settings.SplitSettings(train_end)`: `<new>` -> `''`
- `nig_timenet_torch.workflow.settings.SplitSettings(train_share)`: `<new>` -> `0.8`
- `nig_timenet_torch.workflow.settings.SplitSettings(val_end)`: `<new>` -> `''`
- `nig_timenet_torch.workflow.settings.SplitSettings(val_share)`: `<new>` -> `0.1`
- `nig_timenet_torch.workflow.settings.TrainSettings(batch_size)`: `<new>` -> `256`
- `nig_timenet_torch.workflow.settings.TrainSettings(device)`: `<new>` -> `'auto'`
- `nig_timenet_torch.workflow.settings.TrainSettings(early_stop)`: `<new>` -> `EarlyStopSettings(patience=15, min_delta=0.0001, smoothing='window', window=3, ema_beta=0.7, restore_best_weights=True)`
- `nig_timenet_torch.workflow.settings.TrainSettings(epochs)`: `<new>` -> `5`
- `nig_timenet_torch.workflow.settings.TrainSettings(label_smoothing)`: `<new>` -> `0.1`
- `nig_timenet_torch.workflow.settings.TrainSettings(lr)`: `<new>` -> `0.001`
- `nig_timenet_torch.workflow.settings.TrainSettings(seed)`: `<new>` -> `0`
- `nig_timenet_torch.workflow.settings.TrainSettings(selection_metric)`: `<new>` -> `'val_raw_loss'`
- `nig_timenet_torch.workflow.settings.TrainSettings(weight_decay)`: `<new>` -> `0.0001`
- `nig_timenet_torch.workflow.health_report.HealthRun(probe_points)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.settings.BatterySettings(cost_per_side)`: `<new>` -> `0.0008`
- `nig_timenet_torch.workflow.settings.BatterySettings(enabled)`: `<new>` -> `True`
- `nig_timenet_torch.workflow.settings.BatterySettings(n_variants)`: `<new>` -> `1`
- `nig_timenet_torch.workflow.settings.EvalSettings(battery)`: `<new>` -> `BatterySettings(enabled=True, cost_per_side=0.0008, n_variants=1)`
- `nig_timenet_torch.workflow.settings.RunSettings(evaluation)`: `<new>` -> `EvalSettings(chunk=20000, health_max_rows=100000)`
- `nig_timenet_torch.workflow.settings.RunSettings(evaluation)`: `EvalSettings(chunk=20000, health_max_rows=100000)` -> `EvalSettings(chunk=20000, health_max_rows=100000, battery=BatterySettings(enabled=True, cost_per_side=0.0008, n_variants=1))`
- `nig_timenet_torch.workflow.battery_run.battery_from_folder(cost)`: `<new>` -> `0.0008`
- `nig_timenet_torch.workflow.battery_run.battery_from_folder(n_variants)`: `<new>` -> `1`
- `nig_timenet_torch.workflow.battery_run.battery_from_folder(out_dir)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.battery_run.run_tables(cost)`: `<new>` -> `0.0008`
- `nig_timenet_torch.workflow.battery_run.run_tables(n_variants)`: `<new>` -> `1`
- `nig_timenet_torch.workflow.battery_run.run_tables(out_dir)`: `<new>` -> `None`
