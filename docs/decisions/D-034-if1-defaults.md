# D-034 — Batch influence during training + feature permutation importance (owner 2026-11-17)

Date: 2026-11-17
Owner decision: owner messages 2026-11-17 (Arabic): (1) "see the effect of each batch of data on the model during training; search for the best methods, choose the best and most efficient; then implement a way to measure the effect of each feature on the model; the results should be reliable when tested". (2) "apply the suggestions and add grouping of the samples by date for the single-coin model: e.g. all samples of one date form one batch, the next likewise, and shuffle at the level of the batches, not the samples".
Why: every line is a NEW option of new diagnostic code (`docs/INFLUENCE.md`); no existing behaviour changes: `Trainer.fit(batch_monitor=None)`, `run.train(batch_monitor=None)` and `make_context(monitor_of=None)` train exactly as before (bitwise test `tests/trainer/test_batch_influence.py::test_the_monitor_leaves_the_training_unchanged`).

## What changes behaviour
- `monitor_of` / `batch_monitor` = None: no monitor, the training is unchanged.
- `batch_by = "rows"` (TrainSettings, FitSettings, `FIT_SWITCH_DEFAULTS`): the batches are shuffled rows as before and the run identity is unchanged at the default (left out of the hash like the D-021 switches); "timestamp" / "day" (owner request 2) = one batch per timestamp / UTC day, the batches shuffled; `RunSettings(train)` changes only by carrying the new field at its default.
- `InfluenceSettings(group_every=0, group_rows=256)`: group-level TracIn (IF1 suggestion) off by default (it costs one backward per group); 256 rows per group gradient when on.
- Report headings (`title` parameters) are no longer pinned defaults (`tests/defaults_pin.py::REPORT_ONLY`, IF1 suggestion): the two pinned before this PR are removed below; they change no number, file or behaviour.
- `InfluenceSettings`: probe of 512 rows (two halves of 256: split-half reliability > 0.9 on the known-answer tests and on E1), exact deltas every step, first-order terms off by default (`first_order_every` 0: they cost a probe backward), batch rows kept for the composition analysis, seed 0.
- `ImportanceSettings`: 5 permutations per group, global permutation (`within_time` on request), 40 time blocks for the block-bootstrap CI, no bar blocks unless asked, clusters at |Spearman| >= 0.9, family error rate 0.05 (Bonferroni), 4096 rows per forward, seed 0.
- Report helpers: rolling-median window 9 steps for the excess delta, 1000 permutations for the group null, 10 top steps, titles and file names of the reports, 50,000 VAL rows for the importance of the entry, the training-row probe on.

- `nig_timenet_torch.doctor.common.make_context(monitor_of)`: `<new>` -> `None`
- `nig_timenet_torch.doctor.feature_importance.ImportanceSettings(alpha)`: `<new>` -> `0.05`
- `nig_timenet_torch.doctor.feature_importance.ImportanceSettings(bar_blocks)`: `<new>` -> `0`
- `nig_timenet_torch.doctor.feature_importance.ImportanceSettings(chunk)`: `<new>` -> `4096`
- `nig_timenet_torch.doctor.feature_importance.ImportanceSettings(cluster_threshold)`: `<new>` -> `0.9`
- `nig_timenet_torch.doctor.feature_importance.ImportanceSettings(n_blocks)`: `<new>` -> `40`
- `nig_timenet_torch.doctor.feature_importance.ImportanceSettings(permutation)`: `<new>` -> `'global'`
- `nig_timenet_torch.doctor.feature_importance.ImportanceSettings(repeats)`: `<new>` -> `5`
- `nig_timenet_torch.doctor.feature_importance.ImportanceSettings(seed)`: `<new>` -> `0`
- `nig_timenet_torch.doctor.feature_importance.correlation_clusters(max_rows)`: `<new>` -> `20000`
- `nig_timenet_torch.doctor.feature_importance.feature_importance(groups)`: `<new>` -> `None`
- `nig_timenet_torch.doctor.feature_importance.feature_importance(names)`: `<new>` -> `None`
- `nig_timenet_torch.doctor.feature_importance.feature_importance(settings)`: `<new>` -> `None`
- `nig_timenet_torch.doctor.feature_importance.render_importance(top)`: `<new>` -> `40`
- `nig_timenet_torch.doctor.feature_importance.write_importance(name)`: `<new>` -> `'feature_importance'`
- `nig_timenet_torch.doctor.influence.batch_reports(n_perm)`: `<new>` -> `1000`
- `nig_timenet_torch.doctor.influence.group_sets(max_groups)`: `<new>` -> `60`
- `nig_timenet_torch.doctor.influence.importance_reports(max_rows)`: `<new>` -> `50000`
- `nig_timenet_torch.doctor.influence.importance_reports(within_time)`: `<new>` -> `False`
- `nig_timenet_torch.doctor.influence.monitor_for(group_by)`: `<new>` -> `None`
- `nig_timenet_torch.doctor.influence.monitor_for(max_groups)`: `<new>` -> `60`
- `nig_timenet_torch.doctor.influence.monitor_for(train_probe)`: `<new>` -> `True`
- `nig_timenet_torch.eval.day_metrics.render(title)`: `''` -> `<removed>`
- `nig_timenet_torch.eval.influence_report.excess_delta(column)`: `<new>` -> `'delta'`
- `nig_timenet_torch.eval.influence_report.excess_delta(window)`: `<new>` -> `9`
- `nig_timenet_torch.eval.influence_report.group_alignment(alpha)`: `<new>` -> `0.05`
- `nig_timenet_torch.eval.influence_report.group_attribution(alpha)`: `<new>` -> `0.05`
- `nig_timenet_torch.eval.influence_report.group_attribution(n_perm)`: `<new>` -> `1000`
- `nig_timenet_torch.eval.influence_report.group_attribution(seed)`: `<new>` -> `0`
- `nig_timenet_torch.eval.influence_report.group_attribution(window)`: `<new>` -> `9`
- `nig_timenet_torch.eval.influence_report.influence_summary(window)`: `<new>` -> `9`
- `nig_timenet_torch.eval.influence_report.render_markdown(alignment)`: `<new>` -> `None`
- `nig_timenet_torch.eval.influence_report.render_markdown(groups)`: `<new>` -> `None`
- `nig_timenet_torch.eval.influence_report.render_markdown(top)`: `<new>` -> `None`
- `nig_timenet_torch.eval.influence_report.top_steps(k)`: `<new>` -> `10`
- `nig_timenet_torch.eval.influence_report.top_steps(row_group)`: `<new>` -> `None`
- `nig_timenet_torch.eval.influence_report.top_steps(window)`: `<new>` -> `9`
- `nig_timenet_torch.eval.influence_report.write_influence_report(gframe)`: `<new>` -> `None`
- `nig_timenet_torch.eval.influence_report.write_influence_report(n_perm)`: `<new>` -> `1000`
- `nig_timenet_torch.eval.influence_report.write_influence_report(row_group)`: `<new>` -> `None`
- `nig_timenet_torch.eval.influence_report.write_influence_report(window)`: `<new>` -> `9`
- `nig_timenet_torch.trainer.batch_influence.BatchInfluence(group_channel)`: `<new>` -> `False`
- `nig_timenet_torch.trainer.batch_influence.BatchInfluence(groups)`: `<new>` -> `None`
- `nig_timenet_torch.trainer.batch_influence.BatchInfluence(settings)`: `<new>` -> `None`
- `nig_timenet_torch.trainer.batch_influence.InfluenceSettings(first_order_every)`: `<new>` -> `0`
- `nig_timenet_torch.trainer.batch_influence.InfluenceSettings(group_every)`: `<new>` -> `0`
- `nig_timenet_torch.trainer.batch_influence.InfluenceSettings(group_rows)`: `<new>` -> `256`
- `nig_timenet_torch.trainer.batch_influence.InfluenceSettings(keep_rows)`: `<new>` -> `True`
- `nig_timenet_torch.trainer.batch_influence.InfluenceSettings(probe_rows)`: `<new>` -> `512`
- `nig_timenet_torch.trainer.batch_influence.InfluenceSettings(seed)`: `<new>` -> `0`
- `nig_timenet_torch.trainer.batch_influence.probe_rows(groups)`: `<new>` -> `None`
- `nig_timenet_torch.trainer.identity.FIT_SWITCH_DEFAULTS.batch_by`: `<new>` -> `'rows'`
- `nig_timenet_torch.trainer.trainer.FitSettings(batch_by)`: `<new>` -> `'rows'`
- `nig_timenet_torch.workflow.panel_report.render(title)`: `''` -> `<removed>`
- `nig_timenet_torch.workflow.run.train(batch_monitor)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.settings.RunSettings(train)`: `TrainSettings(epochs=5, batch_size=256, seed=0, lr=0.001, weight_decay=0.0001, label_smoothing=0.1, selection_metric='val_raw_loss', early_stop=EarlyStopSettings(patience=15, min_delta=0.0001, smoothing='window', window=3, ema_beta=0.7, restore_best_weights=True), device='auto', trace_representative=False, steps_per_execution=1, grad_clip_norm=None, ema_decay=None, init='torch', weight_decay_groups='all_params', uncertainty_weighting=True, class_loss_weight=1.0, lambda_reg_end=0.05, lambda_calib_end=0.1, beta_nll=0.0, beta_nll_normalize=False, loss_balance='none', kendall_init_logvar=0.0, staged=None, cross_section_loss='none', cross_section_weight=1.0, direction_balance=0.0)` -> `TrainSettings(epochs=5, batch_size=256, seed=0, lr=0.001, weight_decay=0.0001, label_smoothing=0.1, selection_metric='val_raw_loss', early_stop=EarlyStopSettings(patience=15, min_delta=0.0001, smoothing='window', window=3, ema_beta=0.7, restore_best_weights=True), device='auto', trace_representative=False, steps_per_execution=1, grad_clip_norm=None, ema_decay=None, init='torch', weight_decay_groups='all_params', uncertainty_weighting=True, class_loss_weight=1.0, lambda_reg_end=0.05, lambda_calib_end=0.1, beta_nll=0.0, beta_nll_normalize=False, loss_balance='none', kendall_init_logvar=0.0, staged=None, cross_section_loss='none', cross_section_weight=1.0, direction_balance=0.0, batch_by='rows')`
- `nig_timenet_torch.workflow.settings.TrainSettings(batch_by)`: `<new>` -> `'rows'`
