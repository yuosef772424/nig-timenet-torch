# D-034 — Batch influence during training + feature permutation importance (owner 2026-11-17)

Date: 2026-11-17
Owner decision: owner message 2026-11-17 (Arabic): "see the effect of each batch of data on the model during training; search for the best methods, choose the best and most efficient; then implement a way to measure the effect of each feature on the model; the results should be reliable when tested".
Why: every line is a NEW option of new diagnostic code (`docs/INFLUENCE.md`); no existing behaviour changes: `Trainer.fit(batch_monitor=None)`, `run.train(batch_monitor=None)` and `make_context(monitor_of=None)` train exactly as before (bitwise test `tests/trainer/test_batch_influence.py::test_the_monitor_leaves_the_training_unchanged`).

## What changes behaviour
- `monitor_of` / `batch_monitor` = None: no monitor, the training is unchanged.
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
- `nig_timenet_torch.doctor.influence.importance_reports(max_rows)`: `<new>` -> `50000`
- `nig_timenet_torch.doctor.influence.importance_reports(within_time)`: `<new>` -> `False`
- `nig_timenet_torch.doctor.influence.monitor_for(train_probe)`: `<new>` -> `True`
- `nig_timenet_torch.eval.influence_report.excess_delta(column)`: `<new>` -> `'delta'`
- `nig_timenet_torch.eval.influence_report.excess_delta(window)`: `<new>` -> `9`
- `nig_timenet_torch.eval.influence_report.group_attribution(alpha)`: `<new>` -> `0.05`
- `nig_timenet_torch.eval.influence_report.group_attribution(n_perm)`: `<new>` -> `1000`
- `nig_timenet_torch.eval.influence_report.group_attribution(seed)`: `<new>` -> `0`
- `nig_timenet_torch.eval.influence_report.group_attribution(window)`: `<new>` -> `9`
- `nig_timenet_torch.eval.influence_report.influence_summary(window)`: `<new>` -> `9`
- `nig_timenet_torch.eval.influence_report.render_markdown(groups)`: `<new>` -> `None`
- `nig_timenet_torch.eval.influence_report.render_markdown(top)`: `<new>` -> `None`
- `nig_timenet_torch.eval.influence_report.top_steps(k)`: `<new>` -> `10`
- `nig_timenet_torch.eval.influence_report.top_steps(row_group)`: `<new>` -> `None`
- `nig_timenet_torch.eval.influence_report.top_steps(window)`: `<new>` -> `9`
- `nig_timenet_torch.eval.influence_report.write_influence_report(n_perm)`: `<new>` -> `1000`
- `nig_timenet_torch.eval.influence_report.write_influence_report(row_group)`: `<new>` -> `None`
- `nig_timenet_torch.eval.influence_report.write_influence_report(window)`: `<new>` -> `9`
- `nig_timenet_torch.trainer.batch_influence.BatchInfluence(group_channel)`: `<new>` -> `False`
- `nig_timenet_torch.trainer.batch_influence.BatchInfluence(settings)`: `<new>` -> `None`
- `nig_timenet_torch.trainer.batch_influence.InfluenceSettings(first_order_every)`: `<new>` -> `0`
- `nig_timenet_torch.trainer.batch_influence.InfluenceSettings(keep_rows)`: `<new>` -> `True`
- `nig_timenet_torch.trainer.batch_influence.InfluenceSettings(probe_rows)`: `<new>` -> `512`
- `nig_timenet_torch.trainer.batch_influence.InfluenceSettings(seed)`: `<new>` -> `0`
- `nig_timenet_torch.trainer.batch_influence.probe_rows(groups)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run.train(batch_monitor)`: `<new>` -> `None`
- `nig_timenet_torch.eval.day_metrics.render(title)`: `''` -> `<removed>`
- `nig_timenet_torch.workflow.panel_report.render(title)`: `''` -> `<removed>`
