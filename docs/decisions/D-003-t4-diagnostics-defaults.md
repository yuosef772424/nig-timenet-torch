# D-003 — T4 diagnostics: new defaults (ROADMAP T4)

Date: 2026-11-03
Owner decision: T4 session prompt (2026-11-03): ROADMAP T4 (diagnostics on the targets, `health_report`); the T1 rule
applies (a new default needs a D-nnn file and a pin via `python -m tests.defaults_pin --add`).
Why: nothing here changes an existing default; every key below is NEW (the `<new>` format of docs/decisions/README.md).
The only one that touches an existing object is `FitSettings.trace`: `False` = `Trainer.fit` behaves and logs exactly
as before (no extra history columns, no extra forward/backward pass); it is also left out of the RunIdentity hash, so
no existing identity or checkpoint changes. The others are report parameters: 10 bins = deciles (ROADMAP T-conf),
`rel_tol` 0.05 = "within 5 % of the floor" (chosen here, exploratory, B-001 evidence only), 200 bootstrap draws and
seed 0 (a fixed seed makes the report reproducible), `probe_max_rows` 2000 (keeps the ridge probe CPU-cheap).

New defaults:

- `nig_timenet_torch.eval.nig_floor.nig_floor_hit_report(rel_tol)`: `<new>` -> `0.05`
- `nig_timenet_torch.eval.target_report.target_report(n_bins)`: `<new>` -> `10`
- `nig_timenet_torch.eval.target_report.target_report(seed)`: `<new>` -> `0`
- `nig_timenet_torch.stats.metrics.confidence_error_gap(n_bins)`: `<new>` -> `10`
- `nig_timenet_torch.stats.metrics.confidence_error_gap(n_boot)`: `<new>` -> `200`
- `nig_timenet_torch.stats.metrics.confidence_error_gap(seed)`: `<new>` -> `0`
- `nig_timenet_torch.stats.metrics.confidence_error_spearman(n_bins)`: `<new>` -> `10`
- `nig_timenet_torch.trainer.trainer.FitSettings(trace)`: `<new>` -> `False`
- `nig_timenet_torch.workflow.health_report.HealthRun(probe_max_rows)`: `<new>` -> `2000`
