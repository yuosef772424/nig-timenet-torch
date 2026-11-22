# D-048 — Hardening 2026-11b, package HB6b: workflow structure (owner 2026-11-21)

Date: 2026-11-22
Owner decision: owner message 2026-11-21: "apply the improvement suggestions, fix the unclear places and launch a hardening campaign for the project; the goal is better structure and more clarity" (plan: `docs/hardening/PLAN-2026-11b.md`, package HB6); owner message 2026-11-22: "continue the work, preferably with Sonnet agents for each task"
Why: the audit `docs/hardening/audit-2026-11b/C_workflow_jobs_colab.md` rows C-15 (five hand-kept lists of which training fields are per run and which are configuration, already disagreeing), C-18 (the arm / fold loops repeating one sequence), C-19 (five builders of the coin of each row), C-20 (the model-overrides merge rule written four times) and C-25 (`workflow/run.py` holding eight jobs in 454 lines).

## What changes behaviour
- No output changes: the trainer-config / identity, model-assembly and baseline pins are unchanged; the JSON files are
  written byte-equal (`run_common.write_json` replaces the ten copies of the same `json.dumps` call).
- `workflow/settings_roles.py::TRAIN_FIELD_ROLES` tags every `TrainSettings` field; every former list (resumable
  fields, the gate run's per-run keys, the baseline arm's exempt fields, the gates job's non-configuration keys, the
  study settings' own keys, the rolling study hash's skip list) is derived from it and a test asserts each equals the
  old one; the deliberate disagreements are written in its card.
- `workflow/run.py` is split into `run_state`, `run_train`, `run_restore`, `run_eval`, `run_summary`; `workflow.run`
  re-exports every step, so `from nig_timenet_torch.workflow import run as steps` callers are unchanged; the moved
  parameter defaults below now live in those modules. New `steps.fit` (train or train_staged by `train.staged`).
- `data_io/asset_index.py` (`asset_ids`, `row_coins`; searchsorted, no full-dataset array) replaces the five builders;
  `run_common.excluded_rows` became `excluded_ids`. `settings.model_for` is the one merge rule (the panel study's arm
  merge over the run's own model is a different rule and stays).

- `nig_timenet_torch.workflow.run.Run(extra)`: `{}` -> `<removed>`
- `nig_timenet_torch.workflow.run.Run(fit_settings)`: `None` -> `<removed>`
- `nig_timenet_torch.workflow.run.Run(fold)`: `0` -> `<removed>`
- `nig_timenet_torch.workflow.run.Run(ident)`: `None` -> `<removed>`
- `nig_timenet_torch.workflow.run.Run(ms)`: `None` -> `<removed>`
- `nig_timenet_torch.workflow.run.Run(plan)`: `None` -> `<removed>`
- `nig_timenet_torch.workflow.run.Run(result)`: `None` -> `<removed>`
- `nig_timenet_torch.workflow.run.Run(staged)`: `None` -> `<removed>`
- `nig_timenet_torch.workflow.run.Run(trainer)`: `None` -> `<removed>`
- `nig_timenet_torch.workflow.run.Run(x)`: `{}` -> `<removed>`
- `nig_timenet_torch.workflow.run.Run(y)`: `{}` -> `<removed>`
- `nig_timenet_torch.workflow.run.restore(allow_changes)`: `('code_sha')` -> `<removed>`
- `nig_timenet_torch.workflow.run.restore(weights)`: `'best'` -> `<removed>`
- `nig_timenet_torch.workflow.run.train(batch_monitor)`: `None` -> `<removed>`
- `nig_timenet_torch.workflow.run.train(deadline)`: `None` -> `<removed>`
- `nig_timenet_torch.workflow.run.train(on_epoch)`: `None` -> `<removed>`
- `nig_timenet_torch.workflow.run.train(on_interrupt)`: `None` -> `<removed>`
- `nig_timenet_torch.workflow.run.train_staged(deadline)`: `None` -> `<removed>`
- `nig_timenet_torch.workflow.run.train_staged(on_eval)`: `None` -> `<removed>`
- `nig_timenet_torch.workflow.run.train_staged(on_interrupt)`: `None` -> `<removed>`
- `nig_timenet_torch.workflow.run.train_staged(stages)`: `None` -> `<removed>`
- `nig_timenet_torch.workflow.run.write_summary(extra)`: `None` -> `<removed>`
- `nig_timenet_torch.workflow.run_restore.restore(allow_changes)`: `<new>` -> `('code_sha')`
- `nig_timenet_torch.workflow.run_restore.restore(weights)`: `<new>` -> `'best'`
- `nig_timenet_torch.workflow.run_state.Run(extra)`: `<new>` -> `{}`
- `nig_timenet_torch.workflow.run_state.Run(fit_settings)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run_state.Run(fold)`: `<new>` -> `0`
- `nig_timenet_torch.workflow.run_state.Run(ident)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run_state.Run(ms)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run_state.Run(plan)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run_state.Run(result)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run_state.Run(staged)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run_state.Run(trainer)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run_state.Run(x)`: `<new>` -> `{}`
- `nig_timenet_torch.workflow.run_state.Run(y)`: `<new>` -> `{}`
- `nig_timenet_torch.workflow.run_summary.flat_scalars(prefix)`: `<new>` -> `''`
- `nig_timenet_torch.workflow.run_summary.write_summary(extra)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run_train.fit(deadline)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run_train.fit(on_epoch)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run_train.train(batch_monitor)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run_train.train(deadline)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run_train.train(on_epoch)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run_train.train(on_interrupt)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run_train.train_staged(deadline)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run_train.train_staged(on_eval)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run_train.train_staged(on_interrupt)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run_train.train_staged(stages)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.settings.model_for(overrides)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.settings.model_for(preset)`: `<new>` -> `True`
