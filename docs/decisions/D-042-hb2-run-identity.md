# D-042 — Hardening 2026-11b, package HB2: run identity and resume safety (owner 2026-11-21)

Date: 2026-11-22
Owner decision: owner message 2026-11-21: "apply the improvement suggestions, fix the unclear places and launch a hardening campaign for the project; the goal is better structure and more clarity" (plan: `docs/hardening/PLAN-2026-11b.md`, package HB2); owner message 2026-11-22: "read the plan, then continue the work"
Why: the audit of 2026-11-21 (`docs/hardening/audit-2026-11b/C_workflow_jobs_colab.md`, re-read against the code in this session) found that a resumed or restored run folder accepted another configuration: the identity hashed no split setting (C-03: a smoke `max_rows` run and the full run, another coin set or `["4h","1h"]` vs `["1h","4h"]` had one identity), no selection metric (C-04: a `best.pt` chosen by another criterion; a gate report of `val_raw_loss` accepted for `val_rank_ic_reg`) and no staged settings (C-05: a flat and a staged run in one folder); a rolling summary read every `fold_row.json` under `out` with no check of the study (C-06); a rolling job silently ignored `split.train_end` / `val_share` (C-11); and four of six job entries ignored `--max-minutes`, so the runner SIGKILLed them with no manifest (C-07), the two that read it copied the formula (C-22). Bottleneck B-091.

## What changes behaviour
- Run identity (C-03, C-04, C-05): `workflow/run_common.identity_context` hashes into the identity's `settings_hash`
  every `SplitSettings` field, `data.input_timeframes`, `train.selection_metric` and `train.staged` that is NOT at its
  default; a run on the defaults passes an empty context and keeps its identity byte-equal (the pinned baseline
  fingerprint in `tests/pins/trainer_config.json` is unchanged, B-091). A run with any of them set gets a new
  `settings_hash`: a checkpoint written before this PR is refused on resume (it was already refused across commits,
  `check_resume` compares `code_sha`) and by `run.restore`; to re-read such a finished run on the new code pass
  `restore(run, allow_changes=("code_sha", "settings_hash"))` deliberately, or use the old commit. Nearly every panel and
  rolling job of `docs/jobs` sets `split.top_coins`, `cross_section`, `max_rows` or a selection metric (the panel
  entry always runs whole timestamps), so their old checkpoints are in that case.
- Gate key (C-04): `configuration_key` adds the selection metric and the staged settings (not the data selection:
  the gates use their own data); a gate report written before this PR for such a configuration no longer matches.
- Staged resume checks the identity before it reads the staged part of the checkpoint (a flat checkpoint failed with
  a KeyError).
- Rolling (C-06, C-11): every `fold_row.json` and `rolling_summary.json` holds `study_hash` (fold geometry without the
  share fields `fold_ids` / `seeds`, the run settings without the per-process ones, the data fingerprint); a share of
  a study that finds a row of another study (or one written before this PR) raises `StudyMismatchError`. The fold
  identity gets a second parent hash, the fold-shaping geometry (mode, train span, embargo, inner `val_share`):
  rolling checkpoints of before are refused on resume. `run_rolling` refuses `split.train_end` / `val_end` /
  `train_share` / `val_share` / `embargo_hours` away from their defaults (no job of `docs/jobs` sets them).
- Deadline (C-07, C-22): `colab/contract.py::deadline` (budget less min(`RESERVE_SECONDS` 600, `RESERVE_SHARE` 0.2
  x budget), the formula `baseline` / `panel` had) is used by every training entry; `rolling`, `staged` and `gates`
  now stop before the limit: a rolling fold x seed or a staged arm does not start after the deadline and its training
  stops at an epoch / evaluation border (a cut fold writes no row, the call returns `finished: False`; the staged
  study writes `finished: false` and the verdict "cannot judge"); a gate run starts no env x seed after it and its
  report says `finished: false` and does not pass. Unchanged when the job fits its limit. `panel_study` names its
  start margin `START_SHARE = 0.25` and its arm rows use `run.training_finished`, the one cut test.
- The new parameters below are `None` / `True` = the behaviour before (no context, no deadline); the module constant
  `IDENTITY_CONFIGURATION` names the two training settings that join the identity and the gate key.

- `nig_timenet_torch.colab.identity.identity_for_run(run_context)`: `<new>` -> `None`
- `nig_timenet_torch.trainer.identity.build_run_identity(run_context)`: `<new>` -> `None`
- `nig_timenet_torch.trainer.identity.configuration_hash(run_context)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.gate_run.run_gates(deadline)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.rolling_run.run_rolling(deadline)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run_common.IDENTITY_CONFIGURATION.train`: `<new>` -> `('selection_metric', 'staged')`
- `nig_timenet_torch.workflow.run_common.identity_context(selection)`: `<new>` -> `True`
- `nig_timenet_torch.workflow.staged_study.run_arm(deadline)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.staged_study.run_staged_study(deadline)`: `<new>` -> `None`
