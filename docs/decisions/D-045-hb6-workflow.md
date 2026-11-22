# D-045 — Hardening 2026-11b, package HB6a: workflow corrections and dead code (owner 2026-11-21)

Date: 2026-11-22
Owner decision: owner message 2026-11-21: "apply the improvement suggestions, fix the unclear places and launch a hardening campaign for the project; the goal is better structure and more clarity" (plan: `docs/hardening/PLAN-2026-11b.md`, package HB6); owner message 2026-11-22: "continue the work, preferably with Sonnet agents for each task"
Why: the audit `docs/hardening/audit-2026-11b/C_workflow_jobs_colab.md` (rows C-09, C-12, C-14, C-21, C-23, C-24, C-26..C-31, C-33, re-verified by the agent against the code) found `FitResult.stopped_early` carrying four meanings, "best-epoch weights" printed when the model held other weights, runner steps that let most exceptions escape although the card says they never raise, seven copies of the commit lookup, gates that train a cross-section configuration on flat batches, private reach-ins between modules, dead modules (`run_rolling_study`, `staleness.py`, the `runs` command of another branch), missing cards, the repo / Drive paths defined in several places and two default-override dicts that set the same keys.

## What changes behaviour
- `FitResult` has `finished` (every epoch ran or early stopping ended it); `stopped_early` means early-stop patience
  only. In `baseline_summary.json` the values change for staged runs (`stopped_early` was "finished", now False),
  restored runs (`stopped_early` / `finished` now real) and interrupted runs (`stopped_early` False, `finished`
  False); ordinary runs are unchanged. `run.training_finished` reads `finished`.
- The log says which weights the model holds ("best-epoch" only when `restored_best`).
- `colab/runner.py` ends every step error (not only bootstrap / device / OS errors) as a recorded FAIL bundle with the
  traceback in `run_log.txt`; KeyboardInterrupt / SystemExit still propagate.
- One `colab.contract.code_sha()`: a job entry without `NIG_REPO_DIR` raises a clear error (`gates` returned "").
- `run_gates` / `GatesConfig` refuse a configuration that needs whole timestamps (split.cross_section, coin_channel,
  a cross-section loss, cross-coin layers, cross inputs, cross_center): the gates run on flat batches.
- Removed: `workflow/staleness.py`, `workflow.rolling.run_rolling_study`, the `runs` command and its script.
- Same values, one place: `colab/paths.py` (repo, token file, artifact, work dir, Drive runs); the default model is
  `workflow.settings.DEFAULT_MODEL_OVERRIDES` (one dict; `RUN_DEFAULT_SWITCHES` and `MODEL_FIRST_FIXES` merged, the
  merged model equal key for key and in order, test in `tests/model/test_scalers.py`).

- `nig_timenet_torch.colab.__main__.DEFAULT_REPO`: `'yuosef772424/nig-timenet-torch'` -> `<removed>`
- `nig_timenet_torch.colab.__main__.DEFAULT_TOKEN_PATH`: `'/content/drive/MyDrive/crypto/secrets/github_token.txt'` -> `<removed>`
- `nig_timenet_torch.colab.runner.DEFAULT_DRIVE_ROOT`: `'/content/drive/MyDrive'` -> `<removed>`
- `nig_timenet_torch.colab.runner.DEFAULT_DRIVE_RUNS`: `'/content/drive/MyDrive/nig_timenet_runs'` -> `<removed>`
- `nig_timenet_torch.colab.runner.DEFAULT_WORKDIR`: `'/content/nig_runner'` -> `<removed>`
- `nig_timenet_torch.colab_ops.recipes.runs_line(code)`: `'/content/nig_uhub/src'` -> `<removed>`
- `nig_timenet_torch.colab_ops.recipes.runs_line(device)`: `'cuda'` -> `<removed>`
- `nig_timenet_torch.colab_ops.recipes.runs_line(script)`: `'/content/run_models.py'` -> `<removed>`
- `nig_timenet_torch.workflow.rolling.run_rolling_study(out_dir)`: `None` -> `<removed>`
- `nig_timenet_torch.workflow.run.flat_scalars(prefix)`: `<new>` -> `''`
- `nig_timenet_torch.workflow.settings.DEFAULT_MODEL_OVERRIDES.bn_warmup`: `<new>` -> `True`
- `nig_timenet_torch.workflow.settings.DEFAULT_MODEL_OVERRIDES.input_scaler`: `<new>` -> `'train_std'`
- `nig_timenet_torch.workflow.settings.DEFAULT_MODEL_OVERRIDES.nig_scale_fit`: `<new>` -> `'train_std'`
- `nig_timenet_torch.workflow.settings.DEFAULT_MODEL_OVERRIDES.nig_trunk_grad`: `<new>` -> `0.3`
- `nig_timenet_torch.workflow.settings.DEFAULT_MODEL_OVERRIDES.order_detach_close`: `<new>` -> `True`
- `nig_timenet_torch.workflow.settings.RUN_DEFAULT_SWITCHES.bn_warmup`: `True` -> `<removed>`
- `nig_timenet_torch.workflow.settings.RUN_DEFAULT_SWITCHES.nig_scale_fit`: `'train_std'` -> `<removed>`
- `nig_timenet_torch.workflow.settings.RUN_DEFAULT_SWITCHES.nig_trunk_grad`: `0.3` -> `<removed>`
- `nig_timenet_torch.workflow.settings.RUN_DEFAULT_SWITCHES.order_detach_close`: `True` -> `<removed>`
- `nig_timenet_torch.workflow.staleness.label_permutation_null(label_col)`: `'label'` -> `<removed>`
- `nig_timenet_torch.workflow.staleness.label_permutation_null(n_perm)`: `200` -> `<removed>`
- `nig_timenet_torch.workflow.staleness.label_permutation_null(seed)`: `0` -> `<removed>`
