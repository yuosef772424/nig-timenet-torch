# D-021: training switches, model taps, versioned Colab results

Date: 2026-11-09
Owner decision: owner message of 2026-11-09 (Arabic, translated): "start; and improve the protection item so that a
re-run writes a separate version (v1, v2) instead of overwriting; after that fix every error found, each behind a
switch so it can be turned off later; then train the model, test it intensively, locate the errors precisely and fix
them; different architectures may be tried once the structure is stable; add inspection-only layers / a dict of every
layer's data that training ignores and the tests read."
Why: the gate run of G2 (D-019) found B-060 / B-061 / B-062 and the TR1 list (B-007, B-018) had no switches; the
results branch refused a second run of a job id (B-047) and overwrote with `rerun`.

## What changes behaviour
- Nothing at the defaults (every new default reproduces the run before D-021; trainer config, optimizer groups and the
  run identity are byte-equal, tests/trainer/test_switches.py, tests/pins/trainer_config.json).
- `colab/job.py`: new optional key `on_existing` (`refuse` default = today; `new_version`: a finished job runs as
  `<job_id>-v2`, `-v3`, ... with its own bundle folder and its own `config.run_dir`; `base_job_id` recorded).
- `health_report`: new section `layers` (per-leaf-layer summary from the inspection taps) in the JSON and one line in
  the markdown (report-only).
- `trainer.switches.EmaWeights` uses the TF warm-up `d_n = min(decay, (1 + n) / (10 + n))` (without it an EMA of
  0.999 over a few hundred steps stayed at the init weights on E1).

## New switches (TrainSettings -> FitSettings / trainer config)
`grad_clip_norm`, `ema_decay`, `init` (`torch` | `keras`), `weight_decay_groups` (`all_params` | `exclude_norm_bias`),
`uncertainty_weighting`, `class_loss_weight`, `lambda_reg_end`, `lambda_calib_end`. Model switches stay in
`ModelSettings.overrides` (dropout, beta_min, input_clip, enforce_order, ...).

## Changed default reprs (new fields at their defaults; same behaviour)
- `nig_timenet_torch.workflow.settings.RunSettings(train)`: `TrainSettings(epochs=5, batch_size=256, seed=0, lr=0.001, weight_decay=0.0001, label_smoothing=0.1, selection_metric='val_raw_loss', early_stop=EarlyStopSettings(patience=15, min_delta=0.0001, smoothing='window', window=3, ema_beta=0.7, restore_best_weights=True), device='auto', trace_representative=False)` -> `TrainSettings(epochs=5, batch_size=256, seed=0, lr=0.001, weight_decay=0.0001, label_smoothing=0.1, selection_metric='val_raw_loss', early_stop=EarlyStopSettings(patience=15, min_delta=0.0001, smoothing='window', window=3, ema_beta=0.7, restore_best_weights=True), device='auto', trace_representative=False, grad_clip_norm=None, ema_decay=None, init='torch', weight_decay_groups='all_params', uncertainty_weighting=True, class_loss_weight=1.0, lambda_reg_end=0.05, lambda_calib_end=0.1)`

## New defaults
- `nig_timenet_torch.colab.publish.free_job_id(max_versions)`: `<new>` -> `999`
- `nig_timenet_torch.model.taps.record_taps(points)`: `<new>` -> `None`
- `nig_timenet_torch.model.taps.signal_trace(batch_size)`: `<new>` -> `256`
- `nig_timenet_torch.model.taps.signal_trace(ridge)`: `<new>` -> `1.0`
- `nig_timenet_torch.model.taps.tap_points(leaves_only)`: `<new>` -> `False`
- `nig_timenet_torch.trainer.identity.FIT_SWITCH_DEFAULTS.ema_decay`: `<new>` -> `None`
- `nig_timenet_torch.trainer.identity.FIT_SWITCH_DEFAULTS.grad_clip_norm`: `<new>` -> `None`
- `nig_timenet_torch.trainer.identity.FIT_SWITCH_DEFAULTS.init`: `<new>` -> `'torch'`
- `nig_timenet_torch.trainer.trainer.FitSettings(ema_decay)`: `<new>` -> `None`
- `nig_timenet_torch.trainer.trainer.FitSettings(grad_clip_norm)`: `<new>` -> `None`
- `nig_timenet_torch.trainer.trainer.FitSettings(init)`: `<new>` -> `'torch'`
- `nig_timenet_torch.workflow.flat_run.trainer_config(train)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.settings.TrainSettings(class_loss_weight)`: `<new>` -> `1.0`
- `nig_timenet_torch.workflow.settings.TrainSettings(ema_decay)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.settings.TrainSettings(grad_clip_norm)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.settings.TrainSettings(init)`: `<new>` -> `'torch'`
- `nig_timenet_torch.workflow.settings.TrainSettings(lambda_calib_end)`: `<new>` -> `0.1`
- `nig_timenet_torch.workflow.settings.TrainSettings(lambda_reg_end)`: `<new>` -> `0.05`
- `nig_timenet_torch.workflow.settings.TrainSettings(uncertainty_weighting)`: `<new>` -> `True`
- `nig_timenet_torch.workflow.settings.TrainSettings(weight_decay_groups)`: `<new>` -> `'all_params'`
