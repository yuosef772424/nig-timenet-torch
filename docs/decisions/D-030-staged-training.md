# D-030: staged training (ROADMAP ST1, card F): new options, all off or at the old behaviour

Date: 2026-11-15
Owner decision: the owner's request of 2026-11-15 (STATUS row ST1: "train on a part of the data until the model shows a
sign of memorisation, then add data and lower the learning rate", spec
`docs/audit/2026-11-14-model-first/cards/F_iterative_training.md`) and the session prompt that asked for the build
("the default is OFF, so today's runs are unchanged and the pins hold").

## What changes behaviour
- Nothing by default: `TrainSettings.staged` is `None`, so `workflow.run.run_all` calls `steps.train` exactly as before
  (`tests/jobs/test_baseline_pin.py` and `tests/trainer/test_trainer_config_pins.py` are unchanged and green). Staged
  training is a different `Trainer` loop only when a `StageSettings` is given (`steps.train_staged`).
- `Run`, `flat_run.predict_stage` and the other helpers gain optional arguments whose defaults are today's behaviour
  (`fold` 0, `staged` / `plan` None).
- The `StageSettings` defaults are the card's section 2-4 numbers (the experiment's definitions, fixed before any run):
  start 1/32 of the rows, the share doubles, lr x 0.5 per stage down to 1e-6, the sign needs `k` 3 evaluations without a
  `min_delta` 1e-3 improvement of the 3-point mean of the validation loss while the eval-mode train probe falls, probe 2000
  rows, validation strided to 60000 rows, step cap 20 passes, 100 warm-up steps, 50 BatchNorm refresh steps, stage best
  weights restored, optimizer reset. The study entry `staged` has its own job defaults (lr 3e-4 = the lr of the card's
  lr example, seeds 0 and 1 = rule C4).

## Changed pin (a repr only)
- The pinned repr of the default `RunSettings.train` gains the new field `staged=None`; the value of every existing field
  is the same (the pin tool compares the printed repr of the default object).
- `nig_timenet_torch.workflow.settings.RunSettings(train)`: `TrainSettings(epochs=5, batch_size=256, seed=0, lr=0.001, weight_decay=0.0001, label_smoothing=0.1, selection_metric='val_raw_loss', early_stop=EarlyStopSettings(patience=15, min_delta=0.0001, smoothing='window', window=3, ema_beta=0.7, restore_best_weights=True), device='auto', trace_representative=False, steps_per_execution=1, grad_clip_norm=None, ema_decay=None, init='torch', weight_decay_groups='all_params', uncertainty_weighting=True, class_loss_weight=1.0, lambda_reg_end=0.05, lambda_calib_end=0.1, beta_nll=0.0, beta_nll_normalize=False, loss_balance='none', kendall_init_logvar=0.0)` -> `TrainSettings(epochs=5, batch_size=256, seed=0, lr=0.001, weight_decay=0.0001, label_smoothing=0.1, selection_metric='val_raw_loss', early_stop=EarlyStopSettings(patience=15, min_delta=0.0001, smoothing='window', window=3, ema_beta=0.7, restore_best_weights=True), device='auto', trace_representative=False, steps_per_execution=1, grad_clip_norm=None, ema_decay=None, init='torch', weight_decay_groups='all_params', uncertainty_weighting=True, class_loss_weight=1.0, lambda_reg_end=0.05, lambda_calib_end=0.1, beta_nll=0.0, beta_nll_normalize=False, loss_balance='none', kendall_init_logvar=0.0, staged=None)`

## New defaults (pinned as they are)
- `nig_timenet_torch.jobs.staged.CONFIG`: `<new>` -> `<class 'nig_timenet_torch.jobs.staged.StagedConfig'>`
- `nig_timenet_torch.jobs.staged.StagedConfig(artifact)`: `<new>` -> `'/content/drive/MyDrive/crypto_model/torch_artifacts/preprocessing_output_1'`
- `nig_timenet_torch.jobs.staged.StagedConfig(base_tf)`: `<new>` -> `None`
- `nig_timenet_torch.jobs.staged.StagedConfig(baselines)`: `<new>` -> `('b1', 'b2', 'b3')`
- `nig_timenet_torch.jobs.staged.StagedConfig(batch_size)`: `<new>` -> `512`
- `nig_timenet_torch.jobs.staged.StagedConfig(lr)`: `<new>` -> `0.0003`
- `nig_timenet_torch.jobs.staged.StagedConfig(max_rows)`: `<new>` -> `None`
- `nig_timenet_torch.jobs.staged.StagedConfig(model_overrides)`: `<new>` -> `{}`
- `nig_timenet_torch.jobs.staged.StagedConfig(seeds)`: `<new>` -> `(0, 1)`
- `nig_timenet_torch.jobs.staged.StagedConfig(split)`: `<new>` -> `{}`
- `nig_timenet_torch.jobs.staged.StagedConfig(staged)`: `<new>` -> `{}`
- `nig_timenet_torch.jobs.staged.StagedConfig(train)`: `<new>` -> `{}`
- `nig_timenet_torch.jobs.staged.run_settings(code_sha)`: `<new>` -> `''`
- `nig_timenet_torch.trainer.staged.GrowthContext(seed)`: `<new>` -> `0`
- `nig_timenet_torch.trainer.staged.StageSettings(bn_refresh_steps)`: `<new>` -> `50`
- `nig_timenet_torch.trainer.staged.StageSettings(eval_every)`: `<new>` -> `None`
- `nig_timenet_torch.trainer.staged.StageSettings(gamma)`: `<new>` -> `0.5`
- `nig_timenet_torch.trainer.staged.StageSettings(grow)`: `<new>` -> `2.0`
- `nig_timenet_torch.trainer.staged.StageSettings(growth)`: `<new>` -> `'stride'`
- `nig_timenet_torch.trainer.staged.StageSettings(k)`: `<new>` -> `3`
- `nig_timenet_torch.trainer.staged.StageSettings(lr_floor)`: `<new>` -> `1e-06`
- `nig_timenet_torch.trainer.staged.StageSettings(min_delta)`: `<new>` -> `0.001`
- `nig_timenet_torch.trainer.staged.StageSettings(new_weight)`: `<new>` -> `1`
- `nig_timenet_torch.trainer.staged.StageSettings(probe_rows)`: `<new>` -> `2000`
- `nig_timenet_torch.trainer.staged.StageSettings(reset_optimizer)`: `<new>` -> `True`
- `nig_timenet_torch.trainer.staged.StageSettings(restore_stage_best)`: `<new>` -> `True`
- `nig_timenet_torch.trainer.staged.StageSettings(smooth_points)`: `<new>` -> `3`
- `nig_timenet_torch.trainer.staged.StageSettings(start_share)`: `<new>` -> `0.03125`
- `nig_timenet_torch.trainer.staged.StageSettings(step_cap_epochs)`: `<new>` -> `20.0`
- `nig_timenet_torch.trainer.staged.StageSettings(val_max_rows)`: `<new>` -> `60000`
- `nig_timenet_torch.trainer.staged.StageSettings(warmup_steps)`: `<new>` -> `100`
- `nig_timenet_torch.trainer.staged.StageSpec(lr_at)`: `<new>` -> `()`
- `nig_timenet_torch.trainer.staged.StageSpec(restart)`: `<new>` -> `True`
- `nig_timenet_torch.trainer.staged.StageSpec(stop_at_sign)`: `<new>` -> `True`
- `nig_timenet_torch.trainer.staged.StagedResult(extra)`: `<new>` -> `{}`
- `nig_timenet_torch.trainer.staged.StagedResult(val_rows)`: `<new>` -> `0`
- `nig_timenet_torch.trainer.staged.staged_fit(deadline)`: `<new>` -> `None`
- `nig_timenet_torch.trainer.staged.staged_fit(identity)`: `<new>` -> `None`
- `nig_timenet_torch.trainer.staged.staged_fit(on_eval)`: `<new>` -> `None`
- `nig_timenet_torch.trainer.staged.staged_fit(resume)`: `<new>` -> `True`
- `nig_timenet_torch.trainer.staged.staged_fit(run_dir)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run.Run(plan)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run.Run(staged)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run.train_staged(deadline)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run.train_staged(on_eval)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run.train_staged(stages)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.settings.TrainSettings(staged)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.staged_report.growth_context(seed)`: `<new>` -> `0`
- `nig_timenet_torch.workflow.staged_study.run_arm(ckpt_root)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.staged_study.run_arm(on_eval)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.staged_study.run_arm(plan_from)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.staged_study.run_arm(test)`: `<new>` -> `True`
- `nig_timenet_torch.workflow.staged_study.run_staged_study(baselines)`: `<new>` -> `('b1', 'b2', 'b3')`
- `nig_timenet_torch.workflow.staged_study.run_staged_study(ckpt_root)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.staged_study.run_staged_study(on_eval)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.staged_study.run_staged_study(test)`: `<new>` -> `True`
