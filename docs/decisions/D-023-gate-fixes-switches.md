# D-023: gate fixes (FX2): loss balancer, Kendall start value, NIG trunk-gradient scale, fitted NIG label scale

Date: 2026-11-09
Owner decision: owner task of 2026-11-09 (FX2 step prompt): "B-060 / gate A5 ... fix it behind a switch (default
off) ... register a new method as a registered component"; "B-052, the beta floor ... add a switch that scales the
floor with the training label variance (a learned-state transform: fit on train only and stored with the model, rule
B4)"; same standing instruction as D-021 / D-022: every fix is a switch whose default is today's behaviour.
Why: on E1 the class heads get < 0.02 of the shared-trunk gradient (gate A5, B-060) and the beta floor (0.01) is ~25x
the label variance at label scale 1 (B-052); `docs/results/gates-fixes-001.md` localises both and judges every arm.

## Owner decision 2026-11-09 (later message, relayed by the parent session): useful switches become the default
"every switch that the evidence so far shows to be USEFUL becomes the DEFAULT ... the FX1 bundle (bn_warmup,
order_detach_close), any FX2 switch that clearly helped, and nig_scale if your evidence supports it. Switches that
were neutral or harmful stay off." Applied to the RUN default model (`workflow.settings.default_model_settings`,
new constant `RUN_DEFAULT_SWITCHES`): `bn_warmup`, `order_detach_close` (FX1, verified), `nig_scale_fit: train_std`
(FX2: sd scale 1.26 -> 1.04, T-sharp, close corr, fewest gate fails; the fitted form replaces the fixed
`nig_scale 0.01`, which stays off) and `nig_trunk_grad: 0.3` (FX2: gate A5 passes on 3 of 3 seeds, corr unchanged).
Off (neutral or harmful): `loss_balance equal_grad_norm` (regression corr 0.96 -> 0.74-0.84), `kendall_init_logvar`,
`class_head_hidden 128`, `label_smoothing 0`, `readout_dropout 0`, fixed `nig_scale`. The builder's own defaults and
the TF-parity preset `ANTI_MEMORIZATION_CONFIG` are unchanged (oracle tests). `workflow.gate_run.baseline_arm` compares
against `default_model_settings()`, so the baseline-arm gate exemption follows the new default automatically; the
baseline-job fingerprint pins change with it.

- `nig_timenet_torch.workflow.settings.RUN_DEFAULT_SWITCHES.bn_warmup`: `<new>` -> `True`
- `nig_timenet_torch.workflow.settings.RUN_DEFAULT_SWITCHES.order_detach_close`: `<new>` -> `True`
- `nig_timenet_torch.workflow.settings.RUN_DEFAULT_SWITCHES.nig_scale_fit`: `<new>` -> `'train_std'`
- `nig_timenet_torch.workflow.settings.RUN_DEFAULT_SWITCHES.nig_trunk_grad`: `<new>` -> `0.3`
- `nig_timenet_torch.workflow.settings.RunSettings(model)`: `ModelSettings(architecture='nig_timenet_v2', overrides={'d_model': 64, 'num_layers': 2, 'head_hidden': 64, 'class_head_hidden': 32, 'dropout': 0.25, 'input_clip': 4.0, 'stats_mode': 'symlog', 'input_noise_std': 0.1, 'feature_dropout': 0.1, 'linear_path_l2': 0.001, 'level_passthrough': True, 'level_norm': 'batch', 'encoder': 'gru'})` -> `ModelSettings(architecture='nig_timenet_v2', overrides={'d_model': 64, 'num_layers': 2, 'head_hidden': 64, 'class_head_hidden': 32, 'dropout': 0.25, 'input_clip': 4.0, 'stats_mode': 'symlog', 'input_noise_std': 0.1, 'feature_dropout': 0.1, 'linear_path_l2': 0.001, 'level_passthrough': True, 'level_norm': 'batch', 'encoder': 'gru', 'bn_warmup': True, 'order_detach_close': True, 'nig_scale_fit': 'train_std', 'nig_trunk_grad': 0.3})`

## What changes behaviour
- The run default model (above): every run, job, notebook and gate run that does not set these overrides now trains
  with them; checkpoints of earlier runs were built without them (state_dict gains `label_scale` buffers).
- At the builder level nothing changes: every switch below is off in `build_nig_timenet_v2`; the model, its state_dict keys, the trainer config (and so the
  run identity) and a default checkpoint's keys are byte-equal to the run before (tests/model/test_fx2_switches.py,
  tests/trainer/test_loss_balance.py, the trainer-config and assembly pins).
- `workflow/e1_arms.py` result rows gain `a5`, `a6`, `a6_probe_end`, `class_auc` (tool output only).

## New switches
- Model overrides (nig_timenet_v2): `nig_trunk_grad` (float, 1.0 = off: the gradient of the NIG heads into the trunk
  is multiplied by it, `nig_layers.grad_scale`), `nig_scale_fit` (`model.label_scale.LABEL_SCALES`: `fixed` = off,
  `train_std` = the NIG head scale s is fitted on the TRAINING labels (mean std of the NIG labels) and stored as the
  heads' `label_scale` buffers; beta floor = beta_min x s^2). `nig_scale_fit` and a fixed `nig_scale` are exclusive.
- Trainer config `loss` keys: `balance` (`trainer.loss_balance.LOSS_BALANCERS`: `none`, `equal_grad_norm`; a
  balancer other than `none` refuses Kendall tasks) and `kendall_init_logvar` (start value of the Kendall
  log-variances). Written into the config only when not at their default (the baseline config stays pinned).
- TrainSettings: `loss_balance`, `kendall_init_logvar` (feed the two keys above).

## Changed default reprs (new fields at their defaults; same behaviour)
- `nig_timenet_torch.workflow.settings.RunSettings(train)`: `TrainSettings(epochs=5, batch_size=256, seed=0, lr=0.001, weight_decay=0.0001, label_smoothing=0.1, selection_metric='val_raw_loss', early_stop=EarlyStopSettings(patience=15, min_delta=0.0001, smoothing='window', window=3, ema_beta=0.7, restore_best_weights=True), device='auto', trace_representative=False, grad_clip_norm=None, ema_decay=None, init='torch', weight_decay_groups='all_params', uncertainty_weighting=True, class_loss_weight=1.0, lambda_reg_end=0.05, lambda_calib_end=0.1, beta_nll=0.0, beta_nll_normalize=False)` -> `TrainSettings(epochs=5, batch_size=256, seed=0, lr=0.001, weight_decay=0.0001, label_smoothing=0.1, selection_metric='val_raw_loss', early_stop=EarlyStopSettings(patience=15, min_delta=0.0001, smoothing='window', window=3, ema_beta=0.7, restore_best_weights=True), device='auto', trace_representative=False, grad_clip_norm=None, ema_decay=None, init='torch', weight_decay_groups='all_params', uncertainty_weighting=True, class_loss_weight=1.0, lambda_reg_end=0.05, lambda_calib_end=0.1, beta_nll=0.0, beta_nll_normalize=False, loss_balance='none', kendall_init_logvar=0.0)`

## New defaults
- `nig_timenet_torch.model.builder.NigTimeNetV2(nig_scale_fit)`: `<new>` -> `'fixed'`
- `nig_timenet_torch.model.builder.NigTimeNetV2(nig_trunk_grad)`: `<new>` -> `1.0`
- `nig_timenet_torch.model.builder.build_nig_timenet_v2(nig_scale_fit)`: `<new>` -> `'fixed'`
- `nig_timenet_torch.model.builder.build_nig_timenet_v2(nig_trunk_grad)`: `<new>` -> `1.0`
- `nig_timenet_torch.model.nig_layers.NIGHead(fitted_scale)`: `<new>` -> `False`
- `nig_timenet_torch.model.nig_layers.OrderedMeans(fitted_scale)`: `<new>` -> `False`
- `nig_timenet_torch.trainer.targets.LOSS_DEFAULTS.balance`: `<new>` -> `'none'`
- `nig_timenet_torch.trainer.targets.LOSS_DEFAULTS.kendall_init_logvar`: `<new>` -> `0.0`
- `nig_timenet_torch.trainer.task_weighting.UncertaintyWeightedLoss(init_log_var)`: `<new>` -> `0.0`
- `nig_timenet_torch.trainer.trainer.compute_losses(multipliers)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.settings.TrainSettings(kendall_init_logvar)`: `<new>` -> `0.0`
- `nig_timenet_torch.workflow.settings.TrainSettings(loss_balance)`: `<new>` -> `'none'`

## Not decided here
- Further default changes are owner decisions.
- The A5 / A6 protocol changes proposed in `docs/suggestions.csv` (FX2 rows); the gate thresholds are unchanged.
