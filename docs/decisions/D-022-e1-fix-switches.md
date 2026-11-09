# D-022: E1 fix switches (FX1): BatchNorm warm-up, detached close mean, NIG head scale, readout dropout, beta-NLL

Date: 2026-11-09
Owner decision: owner message of 2026-11-09 (Arabic, translated; the same instruction as D-021): "fix every error the
gates found, each behind a switch so it can be turned off later; then train the model, test it intensively, locate
the errors precisely and fix them". The step prompt of FX1 restates it: each fix a switch, default = today's behaviour.
Why: on E1 (where the true conditional mean is known) the high / low mu were compressed (slope of the oracle mean on
mu 1.5-1.7, B-062) and the close mu had the wrong sign of correlation with its true mean (B-063);
`docs/results/e1-fixes-001.md` locates both causes.

## What changes behaviour
- Nothing at the defaults: every switch below is off by default and the model, its state_dict keys, the trainer
  config and the run identity are byte-equal to the run before (tests/model/test_head_switches.py,
  tests/model/test_bn_warmup.py, tests/trainer/test_beta_nll.py, the oracle model tests).
- `trainer/tasks.py` `beta_nll` is now ported (it raised NotImplementedError before; TF semantics) and the evidential
  task has a new option `beta_nll_normalize` (default False). Requesting `huber_weight` still raises.

## New switches
- Model overrides (`ModelSettings.overrides`, nig_timenet_v2; `nig_scale` / `order_detach_close` also tiny_mlp):
  `bn_warmup` (bool), `order_detach_close` (bool), `nig_scale` (float or None), `readout_dropout` (float or None;
  nig_timenet_v2 only).
- TrainSettings: `beta_nll` (float >= 0), `beta_nll_normalize` (bool).
- Tool: `workflow/e1_arms.py` (E1 arm runner, mu calibration vs the oracle, head readout gap).

## Changed default reprs (new fields at their defaults; same behaviour)
- `nig_timenet_torch.workflow.settings.RunSettings(train)`: `TrainSettings(epochs=5, batch_size=256, seed=0, lr=0.001, weight_decay=0.0001, label_smoothing=0.1, selection_metric='val_raw_loss', early_stop=EarlyStopSettings(patience=15, min_delta=0.0001, smoothing='window', window=3, ema_beta=0.7, restore_best_weights=True), device='auto', trace_representative=False, grad_clip_norm=None, ema_decay=None, init='torch', weight_decay_groups='all_params', uncertainty_weighting=True, class_loss_weight=1.0, lambda_reg_end=0.05, lambda_calib_end=0.1)` -> `TrainSettings(epochs=5, batch_size=256, seed=0, lr=0.001, weight_decay=0.0001, label_smoothing=0.1, selection_metric='val_raw_loss', early_stop=EarlyStopSettings(patience=15, min_delta=0.0001, smoothing='window', window=3, ema_beta=0.7, restore_best_weights=True), device='auto', trace_representative=False, grad_clip_norm=None, ema_decay=None, init='torch', weight_decay_groups='all_params', uncertainty_weighting=True, class_loss_weight=1.0, lambda_reg_end=0.05, lambda_calib_end=0.1, beta_nll=0.0, beta_nll_normalize=False)`

## New defaults
- `nig_timenet_torch.model.architectures.TinyMLP(nig_scale)`: `<new>` -> `None`
- `nig_timenet_torch.model.architectures.TinyMLP(order_detach_close)`: `<new>` -> `False`
- `nig_timenet_torch.model.builder.NigTimeNetV2(bn_warmup)`: `<new>` -> `False`
- `nig_timenet_torch.model.builder.NigTimeNetV2(nig_scale)`: `<new>` -> `None`
- `nig_timenet_torch.model.builder.NigTimeNetV2(order_detach_close)`: `<new>` -> `False`
- `nig_timenet_torch.model.builder.NigTimeNetV2(readout_dropout)`: `<new>` -> `None`
- `nig_timenet_torch.model.builder.build_nig_timenet_v2(bn_warmup)`: `<new>` -> `False`
- `nig_timenet_torch.model.builder.build_nig_timenet_v2(nig_scale)`: `<new>` -> `None`
- `nig_timenet_torch.model.builder.build_nig_timenet_v2(order_detach_close)`: `<new>` -> `False`
- `nig_timenet_torch.model.builder.build_nig_timenet_v2(readout_dropout)`: `<new>` -> `None`
- `nig_timenet_torch.model.layers.BatchNormChannelsLast(warmup)`: `<new>` -> `False`
- `nig_timenet_torch.model.nig_layers.NIGHead(scale)`: `<new>` -> `None`
- `nig_timenet_torch.model.nig_layers.OrderedMeans(detach_close)`: `<new>` -> `False`
- `nig_timenet_torch.model.nig_layers.OrderedMeans(scale)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.e1_arms.readout_gap(n_train)`: `<new>` -> `20000`
- `nig_timenet_torch.workflow.e1_arms.readout_gap(ridge)`: `<new>` -> `1.0`
- `nig_timenet_torch.workflow.e1_arms.run_arm(epochs)`: `<new>` -> `3`
- `nig_timenet_torch.workflow.settings.TrainSettings(beta_nll)`: `<new>` -> `0.0`
- `nig_timenet_torch.workflow.settings.TrainSettings(beta_nll_normalize)`: `<new>` -> `False`
- `registry:task:evidential.default_options.beta_nll_normalize`: `<new>` -> `False`
