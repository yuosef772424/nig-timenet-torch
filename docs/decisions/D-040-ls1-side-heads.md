# D-040 — Loss per trade side: side-payoff and upper-quantile heads on the close (owner 2026-11-19)

Date: 2026-11-19
Owner decision: owner message 2026-11-19: "I want you to test improving the loss function", with the proposals of
the previous session: keep mu on the symmetric log loss; (a) the closed-form expected return e^(mu + s^2/2) - 1 (H1a);
(b) a separate head per side trained on the real payoff (long e^l - 1 - cost, short 1 - e^l - cost), with a small
weight or a partial stop of its gradient into the trunk (as nig_trunk_grad); (c) a quantile (pinball) head for the
upper tail, so that no short is opened when a large rise is likely (translated from Arabic). Confirmed by merging
this PR.
Why: the top SHORT rows lose in every walk-forward run while being right in log units: a short is paid 1 - e^l, the
exp() part costs it -0.08 .. -0.29 % per trade on the top rows and the worst trades are +40 .. +88 % squeezes
(`docs/results/short-side-001.md`, `docs/results/rolling-single-002.md`). The NIG close loss is symmetric in log
units, so nothing in training sees that cost; these switches let a head learn each side's real payoff and the upper
tail without changing mu.

## What changes behaviour
- Nothing changes by default: every new default reproduces the model and the trainer config of before (no auxiliary
  head, `aux_heads=()`; the trainer config is byte-equal, tested in `tests/trainer/test_aux_heads.py`).
- `aux_heads` names auxiliary head kinds on the close target (`side_payoff`: two outputs, each side's payoff after a
  round trip of the battery's `cost_per_side`; `upper_quantile`: the `aux_quantile` quantile of the log-return label,
  pinball loss). `aux_trunk_grad` scales their gradient into the trunk (1.0 = full, 0 = a read-out that leaves the
  trunk and the other heads as they train without them; the owner's "partial stop" is a value in between).
- The task options: `scale` 0.02 divides the errors so the losses are O(1) on a ~2-3 % 8 h move; `huber_delta` 0 =
  squared error (the conditional MEAN payoff, the decision quantity; a Huber delta would down-weight exactly the
  squeezes the short must learn to fear); `short_weight` 1.0 (the owner's "small weight" is available, not used by
  default); `cost_per_side` 0.0008 = the battery's cost; `quantile` 0.9 = the owner's example.
- `aux_loss_weight` 1.0: the auxiliary losses enter with a fixed weight and are never Kendall-weighted (a learned
  weight would grow without bound on a loss of a different scale). `trainer_config(cost_per_side)` 0.0008 = the
  battery default; the run passes its own battery cost.
- The battery table gains `pay_long`, `pay_short`, `q_up` only when the model has the heads.
- `aux_variance_weight` 0.0 (LS1 amendment 1): the side-payoff rows unweighted, as built; > 0 weights each row by its
  detached NIG predictive variance^-v (batch mean 1). Tried on E1 (arm W): no consistent gain, kept off.
- `lr_schedule` "constant" and `lr_min_ratio` 0.1 (LR1, owner 2026-11-19: "do you notice a gap in the settings or in
  the training that can be improved, then train the model"): the run path never passed a learning-rate schedule to
  the trainer (`flat_run.fit_settings_of` left `FitSettings.lr` None), so every run trained at a constant lr; in
  `rolling-ls1-a0-001` the best epoch was the last of 8 in 4 of 7 folds. "constant" keeps that; "cosine" decays from
  `lr` to `lr * lr_min_ratio` in the last epoch (`trainer.schedules.LR_KINDS`).

- `nig_timenet_torch.model.builder.NigTimeNetV2(aux_heads)`: `<new>` -> `()`
- `nig_timenet_torch.model.builder.NigTimeNetV2(aux_trunk_grad)`: `<new>` -> `1.0`
- `nig_timenet_torch.model.builder.build_nig_timenet_v2(aux_heads)`: `<new>` -> `()`
- `nig_timenet_torch.model.builder.build_nig_timenet_v2(aux_trunk_grad)`: `<new>` -> `1.0`
- `nig_timenet_torch.model.heads.aux_head_forward(drop)`: `<new>` -> `None`
- `nig_timenet_torch.trainer.tasks.quantile_task_loss(sample_weight)`: `<new>` -> `None`
- `nig_timenet_torch.trainer.tasks.side_payoff_task_loss(sample_weight)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.flat_run.trainer_config(aux_kinds)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.flat_run.trainer_config(cost_per_side)`: `<new>` -> `0.0008`
- `nig_timenet_torch.workflow.settings.RunSettings(train)`: `TrainSettings(epochs=5, batch_size=256, seed=0, lr=0.001, weight_decay=0.0001, label_smoothing=0.1, selection_metric='val_raw_loss', early_stop=EarlyStopSettings(patience=15, min_delta=0.0001, smoothing='window', window=3, ema_beta=0.7, restore_best_weights=True), device='auto', trace_representative=False, steps_per_execution=1, grad_clip_norm=None, ema_decay=None, init='torch', weight_decay_groups='all_params', uncertainty_weighting=True, class_loss_weight=1.0, lambda_reg_end=0.05, lambda_calib_end=0.1, beta_nll=0.0, beta_nll_normalize=False, loss_balance='none', kendall_init_logvar=0.0, staged=None, cross_section_loss='none', cross_section_weight=1.0, direction_balance=0.0, batch_by='rows', batch_order='rows', time_block_bars=1)` -> `TrainSettings(epochs=5, batch_size=256, seed=0, lr=0.001, weight_decay=0.0001, label_smoothing=0.1, selection_metric='val_raw_loss', early_stop=EarlyStopSettings(patience=15, min_delta=0.0001, smoothing='window', window=3, ema_beta=0.7, restore_best_weights=True), device='auto', trace_representative=False, steps_per_execution=1, grad_clip_norm=None, ema_decay=None, init='torch', weight_decay_groups='all_params', uncertainty_weighting=True, class_loss_weight=1.0, lambda_reg_end=0.05, lambda_calib_end=0.1, beta_nll=0.0, beta_nll_normalize=False, loss_balance='none', kendall_init_logvar=0.0, staged=None, cross_section_loss='none', cross_section_weight=1.0, direction_balance=0.0, batch_by='rows', batch_order='rows', time_block_bars=1, aux_loss_weight=1.0, aux_quantile=0.9, aux_variance_weight=0.0, lr_schedule='constant', lr_min_ratio=0.1)`
- `nig_timenet_torch.workflow.settings.TrainSettings(aux_loss_weight)`: `<new>` -> `1.0`
- `nig_timenet_torch.workflow.settings.TrainSettings(aux_quantile)`: `<new>` -> `0.9`
- `nig_timenet_torch.workflow.settings.TrainSettings(aux_variance_weight)`: `<new>` -> `0.0`
- `nig_timenet_torch.workflow.settings.TrainSettings(lr_min_ratio)`: `<new>` -> `0.1`
- `nig_timenet_torch.workflow.settings.TrainSettings(lr_schedule)`: `<new>` -> `'constant'`
- `registry:head:side_payoff.default_weight`: `<new>` -> `1.0`
- `registry:head:upper_quantile.default_weight`: `<new>` -> `1.0`
- `registry:task:pinball.default_options.quantile`: `<new>` -> `0.9`
- `registry:task:pinball.default_options.scale`: `<new>` -> `0.02`
- `registry:task:side_payoff.default_options.cost_per_side`: `<new>` -> `0.0008`
- `registry:task:side_payoff.default_options.huber_delta`: `<new>` -> `0.0`
- `registry:task:side_payoff.default_options.nig_keys`: `<new>` -> `{}`
- `registry:task:side_payoff.default_options.scale`: `<new>` -> `0.02`
- `registry:task:side_payoff.default_options.short_weight`: `<new>` -> `1.0`
- `registry:task:side_payoff.default_options.variance_weight`: `<new>` -> `0.0`
