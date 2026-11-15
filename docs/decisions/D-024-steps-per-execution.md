# D-024: `steps_per_execution` in the trainer (Keras option, speed only)

Date: 2026-11-15
Owner decision: owner request 2026-11-15 ("update the trainer to support steps_per_execution=32 and measure whether the
weights differ").
Evidence: `tests/trainer/test_steps_per_execution.py`: for N in 2, 5, 32, 1000, with and without gradient clipping and
EMA, the final weights are bit-identical to N = 1 (`torch.equal` on every tensor), the history agrees to rtol 1e-6, a
resume across groups equals an uninterrupted run. The option groups the HOST READ-BACK of the step logs only (one
`.tolist()` per N steps instead of one `float()` per step); the optimizer steps are the same, in the same order.

## What changes behaviour
- Nothing at the default (`steps_per_execution = 1` reproduces the old per-step reads). The option is left out of the
  run identity (like `trace`): a run may be resumed with another N. Not Keras' graph fusion: no CUDA-graph / compile
  step is involved, so the speed gain exists only where a per-step sync was the cost (GPU).

## Changed default reprs (a new field at its old behaviour)
- `nig_timenet_torch.workflow.settings.RunSettings(train)`: `TrainSettings(epochs=5, batch_size=256, seed=0, lr=0.001, weight_decay=0.0001, label_smoothing=0.1, selection_metric='val_raw_loss', early_stop=EarlyStopSettings(patience=15, min_delta=0.0001, smoothing='window', window=3, ema_beta=0.7, restore_best_weights=True), device='auto', trace_representative=False, grad_clip_norm=None, ema_decay=None, init='torch', weight_decay_groups='all_params', uncertainty_weighting=True, class_loss_weight=1.0, lambda_reg_end=0.05, lambda_calib_end=0.1, beta_nll=0.0, beta_nll_normalize=False)` -> `TrainSettings(epochs=5, batch_size=256, seed=0, lr=0.001, weight_decay=0.0001, label_smoothing=0.1, selection_metric='val_raw_loss', early_stop=EarlyStopSettings(patience=15, min_delta=0.0001, smoothing='window', window=3, ema_beta=0.7, restore_best_weights=True), device='auto', trace_representative=False, steps_per_execution=1, grad_clip_norm=None, ema_decay=None, init='torch', weight_decay_groups='all_params', uncertainty_weighting=True, class_loss_weight=1.0, lambda_reg_end=0.05, lambda_calib_end=0.1, beta_nll=0.0, beta_nll_normalize=False)`
