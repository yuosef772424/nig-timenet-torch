# D-023: the default model gets MODEL_FIRST_FIXES (model-first audit 2026-11-14)

Date: 2026-11-14
Owner decision: owner-approved behaviour change: `default_model_settings()` returns the anti-memorization preset plus
`MODEL_FIRST_FIXES` = bn_warmup, order_detach_close, nig_scale='train_std', input_scaler='train_std'.
Evidence: E1, 3 seeds: the blocking gate fails 7.7 -> 2.7 at 3 epochs and 7.3 -> 2.7 at 10 epochs; the close mu
correlation with its true mean 0.11 -> 0.88 at 10 epochs. Details: docs/audit/2026-11-14-model-first/NOTES.md.
Old model: `ModelSettings("nig_timenet_v2", dict(ANTI_MEMORIZATION_CONFIG))`.

## What changes behaviour
- The default model has an input scaler and a fitted NIG label scale (fitted in `workflow.run.build`), so its
  state_dict keys, run identity and configuration hashes change. All other defaults are unchanged.

## Changed defaults
- `nig_timenet_torch.workflow.settings.RunSettings(model)`: `ModelSettings(architecture='nig_timenet_v2', overrides={'d_model': 64, 'num_layers': 2, 'head_hidden': 64, 'class_head_hidden': 32, 'dropout': 0.25, 'input_clip': 4.0, 'stats_mode': 'symlog', 'input_noise_std': 0.1, 'feature_dropout': 0.1, 'linear_path_l2': 0.001, 'level_passthrough': True, 'level_norm': 'batch', 'encoder': 'gru'})` -> `ModelSettings(architecture='nig_timenet_v2', overrides={'d_model': 64, 'num_layers': 2, 'head_hidden': 64, 'class_head_hidden': 32, 'dropout': 0.25, 'input_clip': 4.0, 'stats_mode': 'symlog', 'input_noise_std': 0.1, 'feature_dropout': 0.1, 'linear_path_l2': 0.001, 'level_passthrough': True, 'level_norm': 'batch', 'encoder': 'gru', 'bn_warmup': True, 'order_detach_close': True, 'nig_scale': 'train_std', 'input_scaler': 'train_std'})`
