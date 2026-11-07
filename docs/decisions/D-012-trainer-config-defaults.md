# D-012: trainer config validation, task option defaults and loss defaults move into registries (package H4)

Date: 2026-11-07
Owner decision: hardening package H4 of `docs/hardening/PLAN.md` (findings HF-20, HF-21). No value changes: every
new key records the literal that was an inline `.get(key, default)` in `trainer/` before; `Trainer` fills them from the task registry
(`default_options`) and `trainer.targets` (`LOSS_DEFAULTS`, `TARGET_DEFAULTS`) without writing them into `Trainer.config`, so the RunIdentity of
a run is unchanged (`tests/trainer/test_trainer_config_pins.py`). Parameters `register_task_type(default_options)` and `targets_config(options_by_kind)`
are new, their default `None` means no options.

- `nig_timenet_torch.trainer.targets.LOSS_DEFAULTS.constraints`: `<new>` -> `[]`
- `nig_timenet_torch.trainer.targets.LOSS_DEFAULTS.schedules`: `<new>` -> `{}`
- `nig_timenet_torch.trainer.targets.LOSS_DEFAULTS.uncertainty_weighting_exclude_task_types`: `<new>` -> `['evidential']`
- `nig_timenet_torch.trainer.targets.LOSS_DEFAULTS.use_uncertainty_weighting`: `<new>` -> `False`
- `nig_timenet_torch.trainer.targets.TARGET_DEFAULTS.flood_level`: `<new>` -> `None`
- `nig_timenet_torch.trainer.targets.TARGET_DEFAULTS.loss_weight`: `<new>` -> `1.0`
- `nig_timenet_torch.trainer.targets.targets_config(options_by_kind)`: `<new>` -> `None`
- `nig_timenet_torch.trainer.tasks.register_task_type(default_options)`: `<new>` -> `None`
- `registry:head:nig_regression.default_weight`: `<new>` -> `1.0`
- `registry:task:classification.default_options.binary`: `<new>` -> `True`
- `registry:task:classification.default_options.label_smoothing`: `<new>` -> `0.0`
- `registry:task:evidential.default_options.alpha_floor_straight_through`: `<new>` -> `False`
- `registry:task:evidential.default_options.beta_nll`: `<new>` -> `0.0`
- `registry:task:evidential.default_options.huber_weight`: `<new>` -> `0.0`
- `registry:task:evidential.default_options.lambda_calib_var`: `<new>` -> `'lambda_calib'`
- `registry:task:evidential.default_options.lambda_reg_var`: `<new>` -> `'lambda_reg'`
- `registry:task:evidential.default_options.normalize_reg`: `<new>` -> `False`
- `registry:task:evidential.default_options.use_calibration_loss`: `<new>` -> `False`
- `registry:task:regression.default_options.huber_delta`: `<new>` -> `1.0`
- `registry:task:regression.default_options.loss_fn`: `<new>` -> `'mse'`
