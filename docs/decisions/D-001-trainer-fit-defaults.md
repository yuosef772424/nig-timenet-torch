# D-001 — trainer fit: schedule, early-stopping and fit defaults (ROADMAP T1)

Date: 2026-11-02
Owner decision: T1 session prompt (2026-11-02): "any new or changed default needs a docs/decisions/D-nnn file and a
pin in tests/pins/defaults.json via `python -m tests.defaults_pin --add`".
Why: T1 ports the TF trainer's epoch loop pieces. Every default below is the value the TF source uses, so a run
configured only with what the TF code needed behaves the same. Nothing here changes an existing default: all keys are
NEW (the format `<new>` -> value is documented in docs/decisions/README.md).

Sources of the values:
- `schedules.ParamSchedule`: TF trainer/schedules.py `pcfg.get("warmup_epochs", 0)`, `pcfg.get("schedule", "linear")`
- `schedules.LrSettings`: TF `opt_cfg.get("lr_warmup_epochs", 0)`, `sched = opt_cfg.get("lr_schedule", {"type": "constant"})`, `sched.get("cycle_length", 10)`, `sched.get("cycle_mult", 1.5)`, `sched.get("total_epochs", 100)`
- `schedules.cosine_warm_restarts`: TF `cosine_warm_restarts(epoch, lr_initial, lr_min, cycle_length=10, cycle_mult=1.5)`
- `tracker.EarlyStopSettings`: TF BestModelTracker defaults (`patience=15, min_delta=1e-4, smoothing="window", window=3, ema_beta=0.7, restore_best_weights=True`)
- `trainer.FitSettings`: TF TrainSettings.batch_size = 64; seed 0 = seed_everything's own default (core/seeds.py); `early_stop` = the EarlyStopSettings defaults; `lr=None` keeps the optimizer's own rate (no TF equivalent)

Deliberate differences from TF (rule B5, unknown names raise): an unknown schedule curve or LR kind raises
(TF used linear / constant); a negative `warmup_epochs` raises (TF clamped to 0). No default is involved in these.

New defaults:

- `nig_timenet_torch.trainer.schedules.LrSettings(cycle_length)`: `<new>` -> `10`
- `nig_timenet_torch.trainer.schedules.LrSettings(cycle_mult)`: `<new>` -> `1.5`
- `nig_timenet_torch.trainer.schedules.LrSettings(kind)`: `<new>` -> `'constant'`
- `nig_timenet_torch.trainer.schedules.LrSettings(total_epochs)`: `<new>` -> `100`
- `nig_timenet_torch.trainer.schedules.LrSettings(warmup_epochs)`: `<new>` -> `0`
- `nig_timenet_torch.trainer.schedules.ParamSchedule(schedule)`: `<new>` -> `'linear'`
- `nig_timenet_torch.trainer.schedules.ParamSchedule(warmup_epochs)`: `<new>` -> `0`
- `nig_timenet_torch.trainer.schedules.cosine_warm_restarts(cycle_length)`: `<new>` -> `10`
- `nig_timenet_torch.trainer.schedules.cosine_warm_restarts(cycle_mult)`: `<new>` -> `1.5`
- `nig_timenet_torch.trainer.tracker.EarlyStopSettings(ema_beta)`: `<new>` -> `0.7`
- `nig_timenet_torch.trainer.tracker.EarlyStopSettings(min_delta)`: `<new>` -> `0.0001`
- `nig_timenet_torch.trainer.tracker.EarlyStopSettings(patience)`: `<new>` -> `15`
- `nig_timenet_torch.trainer.tracker.EarlyStopSettings(restore_best_weights)`: `<new>` -> `True`
- `nig_timenet_torch.trainer.tracker.EarlyStopSettings(smoothing)`: `<new>` -> `'window'`
- `nig_timenet_torch.trainer.tracker.EarlyStopSettings(window)`: `<new>` -> `3`
- `nig_timenet_torch.trainer.trainer.FitSettings(batch_size)`: `<new>` -> `64`
- `nig_timenet_torch.trainer.trainer.FitSettings(early_stop)`: `<new>` -> `EarlyStopSettings(patience=15, min_delta=0.0001, smoothing='window', window=3, ema_beta=0.7, restore_best_weights=True)`
- `nig_timenet_torch.trainer.trainer.FitSettings(lr)`: `<new>` -> `None`
- `nig_timenet_torch.trainer.trainer.FitSettings(seed)`: `<new>` -> `0`
