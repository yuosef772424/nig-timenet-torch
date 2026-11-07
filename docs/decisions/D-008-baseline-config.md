# D-008: baseline job config (package H2)

Date: 2026-11-07
Owner decision: package H2 of `docs/hardening/PLAN.md` ("Final campaign order"); the T1 rule applies (a new default needs a
D-nnn file and a pin via `python -m tests.defaults_pin --add`).
Why: the job's free config dict with `.get(key, default)` became the strict `BaselineConfig` (unknown keys raise, rule B5).
Every value equals the old `.get` default, so a job file without the key behaves as before; `max_rows` is new (None = off,
the smoke mode of rule E5). No existing default changed or moved (the `DEFAULT_ARTIFACT` / `DEFAULT_HEALTH_MAX_ROWS`
constants stay in `jobs/baseline.py`). `CONFIG` is the class an entry module declares so `colab.job` validates the config.

- `nig_timenet_torch.jobs.baseline.BaselineConfig(artifact)`: `<new>` -> `'/content/drive/MyDrive/crypto_model/torch_artifacts/preprocessing_output_1'`
- `nig_timenet_torch.jobs.baseline.BaselineConfig(embargo_hours)`: `<new>` -> `0.0`
- `nig_timenet_torch.jobs.baseline.BaselineConfig(health_max_rows)`: `<new>` -> `100000`
- `nig_timenet_torch.jobs.baseline.BaselineConfig(max_rows)`: `<new>` -> `None`
- `nig_timenet_torch.jobs.baseline.BaselineConfig(split_dates)`: `<new>` -> `None`
- `nig_timenet_torch.jobs.baseline.CONFIG`: `<new>` -> `<class 'nig_timenet_torch.jobs.baseline.BaselineConfig'>`
- `nig_timenet_torch.jobs.device_smoke.CONFIG`: `<new>` -> `<class 'nig_timenet_torch.jobs.device_smoke.DeviceSmokeConfig'>`
