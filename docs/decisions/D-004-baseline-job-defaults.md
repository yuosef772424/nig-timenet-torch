# D-004 — T5r baseline job entry: new defaults (ROADMAP T5r)

Date: 2026-11-03
Owner decision: T5r session prompt (2026-11-03): job entry `jobs/baseline.py` for `baseline-001`; the T1 rule applies (a
new default needs a D-nnn file and a pin via `python -m tests.defaults_pin --add`).
Why: nothing here changes an existing default; every key below is NEW (the `<new>` format of docs/decisions/README.md).
`Trainer.fit(deadline=None)` is not a pinned key: `None` = the loop is unchanged. The label smoothing 0.1 is the TF
anti-memorization default (ORACLE_REVIEW V1); the artifact path is the owner's Drive path; the health report uses at
most 100000 evenly strided validation rows so they fit on the GPU.

New defaults:

- `nig_timenet_torch.jobs.baseline.DEFAULT_ARTIFACT`: `<new>` -> `'/content/drive/MyDrive/crypto_model/torch_artifacts/preprocessing_output_1'`
- `nig_timenet_torch.jobs.baseline.DEFAULT_HEALTH_MAX_ROWS`: `<new>` -> `100000`
- `nig_timenet_torch.jobs.baseline.main(argv)`: `<new>` -> `None`
- `nig_timenet_torch.jobs.baseline.trainer_config(label_smoothing)`: `<new>` -> `0.1`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.class_head_hidden`: `<new>` -> `32`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.d_model`: `<new>` -> `64`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.dropout`: `<new>` -> `0.25`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.encoder`: `<new>` -> `'gru'`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.feature_dropout`: `<new>` -> `0.1`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.head_hidden`: `<new>` -> `64`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.input_clip`: `<new>` -> `4.0`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.input_noise_std`: `<new>` -> `0.1`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.level_norm`: `<new>` -> `'batch'`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.level_passthrough`: `<new>` -> `True`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.linear_path_l2`: `<new>` -> `0.001`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.num_layers`: `<new>` -> `2`
- `nig_timenet_torch.jobs.baseline.ANTI_MEMORIZATION_CONFIG.stats_mode`: `<new>` -> `'symlog'`
