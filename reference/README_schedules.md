# Schedule reference values (ROADMAP T1)

`data_schedules/schedules.json` holds values computed by EXECUTING the TF project's `trainer/schedules.py`
(`schedule_value`, `cosine_warm_restarts`, `build_lr_schedule_fn`) on a grid of settings; `source_sha256` records the
file. Regenerate with `python reference/export_schedules.py <path to trainer/schedules.py>` (numpy only, no TF
install needed). `tests/trainer/test_schedules_parity.py` compares `src/nig_timenet_torch/trainer/schedules.py` with
it at rtol 1e-12. The TF file is identical at the port base `aeafc564` and at `cea9cd58` (checked by sha256 when
exported, 2026-11-02).
