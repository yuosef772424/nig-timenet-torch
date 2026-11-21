# D-041 — Hardening 2026-11b, package HB1: correctness fixes found by the structure audit (owner 2026-11-21)

Date: 2026-11-21
Owner decision: owner message 2026-11-21: "apply the improvement suggestions, fix the unclear places and launch a hardening campaign for the project; the goal is better structure and more clarity" (plan: `docs/hardening/PLAN-2026-11b.md`, package HB1)
Why: the audit of 2026-11-21 (`docs/hardening/audit-2026-11b/`) found that a cross-section model (relative inputs, cross-coin layers, cross_center) was predicted on batches that cut timestamps (C-01: `predict_stage` cut whole timestamps, then `predict_table` re-sliced them into fixed 512-row batches; C-02: the health report subsample strided rows, not timestamps), so its VAL / TEST tables and health report read part of a timestamp on the rows at each batch border.

## What changes behaviour
- `predict_table(groups)` None = fixed `batch_size` slices exactly as before (every single-coin run is unchanged);
  `predict_stage` and `health_step` now pass the rows' group ids for a cross-section run (`trainer.group_channel`),
  so every batch holds whole timestamps, as `Trainer.evaluate` already did. The VAL / TEST tables, battery and health
  numbers of cross-section runs change (the panel results notes before 2026-11-21 were computed with cut timestamps
  on about one timestamp in five at 100 coins and 512-row batches).
- Not a default, listed for B1: `RunSettings` refuses `train.cross_section_loss` without whole-timestamp batches
  (it failed at the first training step before, C-08); `FitSettings` refuses a grouping `batch_by` together with a
  timestamp `batch_order` (the order was silently ignored, A-30); the run's "batches per epoch" line and
  `step_seconds_per_batch` count the batches fit really takes (C-13).

- `nig_timenet_torch.workflow.prediction_table.predict_table(groups)`: `<new>` -> `None`
