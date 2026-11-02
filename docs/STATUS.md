# STATUS

> Read this first, then `docs/RULES.md` once, then ONLY your step in `docs/ROADMAP.md` (port batches below are done; `BUILD_PLAN.md` section 0b) — within it, every step in order,
> each step's own tests green before the next step starts. Update your row in the same PR. Append real obstacles
> to `friction_log.csv`. Keep this file under 60 lines.

B1-B8 merged (P00-P39 incl. P33b, none skipped); follow-up B9-B10 (BUILD_PLAN.md section 14, owner request) merged; M2 and M3 gates green, both oracle suites (`reference/data`, `reference/data_default`) green. Open judgment calls: `friction_log.csv` rows marked "owner to confirm". **One branch + one PR per batch** (owner decision, 2026-11-02 — see `BUILD_PLAN.md`
top + section 0b): the step list (P00-P39) is unchanged, only the PR/CI granularity moved from per-step to
per-batch to cut round-trips; every step inside a batch still gets its own acceptance test before the next step
in that batch starts.

| Batch | Steps | Title | State | PR | Note |
|---|---|---|---|---|---|
| B1 | P00-P05 | Skeleton + registry + schema + config + CI (M1) | merged | [#2](https://github.com/yuosef772424/nig-timenet-torch/pull/2) | closes BUILD_PLAN.md V1 via P01 (`reference/data_default/`); `close` target stays enabled (owner decision 2026-11-02); branch `claude/b1-skeleton-registry-config-ci` (owner: one branch per batch, like crypto-signal-prediction) |
| B2 | P06-P10 | Data/oracle loaders + InstanceNorm/Decomp/PatchEmbedding (M2 pt.1) | merged | [#3](https://github.com/yuosef772424/nig-timenet-torch/pull/3) | needs B1; ORACLE_REVIEW.md MAJOR-2 handled in the oracle loader (P07) |
| B3 | P11-P16 | Attention + TransformerBlock + readout + NIG heads + full assembly (M2 pt.2) | merged | [#4](https://github.com/yuosef772424/nig-timenet-torch/pull/4) | needs B2; **P15/P16 = M2 exit gate**: green (24/24 outputs at forward tol, worst ~11% of budget) |
| B4 | P17-P20 | Losses + Kendall + one optimizer step (M3) | merged | [#5](https://github.com/yuosef772424/nig-timenet-torch/pull/5) | needs B3; **P18/P19 = M3 exit gate**; gradients only, see tolerance table |
| B5 | P21-P24 | Metrics + trainer loop + run identity (M4) | merged | [#6](https://github.com/yuosef772424/nig-timenet-torch/pull/6) | needs B4 |
| B6 | P25-P30 | SplitPlan + TargetCodec + PredictionTable + Calibrator (M5) | merged | [#7](https://github.com/yuosef772424/nig-timenet-torch/pull/7) | needs B1, B3; P28 is a wiring test, not a quality claim |
| B7 | P31-P36 | Model contract + tiny_mlp + GRU second-oracle parity (P33b) + input_skip + probe report (M6) | merged | [#8](https://github.com/yuosef772424/nig-timenet-torch/pull/8) | needs B3, B4; P33b needs B1's second oracle; `input_skip` default stays "none" forever |
| B8 | P37-P39 | Rolling/staleness seams + final review (M7) | merged | [#9](https://github.com/yuosef772424/nig-timenet-torch/pull/9) | needs B5, B6; seams only, no experiment run |
| B9 | P40-P44 | tcn + multi-timeframe + multiclass + cross-asset PanelModel (section 14) | merged | [#10](https://github.com/yuosef772424/nig-timenet-torch/pull/10) | new oracle `reference/data_variants/` |
| B10 | P45 | Colab notebook (section 14) | merged | [#11](https://github.com/yuosef772424/nig-timenet-torch/pull/11) | needs B9 |

States: `todo` / `in progress` / `PR open` / `merged` / `closed (rejected)` / `skipped (reason)`.

## Roadmap 2026-11 (`docs/ROADMAP.md`, rules `docs/RULES.md`)
Model work moved here from TF (owner, 2026-11-02). PRs since last R-REVIEW: 1 (R0).

| Step | Title | State | PR / job | Note |
|---|---|---|---|---|
| R0 | Make the rules checkable | merged | [#14](https://github.com/yuosef772424/nig-timenet-torch/pull/14) | tests for B1, B6, C1, C2, E3, G1-G8 + `tests/ratchets.json`; 8 non-zero ratchets recorded as B-010..B-017 |
| T1 | Complete the trainer | PR open | PR_LINK | `Trainer.fit` (schedules, early stop, best weights, checkpoint + resume); 19 new defaults in D-001; ratchets unchanged |
| T2 | Real data in (TF export + manifest lock fields) | todo | | |
| T3 | Colab runner (device assert, republish) | todo | | |
| T4 | Diagnostics on the targets (`health_report`) | todo | | |
| T5r | Baseline on GPU (`baseline-001`) | todo | | needs T1-T4 |
| T6 | Switches (direct_wide, selection, confidence, NIG floor, interval scale) | todo | | |
| T7r | Baseline vs all switches (`all-001`) | todo | | needs T5r, T6 |
| T8 | Ablate or diagnose, fix, propose default | todo | | needs T7r |
