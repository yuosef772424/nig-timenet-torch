# STATUS: PyTorch port build plan

> Read this first, then ONLY your own batch in `BUILD_PLAN.md` (section 0b) — within it, every step in order,
> each step's own tests green before the next step starts. Update your row in the same PR. Append real obstacles
> to `friction_log.csv`. Keep this file under 60 lines.

B1, B2 merged; B3 in progress. **One branch + one PR per batch** (owner decision, 2026-11-02 — see `BUILD_PLAN.md`
top + section 0b): the step list (P00-P39) is unchanged, only the PR/CI granularity moved from per-step to
per-batch to cut round-trips; every step inside a batch still gets its own acceptance test before the next step
in that batch starts.

| Batch | Steps | Title | State | PR | Note |
|---|---|---|---|---|---|
| B1 | P00-P05 | Skeleton + registry + schema + config + CI (M1) | merged | [#2](https://github.com/yuosef772424/nig-timenet-torch/pull/2) | closes BUILD_PLAN.md V1 via P01 (`reference/data_default/`); `close` target stays enabled (owner decision 2026-11-02); branch `claude/b1-skeleton-registry-config-ci` (owner: one branch per batch, like crypto-signal-prediction) |
| B2 | P06-P10 | Data/oracle loaders + InstanceNorm/Decomp/PatchEmbedding (M2 pt.1) | merged | [#3](https://github.com/yuosef772424/nig-timenet-torch/pull/3) | needs B1; ORACLE_REVIEW.md MAJOR-2 handled in the oracle loader (P07) |
| B3 | P11-P16 | Attention + TransformerBlock + readout + NIG heads + full assembly (M2 pt.2) | in progress | | needs B2; **P15/P16 = M2 exit gate** |
| B4 | P17-P20 | Losses + Kendall + one optimizer step (M3) | todo | | needs B3; **P18/P19 = M3 exit gate**; gradients only, see tolerance table |
| B5 | P21-P24 | Metrics + trainer loop + run identity (M4) | todo | | needs B4 |
| B6 | P25-P30 | SplitPlan + TargetCodec + PredictionTable + Calibrator (M5) | todo | | needs B1, B3; P28 is a wiring test, not a quality claim |
| B7 | P31-P36 | Model contract + tiny_mlp + GRU second-oracle parity (P33b) + input_skip + probe report (M6) | todo | | needs B3, B4; P33b needs B1's second oracle; `input_skip` default stays "none" forever |
| B8 | P37-P39 | Rolling/staleness seams + final review (M7) | todo | | needs B5, B6; seams only, no experiment run |

States: `todo` / `in progress` / `PR open` / `merged` / `closed (rejected)` / `skipped (reason)`.
