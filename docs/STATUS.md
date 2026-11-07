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
Model work moved here from TF (owner, 2026-11-02). PRs since last R-REVIEW: 0 (first review 2026-11-07 in H0, `docs/reviews.md`; next after 5 merged PRs).

| Step | Title | State | PR / job | Note |
|---|---|---|---|---|
| R0 | Make the rules checkable | merged | [#14](https://github.com/yuosef772424/nig-timenet-torch/pull/14) | tests for B1, B6, C1, C2, E3, G1-G8 + `tests/ratchets.json`; 8 non-zero ratchets recorded as B-010..B-017 |
| T1 | Complete the trainer | merged | [#16](https://github.com/yuosef772424/nig-timenet-torch/pull/16) | `Trainer.fit` (schedules, early stop, best weights, checkpoint + resume); 19 new defaults in D-001; ratchets unchanged |
| T2 | Real data in (TF export + manifest lock fields) | merged | [#17](https://github.com/yuosef772424/nig-timenet-torch/pull/17) + TF [#65](https://github.com/yuosef772424/crypto-signal-prediction/pull/65) | `data_io/artifact.py` loader, `build_run_identity`, `SeedMismatchError`. Owner Colab run 2026-11-03 (TF `c0634ba`, torch `b34579c`): artifact `preprocessing_output_1` = 2,520,913 of 3,460,789 windows, 535 coins, max_bar_time 2026-09-26T20:00Z, sealed present 183/183 (all excluded), data_sha256 `c572b1b36960`, hash check passed. B-020 partly closed: that `last_candles` column 3 is the bar OPEN time is still unproven. The 183 sealed coins were all inside the TF main data, so TF results (profile-004 incl.) are not a clean holdout (B-022) |
| T3 | Colab runner (device assert, republish) | merged | [#18](https://github.com/yuosef772424/nig-timenet-torch/pull/18) + TF [#66](https://github.com/yuosef772424/crypto-signal-prediction/pull/66) | one thin notebook (setup / data / runner) over `src/nig_timenet_torch/colab/` + `jobs/`; device asserted AND validated (E1); bundle on Drive before push, `republish`, nothing in `/tmp` (E2); `identity_for_run` builds RunIdentity in one place; TF #66 = `--dataset *.pkl.gz`. first Colab runs 2026-11-03: setup, export (fingerprint `c572b1b36960`), device and publish work on a T4; 64 new defaults in D-002 |
| T4 | Diagnostics on the targets (`health_report`) | merged | [#19](https://github.com/yuosef772424/nig-timenet-torch/pull/19) | `workflow/health_report.py` (`HealthRun` -> JSON + md in the run folder), `eval/target_report.py` (T-skill/T-conf/T-sharp/coverage), `eval/nig_floor.py`, probe with raw next to every layer, `FitSettings.trace` (default off, log-only); 4 new metrics; 9 defaults in D-003; G8 ratchet 2 -> 1. trace grad-share needed cuDNN off on GPU (#23) |
| T5r | Baseline on GPU (`baseline-001`) | in progress | [#20](https://github.com/yuosef772424/nig-timenet-torch/pull/20), fixes #21 (4h input), #22 (split bounds when `split_dates` is None), #23 (progress lines, cuDNN eval backward), #24 (test-split evaluation, D-005) | queue `colab-queue` = `baseline-001-s0`, 5 epochs (owner), rerun true. Three Colab runs failed before a full result (each fixed); one epoch trained on the GPU. No result yet; step time unmeasured |
| T6 | Switches (direct_wide, selection, confidence, NIG floor, interval scale) | todo | | |
| T7r | Baseline vs all switches (`all-001`) | todo | | needs T5r, T6 |
| T8 | Ablate or diagnose, fix, propose default | todo | | needs T7r |

## Hardening 2026-11 (`docs/hardening/PLAN.md`, findings `docs/hardening/AUDIT-2026-11.md`)
Runs before T6; one builder session per package, in order, after H0b and H0c. T5r job runs may continue meanwhile.

| Step | Title | State | PR | Note |
|---|---|---|---|---|
| H0 | Audit + plan + first R-REVIEW | merged | [#26](https://github.com/yuosef772424/nig-timenet-torch/pull/26) | docs only; 37 findings HF-01..HF-37 |
| H0b | Read PR #25 (`feat/ar-stream`), decide what to adopt | todo | | planner; file overlap in the audit, section F |
| H0c | Post-hardening ROADMAP | todo | | planner; model notes in the audit, "For H0c" |
| H1 | Card convention + navigation map | todo | | rule B7 + a card check that fails on touched card-less modules |
| H2 | Job runs read the artifact; GPU smoke path | todo | | closes the #21-#23 classes; rule E5 |
| H3 | Model assembly through registries | todo | | G1, G2, G3, G7, C2 down |
| H4 | Strict trainer config from HeadSpec | todo | | unknown keys raise |
| H5 | Rule and test tooling fixes | todo | | high overlap with PR #25 |
| H6 | Refresh STATUS / handoff / ROADMAP facts | todo | | docs |
