# STATUS

> Read this first, then `docs/RULES.md` once, then ONLY your step in `docs/ROADMAP.md` (port batches below are done; `BUILD_PLAN.md` section 0b) — within it, every step in order,
> each step's own tests green before the next step starts. Update your row in the same PR. Append real obstacles
> to `friction_log.csv`. Keep this file under 60 lines.
Port done: B1-B10 merged ([#2](https://github.com/yuosef772424/nig-timenet-torch/pull/2)-[#11](https://github.com/yuosef772424/nig-timenet-torch/pull/11); per-batch table and notes in `docs/BUILD_PLAN.md` section 0b and git history of this file); M2 and M3 gates green, both oracle suites green. Open judgment calls: `friction_log.csv` rows marked "owner to confirm".

States: `todo` / `in progress` / `PR open` / `merged` / `closed (rejected)` / `skipped (reason)`.

## Hardening 2026-11 (`docs/hardening/PLAN.md`, final order at its end; findings `docs/hardening/AUDIT-2026-11.md`)
CAMPAIGN DONE (all rows merged). Outcome per package: `docs/hardening/PLAN.md` "Outcome"; tests 515 -> 749; ratchets G1 0, G2 5 (honest, B-046), G3 0, G4 0, G7 0, C2 0, G8 1 (`model/panel.py`).

| Step | Title | State | PR | Note |
|---|---|---|---|---|
| H0 | Audit + plan + first R-REVIEW | merged | [#26](https://github.com/yuosef772424/nig-timenet-torch/pull/26) | 37 findings HF-01..HF-37 |
| H0b | Read PR #25 (`feat/ar-stream`), decide what to adopt | merged | [#27](https://github.com/yuosef772424/nig-timenet-torch/pull/27) | `docs/hardening/BRANCH_AR_STREAM.md`, packages I1..I7 |
| H0c | Post-hardening ROADMAP + model development protocol | merged | [#28](https://github.com/yuosef772424/nig-timenet-torch/pull/28) | `docs/MODEL_DEV_PROTOCOL.md` (rule C6); I4 split (rolling -> F2, panel parked), I7 not opened |
| T4b | T-sharp per head vs predicted move (ROADMAP fix, first) | merged | [#29](https://github.com/yuosef772424/nig-timenet-torch/pull/29) | key `share_std_below_move`; close verdict informational (O-1); before any T5r number is read |
| I1 | Loading and config foundations | merged | [#30](https://github.com/yuosef772424/nig-timenet-torch/pull/30) | D1 mandatory in `load_dataset` (`allow_unchecked`); D-007; before H2 |
| H2 | Job runs read the artifact; GPU smoke path | merged | [#31](https://github.com/yuosef772424/nig-timenet-torch/pull/31) | closes #21-#23 classes; rule E5; queue `baseline-smoke-001` before any long job |
| H1 | Card convention + navigation map | merged | [#32](https://github.com/yuosef772424/nig-timenet-torch/pull/32) | rule B7; `docs/MAP.md` (`python -m tests.build_map`); allowlist 37; second R-REVIEW |
| I5 | Evaluation battery + adapter | merged | [#33](https://github.com/yuosef772424/nig-timenet-torch/pull/33) | battery ported (`7c9736b`, `f052cf6`); `eval/battery_adapter.py`, `battery_profit.py`, `workflow/battery_run.py`; T-profit target; D-009; O-1 closed |
| I2 | Native data pipeline | merged | [#34](https://github.com/yuosef772424/nig-timenet-torch/pull/34) | `data/` ported (no ar_windows/panel); TF parity bit-exact; native 4h artifact test closes HF-15; D-010 (155 keys) |
| H3 | Model assembly through registries | merged | [#35](https://github.com/yuosef772424/nig-timenet-torch/pull/35) | `model/encoders.py` ENCODERS, heads via HEAD_REGISTRY, `model/contract.py` (D-011), health report for every architecture; G1 0, G2 1, G3 0, G7 0, C2 0 |
| H4 | Strict trainer config | merged | [#36](https://github.com/yuosef772424/nig-timenet-torch/pull/36) | `trainer/targets.py`; unknown keys raise; D-012; `.get` ratchet 22 -> 0; direction AUCs not selectable |
| I3 | Run steps (flat single split) | merged | [#38](https://github.com/yuosef772424/nig-timenet-torch/pull/38) | `workflow/run.py` steps; `jobs/baseline.py` 107 lines (config + 25-line main); battery in the run (cost per side, p_dir); `notebooks/main.ipynb`; D-013; HF-23, B-035 closed |
| H5 | Rule and test tooling fixes | merged | [#39](https://github.com/yuosef772424/nig-timenet-torch/pull/39) | G2 honest 5 (B-046), `tests.pin_diff` in CI, D-014 pin scope, D-015 manifest `base_tf`/`timeframes`, `tests/jobs/`, helpers modules |
| H6 | Facts refresh + third R-REVIEW | merged | [#40](https://github.com/yuosef772424/nig-timenet-torch/pull/40) | STATUS, handoff, ROADMAP T5r, PLAN/AUDIT outcomes, NOTES re-triage |

## Roadmap (`docs/ROADMAP.md`, rules `docs/RULES.md`, protocol `docs/MODEL_DEV_PROTOCOL.md`)
PRs since last R-REVIEW: 2 (#47 the review itself, #48 G2; fourth review 2026-11-08, `docs/reviews.md`, covered #41-#46; the next is due after 5 more merged PRs; the counter is hand-kept until H7 derives it).
Done: R0 [#14](https://github.com/yuosef772424/nig-timenet-torch/pull/14), T1 [#16](https://github.com/yuosef772424/nig-timenet-torch/pull/16), T2 [#17](https://github.com/yuosef772424/nig-timenet-torch/pull/17), T3 [#18](https://github.com/yuosef772424/nig-timenet-torch/pull/18), T4 [#19](https://github.com/yuosef772424/nig-timenet-torch/pull/19), T4b [#29](https://github.com/yuosef772424/nig-timenet-torch/pull/29), the T5r job ([#20](https://github.com/yuosef772424/nig-timenet-torch/pull/20), fixes #21-#24). Data: artifact `c572b1b36960` (2,520,913 windows, 535 coins, sealed 183/183 excluded; B-020, B-022 open).
GPU results (branch `colab-results`): `baseline-002-s0` (2026-11-08, @8451a62, equals r2): read in T5r (`docs/results/baseline-001.md`): high/low rank IC 0.39/0.44 tie the HAR volatility baseline; T-sharp 0.012/0.272 (the recorded 0.84/0.89 was the B-051 misdecode); no use of the outputs profitable after cost on VAL and TEST.

| Step | Title | State | PR / job | Note |
|---|---|---|---|---|
| T5r | Read the baseline (`docs/results/baseline-001.md`) | merged | phase 1 [#44](https://github.com/yuosef772424/nig-timenet-torch/pull/44), phase 2 [#45](https://github.com/yuosef772424/nig-timenet-torch/pull/45) | weights recovered, no re-training; B-051 decode fix (D-017); C1, strategy book, diagnosis, error map; verdict: no T-profit signal |
| T3b | Interactive notebook section + Drive cached env | merged | [#43](https://github.com/yuosef772424/nig-timenet-torch/pull/43) (c026624) | Arabic `main.ipynb` (34 cells), Drive sync both ways, env tar; D-016; not run on real Colab (B-048) |
| G1 | Test environments E1-E3 | merged | [#46](https://github.com/yuosef772424/nig-timenet-torch/pull/46) | `data_io/synthetic.py` (E1, exact oracle moments, `data_io/walk_max.py`), `data_io/environments.py` (E2, E3), `nig-envs`; D-018; oracle T-sharp 1 / 1 / 0; a symmetric head with the TRUE mean / sd covers 0.88 on high / low (B-056) |
| G2 | Gate tooling + gate job | merged | [#48](https://github.com/yuosef772424/nig-timenet-torch/pull/48) | `trainer/grad_report.py` (A1-A7), `eval/gates.py`, `workflow/gate_run.py`, entry `gates`, `gate_check` in the baseline job; D-019; local CPU run, baseline 3 seeds: E1 B pass, A / C fail (mu shrink B-062, class heads starved B-060), E2 canary pass, E3 B / C fail |
| DS1 | Deep side check (owner task 2026-11-08, read-only) | PR open | [#49](https://github.com/yuosef772424/nig-timenet-torch/pull/49) | `eval/deep_side.py`, `docs/results/deep-side-001.md`; D-020; verdict no: IC 0.016 / 0.015, AUC CI has 0.5, the last-bar asymmetry beats it, net of close mu ~0; B-065 closed |
| FX1 | Training switches + taps + E1 fix switches + bundle (owner 2026-11-09) | PR open | [#50](https://github.com/yuosef772424/nig-timenet-torch/pull/50) | D-021, D-022; `docs/results/e1-fixes-001.md`: BN moving stats (B-062) and ordered-means coupling (B-063) verified on E1, switches off by default |
| RW1 | `restore(weights="last")`: read the last finished epoch, not the best (owner 2026-11-10) | PR open | | D-023; default `best` unchanged; new run: `train.early_stop.restore_best_weights=False`; not run on real data |
| EX1 | `split.exclude_coins` + main.ipynb experiment switches (FX1_BUNDLE, FX1_NIG_SCALE, KEEP_LAST_WEIGHTS, LOAD_WEIGHTS) (owner 2026-11-10) | PR open | | D-024; defaults unchanged; after the time split; not run on real data / Colab |
| H7 | Register and review tooling (fourth R-REVIEW) | todo | | counter derived, CSV columns, check-job vs colab-results, rmtree ratchet |
| TR1 | Trainer switches (clip, wd groups, EMA, init, timing, train gap) | todo | | B-018, B-007, HF-37 |
| G3r | Gates on the baseline configuration | todo | | 3 seeds |
| M1 / M2r | Memorization study: CPU part / short GPU arms | todo | | B-038; protocol section 5; r2 shows overfit after epoch 1 |
| TR2 | Input pipeline + step time | todo | | B-008 |
| T5r-s1 | Baseline seed 1 | todo | | job `baseline-001-s1` |
| T6 | Switches: T6a (1-3), T6b (4-6), T6c (architecture, evidence-driven) | todo | | |
| T7r | Gates, then baseline vs all switches (`all-001`) | todo | | needs T5r, T6 |
| T8 | Ablate or diagnose, fix, propose default | todo | | needs T7r |
| D1 / D2 | Model doctor (checks / attribution + run) | todo | | protocol section 6 |
| F1-F5 | Reproducibility + version, walk-forward (rolling half of I4), model card, final evaluation (owner asks), release | todo | | F4 only on the owner's request (C5) |

Owner decision 2026-11-07: the goal is **T-profit** (profit after costs, default 0.0008 per side, from any use of the outputs; T-skill / T-conf / T-sharp are means). **O-1 CLOSED**: close-head T-sharp is informational only. Direction suspended for high/low. Owner decisions open: **O-2** F2 walk-forward in or out; **O-3** model doctor before T7r instead of after T8; **O-4** I7 stays closed (reopen: confidence gate fails for every method); **O-5** 3 vs 2 seeds per arm in T7r; **O-6** holdout definition for F4; **O-7** which runs showed the memorization (run ids / settings).
