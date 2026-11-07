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
PRs since last R-REVIEW: 1 (H6 itself; third review 2026-11-07 in H6, `docs/reviews.md`, covered #33-#39; the next is due after 5 more merged PRs).
Done: R0 [#14](https://github.com/yuosef772424/nig-timenet-torch/pull/14), T1 [#16](https://github.com/yuosef772424/nig-timenet-torch/pull/16), T2 [#17](https://github.com/yuosef772424/nig-timenet-torch/pull/17), T3 [#18](https://github.com/yuosef772424/nig-timenet-torch/pull/18), T4 [#19](https://github.com/yuosef772424/nig-timenet-torch/pull/19), T4b [#29](https://github.com/yuosef772424/nig-timenet-torch/pull/29), the T5r job ([#20](https://github.com/yuosef772424/nig-timenet-torch/pull/20), fixes #21-#24). Data: artifact `c572b1b36960` (2,520,913 windows, 535 coins, sealed 183/183 excluded; B-020, B-022 open).
GPU results (branch `colab-results`): `baseline-001-s0-r2` (2026-11-07, @44f6e71, 5 epochs, T4, 972 s): best epoch 1 (val overfits, train loss keeps falling); test rank IC high 0.393 / low 0.436 / close 0.057; T-sharp share 0.84 / 0.89 but coverage 0.89 / 0.93 / 0.88 (intervals too wide); no battery yet (predates I3). Smoke `baseline-smoke-001` OK.

| Step | Title | State | PR / job | Note |
|---|---|---|---|---|
| T5r | Read the baseline (`docs/results/baseline-001.md`) | todo | | stronger model; fresh run `baseline-002-s0` at current main (smoke first, E5) gives the battery; r2 numbers are the reference |
| T3b | Interactive notebook section + Drive cached env | todo | | I3 done; rows 99, 121, 125 |
| G1 / G2 | Test environments E1-E3 / gate tooling + gate job | todo | | protocol sections 2-3 |
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
