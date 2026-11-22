# STATUS
> Reading order: `CLAUDE.md`. Each step's own tests green before the next step starts. Update your row in the same PR (`python -m tools.close_step`);
> obstacles go under `## Friction` of the PR description. Keep this file under 60 lines.
Closed rows, the port record, the finished hardening campaign 2026-11, the done list and the GPU results are in `docs/status_archive.md` (append-only). All docs: `docs/INDEX.md` (generated). Open judgment calls: `friction_log.csv` rows marked "owner to confirm". States: `todo` / `in progress` / `PR open` / `merged` / `closed (rejected)` / `skipped (reason)`.

## Hardening 2026-11b (`docs/hardening/PLAN-2026-11b.md`; audit `docs/hardening/audit-2026-11b/`, 177 findings)
Launched 2026-11-21 (owner: structure and clarity). HB1 (correctness: whole timestamps in prediction / health batches, early refusals, batches per epoch) done in MG4 #74 (D-041). HB2 (run identity and resume safety, D-042): its row below. Next: HB3 evaluation (before N8); HB7 docs and HB8 tests / tools can run in parallel. HB3-HB8: todo (table in the plan).

## Roadmap (`docs/ROADMAP.md`, rules `docs/RULES.md`, protocol `docs/MODEL_DEV_PROTOCOL.md`)
PRs since last R-REVIEW: 1 (#74 the review itself, merged; fifth review 2026-11-21 in MG4, `docs/reviews.md`, covered #48-#73; the counter had stayed at 5 for 14 merges; it is hand-kept until H7 derives it).

| Step | Title | State | PR / job | Note |
|---|---|---|---|---|
| HB2 | Hardening 2026-11b HB2: run identity and resume safety (split / selection / staged in the identity, rolling study hash, one deadline for every entry) (owner 2026-11-21) | PR open | [#75](https://github.com/yuosef772424/nig-timenet-torch/pull/75) | branch `ccr-bb6e83a4-2gwtez` |
| HB3 | Hardening 2026-11b HB3: evaluation, one implementation per trading rule and helper, honest names (owner 2026-11-21) | in progress |  | branch `ccr-bb6e83a4-2gwtez` |
| HB4 | Hardening 2026-11b HB4a: model and trainer clean-ups (dead options and the Keras PanelModel out, one helper each, cards) (owner 2026-11-21) | in progress |  | branch `ccr-bb6e83a4-2gwtez` |
| HB6 | Hardening 2026-11b HB6a: workflow corrections (FitResult.finished, runner catches every step error, gates refuse cross-section configurations, one code_sha / paths / default overrides, dead modules out, cards) (owner 2026-11-21) | in progress |  | branch `ccr-bb6e83a4-2gwtez` |
| H7 | Register and review tooling (fourth R-REVIEW) | todo | | counter derived, CSV columns, check-job vs colab-results, rmtree ratchet; fifth review adds 12 suggestions (RunIdentity with split settings, PR-section CI check, merged-row archive) |
| TR1 | Trainer switches (clip, wd groups, EMA, init, timing, train gap) | todo | | B-018, B-007, HF-37 |
| G3r | Gates on the baseline configuration | todo | | 3 seeds |
| M1 / M2r | Memorization study: CPU part / short GPU arms | todo | | B-038; protocol section 5; r2 shows overfit after epoch 1 |
| TR2 / T5r-s1 | Input pipeline + step time / baseline seed 1 | todo | | B-008; job `baseline-001-s1` |
| T6 | Switches: T6a (1-3), T6b (4-6), T6c (architecture, evidence-driven) | todo | | |
| T7r | Gates, then baseline vs all switches (`all-001`) | todo | | needs T5r, T6 |
| T8 | Ablate or diagnose, fix, propose default | todo | | needs T7r |
| D1 / D2; F1, F3-F5 | Model doctor (checks / attribution + run); reproducibility + version, model card, final evaluation (owner asks), release | todo | | doctor: protocol section 6; F4 only on the owner's request (C5) |

Owner decision 2026-11-07: the goal is **T-profit** (profit after costs, default 0.0008 per side, from any use of the outputs; T-skill / T-conf / T-sharp are means). **O-1 CLOSED**: close-head T-sharp is informational only. Direction suspended for high/low. Owner decisions open: **O-2** F2 walk-forward in or out; **O-3** model doctor before T7r instead of after T8; **O-4** I7 stays closed (reopen: confidence gate fails for every method); **O-5** 3 vs 2 seeds per arm in T7r; **O-6** holdout definition for F4; **O-7** which runs showed the memorization (run ids / settings).
