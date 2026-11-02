# Planner handoff (written 2026-11-02 before /clear of the planner conversation)

The planner conversation plans, reviews and merges; step sessions (fresh conversations) do the work. It replies to
the owner in Arabic. It does not do step work itself and does not spawn agents unless the owner asks. It merges a PR
only when the owner says so, always with `merge_method=merge` (never squash), and only when CI is green on the head.

## Repositories
| Repo | Role | Default branch |
|---|---|---|
| `yuosef772424/nig-timenet-torch` (this repo) | ACTIVE: all model work (PyTorch). Rules `docs/RULES.md`, plan `docs/ROADMAP.md`, status `docs/STATUS.md`. | `main` |
| `yuosef772424/crypto-signal-prediction` | FROZEN reference (TF): history, failure registry, decisions, holdout lock, data pipeline. Its own plan: `docs/research/hardening/PLAN_2026-11.md` + `STATUS.md`. | `master` |

Planner session branch in the TF repo: `ccr-888d8b52-lzcqoe` (its PRs #59, #63 are merged).

## State at handoff
- Torch repo: PRs #12 (RULES + ROADMAP + CLAUDE.md + bottlenecks) and #13 (rules G1-G8 growth ratchets, F5
  suggestions) MERGED. Next step: **R0** (make the rules checkable). Not started.
- TF repo, merged: #59 (plan + D-012), #60 (S01 registries), #61 (S02 normalize_tf), #63 (bottleneck register + model
  track M0-M3).
- TF repo, OPEN: **#62** (S03 profiler, branch `ccr-facfbac9-l1ortv`). It also carries 3 owner-approved default changes:
  - `suspended_targets` default `()`;
  - `PREDICT_BATCH_SIZE` 2048;
  - `rf_max_samples` 50000.

  It also adds GPU tools and `tools/signal_diagnostics.py`. Before merge it still needs:
  - the D-013 decision file;
  - B-007+ rows in `bottlenecks.csv`;
  - an updated title and description.

  The owner was given the prompt to send to the S03 session (after `/clear`). Then the planner reviews #62 and
  merges it on the owner's word.
- TF repo, pending task given to the owner (prompt already written): "TF-final", meaning `tools/settings_catalog.py`
  (all settings keys + accepted values) plus `workflow/presets.py recommended_settings()`. That preset combines the
  options proven useful before (calibration and others) with the new untried options. Defaults are NOT changed. It
  is selected with one switch in `main.ipynb`.
- Colab job `profile-005` ran (35 min) but its publish failed: `askpass.sh` vanished from `/tmp`. The bundle is only on
  Drive at `MyDrive/crypto_runs/profile-005`. The TF plan's M0 has the fix and a `republish` command. The torch roadmap
  rule E2 covers the same failure.

## Facts worth keeping (do not re-derive)
- **PyTorch port (subagent review 2026-11-02):** 236/236 tests pass. Forward, loss and gradient parity with TF are
  proven. Missing: the epoch loop, schedules (only the start value is used), early stopping, checkpoint/resume,
  diagnostics, and a `.weights.h5` converter. Weight init is the torch default, so trained results will differ from TF.
  The port is ahead of TF on these seams: selection metric, calibrator, `input_skip` + probe, PredictionTable,
  SplitPlan, rolling/staleness, ModelContract. The real TF default model is the GRU encoder + level passthrough with
  anti-memorization (`ORACLE_REVIEW.md` V1).
- **TF profile-004** (4h, horizon 2, stride 1, 3 features H/L/C, `logret_std`, test 2026-06-14 to ~09-26, 2 epochs,
  1 seed):
  - The market-neutral model+GBM rank-average got +0.153 %/period net, CI [+0.082, +0.220]. The model alone and the
    GBM alone were not significant.
  - The owner wants the model IMPROVED first. Strict validation (S26 in TF, "Later 3" in the torch roadmap) happens
    only when the owner asks. Do not over-hedge.
- **Holdout lock:** the TF main pipeline never applied `holdout_guard`. The dataset holds 718 coins vs 677 allowed, so
  sealed coins are probably inside. Torch rule D1 (data only via an artifact whose manifest records the exclusions)
  and step T2 handle this going forward.
- **"Hit rate" in TF reports** = sign(pred - entry) == sign(true - entry). It is ~99 % for high/low for any model, so
  it is meaningless. The torch T-conf target uses error ranking within coin instead.
- **TF runs on Colab were CPU-only** (`requirements-ci.txt` pins tensorflow-cpu). The fix sequence was proven in
  `gpu-diag-001`. Torch on Colab should get the GPU out of the box; T3 verifies it.
- **Owner preferences:**
  - improve the model first;
  - build every improvement as a switch, run ONE test, then ablate/diagnose;
  - fast tests;
  - every step reports improvement suggestions and unclear/stale places (rule F5);
  - prompts for step sessions must be self-contained, because the owner `/clear`s conversations to save tokens.

## Prompt template for a step session (torch repo)
```
نفّذ الخطوة <ID> من docs/ROADMAP.md.
اقرأ CLAUDE.md ثم docs/STATUS.md ثم docs/RULES.md ثم قسم <ID> فقط من ROADMAP.
ابدأ فرعك من آخر main. لا تنشئ أي وكيل. اختبارات سريعة فقط.
حدّث صف <ID> في STATUS.md وfriction_log.csv وbottlenecks.csv وsuggestions.csv في نفس طلب الدمج.
```

## How the planner reviews a step PR
1. Read the PR body and CI (GitHub MCP).
2. Fetch the branch and read the diff, focused on:
   - scope (behaviour changes listed? rule B1);
   - incomplete class fixes (rule B3: grep for remaining instances);
   - new defaults without a decision file;
   - claims without evidence.
3. Report to the owner in Arabic with a merge / do-not-merge verdict and a ready-to-paste fix prompt when needed.
4. Merge only on the owner's word.
5. After R-REVIEW-relevant merges, count PRs since the last review (STATUS line "PRs since last R-REVIEW").
