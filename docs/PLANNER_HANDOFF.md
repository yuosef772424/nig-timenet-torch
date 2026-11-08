# Planner handoff (rewritten 2026-11-07 by H6, after the hardening campaign; for a fresh planner)

> Manager sessions (owner clears conversations): read `docs/hardening/MANAGER_HANDOFF.md` first.

The planner conversation plans, reviews and merges; step sessions or ONE background agent at a time do the work. Reply to
the owner in Arabic. Merge with `merge_method=merge` only, CI green on the head. Owner standing permissions: small run
fixes the planner opens for the baseline job are merged when CI is green; builders may self-merge a green docs PR when the
owner authorised it for that campaign; the planner may queue Colab jobs when the owner says so. Step PRs that change model
behaviour (T6+) and anything large still go to the owner first. Agents work in a separate clone and unsubscribe from their
PRs right after opening them.

## Repositories
| Repo | Role | Default branch |
|---|---|---|
| `yuosef772424/nig-timenet-torch` (this repo) | ACTIVE: all model work. Rules `docs/RULES.md`, plan `docs/ROADMAP.md`, status `docs/STATUS.md`, map `docs/MAP.md`. | `main` |
| `yuosef772424/crypto-signal-prediction` | FROZEN reference (TF). Only `tools/export_torch_artifact.py` changed there (T2/T3). | `master` |

Other branches: `colab-queue` (job.json), `colab-results` (run folders), `feat/ar-stream` (PR #25, open, reference only: ported
piecewise; keep it as `archive/feat-ar-stream` when closed).

## State (2026-11-07)
- The hardening campaign is DONE: H0 #26, H0b #27, H0c #28, T4b #29, I1 #30, H2 #31, H1 #32, I5 #33, I2 #34, H3 #35, H4 #36 + #37,
  I3 #38, H5 #39, H6 (this PR). Tests 515 -> 749. Per-package outcome and finding status: `docs/hardening/PLAN.md` and
  `AUDIT-2026-11.md` "Outcome". Decisions D-007..D-015 (`docs/decisions/`).
- What the campaign gave: a job that reads the artifact (targets, anchors, `reg_target_mode`), strict configs (unknown keys
  raise), registries for encoders / heads / tasks, run steps in `workflow/run.py` (`jobs/baseline.py` is a 107-line entry),
  the evaluation battery in every run (profit per use of the outputs, cost 0.0008 per side, p_dir as the single
  direction), a native data pipeline (`data/`, parity with TF), cards + a generated `docs/MAP.md` (rule B7), smoke job
  before long jobs (rule E5), `tests.pin_diff` in CI (pin loophole closed), ratchets G1 0, G2 5 (honest, B-046), G3 0, G4 0,
  G7 0, C2 0, G8 1 (`model/panel.py`).
- Data: artifact `MyDrive/crypto_model/torch_artifacts/preprocessing_output_1` (TF main data, 4h): 2,520,913 windows, 535
  coins, sealed 183/183 removed, max_bar_time 2026-09-26T20:00Z, fingerprint `c572b1b36960`. `split_dates = None`: the job
  splits 80/10/10 by time quantiles. All 183 sealed coins were inside the TF main data, so TF results are not a clean
  holdout (B-022). B-020 (bar OPEN time) is unproven.
- GPU results on branch `colab-results` (read with `git show origin/colab-results:<path>`):
  - `runs/baseline-001-s0/` (2026-11-03, @55e2db4, 5 epochs): pre-T4b, history only. `runs/trial-002/` (2 epochs, owner trial).
  - `runs/baseline-smoke-001/` (@44f6e71): smoke OK (rule E5).
  - `runs/baseline-001-s0-r2/` (@44f6e71, 5 epochs, T4, 972 s, ~169 s/epoch; windows 2,015,804 / 239,297 / 239,772): val best
    at epoch 1 (-5.062 -> -4.947 by epoch 5) while train loss keeps falling (-5.75 -> -6.12): overfit after epoch 1. Test
    at the best epoch: rank IC high 0.393 / low 0.436 / close 0.057 (shuffled ~0); class AUC high 0.757 / low 0.765 (information
    only) / close 0.512; confidence decile Spearman -0.81 / -1.00 / -0.99 (meets T-conf); T-sharp share high 0.837 / low 0.889
    (>= 0.80 met) but 80 % coverage 0.892 / 0.932 / 0.875 (> 0.85: intervals too wide); close std / move 24.5 (informational,
    O-1). NOT known: whether high/low IC beats the last-bar-range volatility baseline (rule C1), and any profit number: the
    battery was wired in I3, after this run.
- Open PRs/branches that matter: PR #25 (reference). TF #64 (settings catalog) parked.

## Owner decisions (2026-11-07)
- The goal is PROFIT by any use of the outputs (T-profit is the north star; T-skill / T-conf / T-sharp are means and may not
  reject a configuration alone). O-1 CLOSED: close-head T-sharp is informational only. Direction is suspended for high/low.
- Cost default 0.0008 per side (the owner may correct it; D-009).
- PR #25 stays open as a reference; integration was by porting (I1, I2, I3, I5 done; I4 split: rolling -> ROADMAP F2, panel
  parked; I7 closed, reopen only if the confidence gate fails for every method).
- Model steps after H6 run on the STRONGER agent model; hardening-type work on the cheaper one. Rules exist to improve the
  project, not to restrict it.
- Open owner decisions (STATUS bottom): O-2 F2 in/out, O-3 doctor before T7r, O-4 I7, O-5 seeds per arm, O-6 holdout
  definition, O-7 which runs showed the memorization.

## How to queue a Colab job
1. The job file lives in `docs/jobs/<job_id>.json` (reviewed in a PR). `repo_ref` in the file is a placeholder of 40 zeros.
2. To queue: copy the file to `job.json` at the root of branch `colab-queue`, replace `repo_ref` with the 40-hex merge commit
   to run, and set `rerun: true` only to repeat an existing job_id. Validate BEFORE pushing:
   `python -m nig_timenet_torch.colab check-job job.json` (prints `OK: job ... @ <sha> entry ... device ...`; a bad ref, an
   entry without CONFIG, or a job/config device mismatch is refused).
3. SMOKE FIRST (rule E5): any job with `max_minutes` > 30 on a commit that has not run on the GPU is preceded by
   a smoke job file of that entry whose `job_id` and `run_dir` were never used (`runs/<job_id>/` absent on
   `colab-results`; next free: `docs/jobs/baseline-smoke-002.json`; a used id is refused by `check_rerun`, a used run_dir
   resumes another commit's checkpoint and is refused) (same entry, `max_rows` 20000, 1 epoch, a few minutes) on that commit. Read its
   `baseline_summary.json` before queueing the long job.
4. A code change means a NEW `run_dir` (resume across a code change is refused: `RunIdentity` holds `code_sha`); the long job's
   `run_dir` and `job_id` change together, e.g. `baseline-002-s0`.
5. The owner presses Run all in `notebooks/colab_nig_timenet_torch.ipynb` (T4); results appear on `colab-results` under
   `runs/<job_id>/` (`bundle_summary.md` first). A failed publish: `python -m nig_timenet_torch.colab republish <bundle_dir>`.

## Next
1. Queue (when the owner says so): smoke job `baseline-smoke-002`, then a FRESH baseline at current `main`: `baseline-002-s0` (config of
   `docs/jobs/baseline-001-s0.json`, new run_dir). It is the first run with the battery. (No recompute-from-checkpoint entry
   exists; building one costs more than the ~16 minutes of a fresh run.)
2. Step T5r (reading, stronger model): `docs/ROADMAP.md` T5r computes the volatility baseline for high/low, reads the battery
   and writes `docs/results/baseline-001.md`.
3. Then the ROADMAP model plan: T3b -> G1 -> G2 -> TR1 -> G3r -> M1 -> M2r -> TR2 -> T5r-s1 -> T6a/b/c -> T7r -> T8 -> D1 -> D2 ->
   F1 -> F2 -> F3 -> (F4 on the owner's request) -> F5. Every model step obeys `docs/MODEL_DEV_PROTOCOL.md` (gates before
   any full training, 3 seeds, direction only on the close head).
4. The next R-REVIEW is due after 5 more merged PRs (STATUS counter).

## Facts worth keeping (do not re-derive)
- PyTorch port: forward/loss/gradient parity with TF proven; weight init is torch default (results differ from TF).
- T-sharp (owner) = per head, predictive std (price) < |pred_price - last_close| in >= 80 % + coverage 75-85 %. Oracle probe
  (true conditional distribution, 4h, horizon 2): share 1.00 high/low, 0.00 close.
- AdamW weight decay in the baseline is a no-op in effect (lr 1e-3 x wd 1e-4), as in TF (F-0029).
- TF "hit rate" is ~99 % for high/low with any model: meaningless; torch T-conf ranks error within coin instead.
- TF profile-004 (4h, h2, 3 features, `logret_std`): model+GBM rank average +0.153 %/period, CI [+0.082, +0.220], on data that
  contained the sealed coins.
- Owner preferences: improve the model first; every improvement is a switch, one test, then ablate; fast tests; F5 suggestions
  every step; self-contained prompts (the owner clears conversations); lean usage.

## How the planner reviews a step PR
Read PR body + CI; fetch the branch and read the diff for scope (B1 behaviour changes listed), incomplete class fixes (B3),
defaults without a decision file, claims without evidence; run ruff + pytest locally (CPU torch venv); report in Arabic with a
verdict.
