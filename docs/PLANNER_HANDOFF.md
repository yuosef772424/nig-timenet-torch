# Planner handoff (rewritten 2026-11-03, after the first GPU runs)

The planner conversation plans, reviews and merges; step sessions or ONE background agent (owner: one at a time,
Sonnet, lean: the weekly usage budget is nearly spent) do the work. Reply to the owner in Arabic. Merge with
`merge_method=merge` only, CI green on the head. Owner standing permission (2026-11-03): small run fixes the planner
opens for the baseline job are merged automatically when CI is green, and the queue is updated without asking; step
PRs (T6+) and anything large still go to the owner first.

## Repositories
| Repo | Role | Default branch |
|---|---|---|
| `yuosef772424/nig-timenet-torch` (this repo) | ACTIVE: all model work. Rules `docs/RULES.md`, plan `docs/ROADMAP.md`, status `docs/STATUS.md`. | `main` |
| `yuosef772424/crypto-signal-prediction` | FROZEN reference (TF). Only `tools/export_torch_artifact.py` changes there (T2/T3). | `master` |

## State
- Merged in torch: R0 #14, T1 #16, T2 #17, T3 #18, T4 #19, T5r job #20, fixes #21-#23. TF: #62 (D-013 defaults),
  #65 + #66 (export tool, `.pkl.gz` input). TF #64 (TF-final settings catalog) still OPEN with conflicts, parked.
- Data: artifact `MyDrive/crypto_model/torch_artifacts/preprocessing_output_1` from
  `preprocessing_output_latest.pkl.gz` (TF main data, 4h): 2,520,913 windows, 535 coins, sealed 183/183 removed,
  max_bar_time 2026-09-26T20:00Z, data fingerprint `c572b1b36960`. The artifact has `split_dates = None`: the job
  splits 80/10/10 by time quantiles (TF `raw_quantile`; TF's default is `kept_share`).
- All 183 sealed coins were inside the TF main data, so TF results (profile-004 etc.) are not a clean holdout (B-022).
- Colab: ONE notebook `notebooks/colab_nig_timenet_torch.ipynb` (link:
  https://colab.research.google.com/github/yuosef772424/nig-timenet-torch/blob/main/notebooks/colab_nig_timenet_torch.ipynb),
  T4 GPU, Run all. The GitHub token at `MyDrive/crypto/secrets/github_token.txt` now covers both repos (owner
  edited it). It runs the job in `job.json` on branch `colab-queue`; results go to Drive
  `MyDrive/nig_timenet_runs/<job_id>/` then branch `colab-results`.
- Queue now: `baseline-001-s0`, 5 epochs (owner), `rerun: true`, repo_ref = the merge of the latest fix. Runs so far
  failed on: hard-coded 1h input (#21), `split_dates` None (#22), cuDNN refusing an eval-mode RNN backward in the T4
  trace (#23). One epoch did train on the GPU before the last failure. No result yet.
- How to queue: edit `job.json` on `colab-queue` (repo_ref = 40-hex merged commit, `rerun: true` to repeat a
  job_id), validate with `nig_timenet_torch.colab.job.parse_job`, push. Resume across a code change is REFUSED by
  design (RunIdentity holds `code_sha`): a new commit means a fresh start of that run_dir (or use a new run_dir).

## Next
1. Read the first full result (`colab-results/runs/baseline-001-s0/`: `baseline_summary.json` with epoch seconds,
   `health_report.md` on validation, `test_report.json` on the test split). Write `docs/results/baseline-001.md`.
2. Owner asked for: an interactive notebook section (settings cell -> train -> test report shown inline) and a Drive
   cached Python env (prompt "T3b" written earlier: venv `--system-site-packages`, one tar on Drive keyed by
   Python + torch + requirements hash). Neither is built.
3. Then seed 1, then T6 (switches). R-REVIEW is due (10 PRs since the last one).

## Facts worth keeping (do not re-derive)
- PyTorch port: forward/loss/gradient parity with TF proven; weight init is torch default (results differ from TF).
- TF "hit rate" (sign of pred-entry) is ~99 % for high/low with any model: meaningless; torch T-conf ranks error
  within coin instead.
- TF profile-004 (4h, h2, 3 features, `logret_std`): model+GBM rank average +0.153 %/period, CI [+0.082, +0.220],
  on data that contained the sealed coins.
- Owner preferences: improve the model first; every improvement is a switch, one test, then ablate; fast tests; F5
  suggestions every step; self-contained prompts (the owner clears conversations); very lean usage now.

## How the planner reviews a step PR
Read PR body + CI; fetch the branch and read the diff for scope (B1 behaviour changes listed), incomplete class
fixes (B3), defaults without a decision file (tests/test_decisions.py), claims without evidence; run ruff + pytest
locally when torch is available (a venv with the CPU wheel works); report in Arabic with a verdict.
