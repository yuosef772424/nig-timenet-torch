# Planner handoff (rewritten 2026-11-03, after the first GPU runs; State / Next updated 2026-11-07 by H0c)

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
- Merged in torch: R0 #14, T1 #16, T2 #17, T3 #18, T4 #19, T5r job #20, fixes #21-#24, hardening H0 #26 (audit),
  H0b #27 (PR #25 evaluation), H0c #28 (this ROADMAP + `docs/MODEL_DEV_PROTOCOL.md`, rule C6). PR #25
  (`feat/ar-stream`) stays open, not merged: ported piecewise in I1, I2, I3, I5; I4 split; I7 not opened. TF: #62
  (D-013 defaults), #65 + #66 (export tool, `.pkl.gz` input). TF #64 (TF-final settings catalog) still OPEN, parked.
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
1. Campaign (one builder session each, merged before the next): `T4b -> I1 -> H2 -> H1 -> I5 -> I2 -> H3 -> H4 -> I3 ->
   H5 -> H6` (`docs/hardening/PLAN.md` "Final campaign order"). T4b first: T-sharp per head vs the predicted move.
2. Then the ROADMAP model plan: T5r reading (check `reg_target_mode` first) -> T3b -> G1 -> G2 -> TR1 -> G3r -> M1 ->
   M2r -> TR2 -> T5r-s1 -> T6a/b/c -> T7r -> T8 -> D1 -> D2 -> F1 -> F2 -> F3 -> (F4 on the owner's request) -> F5.
   Builder steps take the cheaper agent model; G3r, M1, M2r, T7r, T8, F4 are research steps.
3. Every model step obeys `docs/MODEL_DEV_PROTOCOL.md`: gates A (gradient), B (confidence), C (uncertainty) on E1
   synthetic / E2 shuffled / E3 small real, 3 seeds, before any full training; direction only on the close head.
4. Owner decisions open (STATUS bottom): O-1 CLOSED 2026-11-07 (close-head T-sharp informational only; goal is T-profit), O-2
   F2 in/out, O-3 doctor earlier, O-4 I7, O-5 seeds per arm, O-6 holdout definition, O-7 memorization run ids.
5. The T5r queue may keep running during the campaign; read its result in step T5r (after H6) unless the owner asks
   earlier (then T4b must be merged first).

## Facts worth keeping (do not re-derive)
- PyTorch port: forward/loss/gradient parity with TF proven; weight init is torch default (results differ from TF).
- T-sharp (owner 2026-11-07) = per head, predictive std (price) < |pred_price - last_close| in >= 80 % + coverage
  75-85 %. H0c oracle probe (true conditional distribution, 4h, horizon 2): share 1.00 high/low, 0.00 close; a
  calibrated close head would need R^2 >= 0.94 (decision O-1).
- AdamW weight decay in the baseline job is a no-op in effect (lr 1e-3 x wd 1e-4 = 1e-7 per step), as in TF (F-0029).
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
