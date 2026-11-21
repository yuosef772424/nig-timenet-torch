# Handoff 2026-11-19: LS1, a loss per trade side (side-payoff and upper-quantile heads)

## What was asked (owner, 2026-11-19)
Test improving the loss function, from the proposals of the previous session: keep mu on the symmetric log loss;
(a) the closed form e^(mu + s^2/2) - 1 (H1a, no training); (b) one head per side trained on the real payoff (long
e^l - 1 - cost, short 1 - e^l - cost) with a small weight or a partial stop of its gradient into the trunk; (c) a
quantile (pinball) head for the upper tail, to veto shorts into likely squeezes. Before that: merge the open PRs into
main (done, #69) and work on a new branch (`claude/loss-side-heads`).

## What was built (D-040, off by default)
- `model/heads.py`: head kinds `side_payoff` (outputs `y_close_long_payoff`, `y_close_short_payoff`) and
  `upper_quantile` (`y_close_upper_q`); `trainer/tasks.py`: tasks `side_payoff` and `pinball`.
- Model options `aux_heads` / `aux_trunk_grad` (0 = read-out); `TrainSettings.aux_loss_weight` / `aux_quantile`;
  battery columns `pay_long`, `pay_short`, `q_up`. Tests: `tests/trainer/test_aux_heads.py`.
- Pre-registration (written before any number): `docs/audit/2026-11-19-loss-side-heads/PREREGISTERED.md`.

## What runs where (Colab L4, started 2026-11-19 ~17:29 UTC, commit b8e0051)
Smoke `rolling-ls1-smoke-001` ran first on the same commit: OK, 2 folds in 75 s, the new columns in the battery
tables, q_up coverage 0.90 (rule E5).

| process | arm | outputs (Drive `MyDrive/nig_timenet_runs/`) |
|---|---|---|
| `rolling-ls1-a0-001` | A0 baseline | `rolling-ls1-a0-001/out_s2`, then `out_s3` |
| `rolling-ls1-r-001` | R: side_payoff + upper_quantile, aux_trunk_grad 0 | `rolling-ls1-r-001/out_s2`, `out_s3` |
| `rolling-ls1-s-001` | S: side_payoff, aux_trunk_grad 0.3 | `rolling-ls1-s-001/out_s2`, `out_s3` |

- Each process runs seed 2 and then seed 3 (seeds outer); configs `/content/cfg/rolling-ls1-<arm>-001-s<seed>.json`
  (the job file with `seeds` = [s], `run_dir` = `.../train_s<s>`). If cut: start the same command again, it resumes.
- Poll: `colab_ops rt`, `colab_ops out <pid>`. The bridge URL and token come from the owner's Colab cell; they are
  never written to a file.

## What is next
1. When all six runs finish: read them with the pre-registered rules (P1, P2, P3, P1L, H): script
   `docs/audit/2026-11-19-loss-side-heads/read_ls1.py` on the fold folders.
2. Write `docs/results/ls1-side-heads-001.md`, with the failure reading of every test that fails (rule C7).
3. E1 known-answer results: `docs/audit/2026-11-19-loss-side-heads/E1_OUTPUT.txt`.

## Update 18:30 UTC
- E1: the side-payoff heads fail E1-a / E1-b on every seed (also W = inverse-variance weights, also 10 epochs); the
  quantile head passes; a ridge on the trunk reaches the closed form (amendments 1-2 of PREREGISTERED.md). W is not run.
- Seed 2 of A0 / R / S read (`read_ls1.py --seeds 2`): P1, P1L, P3, P4 already fail on seed 2; P2 passes on seed 2.
- Gap found (owner question "a gap in the settings or the training"): the run path never passed an lr schedule, every
  run trained at a constant lr; best epoch = last of 8 in 4 / 7 A0 folds. New switch `TrainSettings.lr_schedule`
  (D-040). E1 gate: cosine beats constant on 3 / 3 seeds (`E1_LR_COSINE.txt`). Smoke `rolling-lr1-smoke-001` OK.
- Running since 18:29 UTC from the second checkout `/content/nig_lr1` (commit 2764907; PYTHONPATH =
  `/content/nig_venv/lib/python3.13/site-packages:/content/nig_lr1/src`): `rolling-lr1-cos-001` seeds 2 and 3, one
  process each, outputs `rolling-lr1-cos-001/out_s{2,3}`. Compare with `rolling-ls1-a0-001` (same config and seeds).
