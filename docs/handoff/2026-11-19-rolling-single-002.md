# Handoff 2026-11-19: SS1 (short side + day scorecard) and the job rolling-single-002

Status: closed (SS1 #68; `docs/results/rolling-single-002.md`, `docs/results/short-side-001.md`)

## What was asked (owner, 2026-11-19)
1. Read the past experiment reports and what was learned: `docs/audit/2026-11-19-lessons/LESSONS.md`.
2. Make the evaluation weigh confidence, the distribution of losses over days, the size of the uncertainty and the
   top rows: `eval/battery_scorecard.py`, wired into the battery (D-037).
3. A report on why "top short" lost in every run (is the short loss exponential?): `docs/results/short-side-001.md`.
4. Launch a GPU training on the Colab bridge the owner opened.

PR: #68 (draft), branch `claude/reports-analysis-evaluation-jv4sqr`.

## What runs where (Colab L4, started 2026-11-19 ~12:37 UTC, commit cc3b011)
| process | what | outputs (Drive `MyDrive/nig_timenet_runs/`) |
|---|---|---|
| `rolling-single-002-s2` | `jobs.rolling`, config of `docs/jobs/rolling-single-002.json` with seeds [2] | `rolling-single-002/out_s2`, checkpoints `rolling-single-002/train_s2` |
| `rolling-single-002-s3` | the same with seeds [3] | `rolling-single-002/out_s3`, `rolling-single-002/train_s3` |
| `reread-rolling-single-001` | `battery_from_folder` on the 14 old folds (real VAL thresholds, new sections) | `rolling-single-001/out/fold_*/ss1/battery_report.txt` |

- Smoke `rolling-single-smoke-001` ran first on the same commit (fresh run dir `rolling-single-smoke-ss1`): 2 folds OK,
  the four new sections are in its `battery_report.txt` (rule E5).
- The job is split into one process per seed (GPU at 93 % with both). Each writes its own `rolling_summary.*` and
  `pooled_days_seed_<s>/`.
- Expected time: about 2-3 h (the 001 run took about 75 min for 14 folds x seeds in one process). The rolling entry
  does not read `--max-minutes`. If it is cut, start the SAME command again: it resumes from its run dir.
- Poll: `colab_ops rt`, then `colab_ops out <pid>` (the pids are printed by `colab_ops start`). The bridge URL and
  token come from the owner's Colab cell. They are never written to a file.

## What is decided
- The accept rule is fixed in `docs/audit/2026-11-19-short-side/PREREGISTERED.md`: does LONG top 20 % confidence pass
  the scorecard verdict on BOTH seeds 2 and 3 (pooled days)?

## What is next
1. Read `out_s2` / `out_s3` (`fold_*/battery_report.txt`, scorecard sections; `pooled_days_seed_*`), and pool the
   candidate row over the 7 folds per seed with `docs/audit/2026-11-19-short-side/read_short_side.py`. That script
   needs the tables copied as `test_f{k}_s{s}.parquet` and `val_f{k}_s{s}.parquet`; on Colab they are readable
   directly.
2. Write `docs/results/rolling-single-002.md`, with the failure reading if the candidate fails (rule C7).
3. Read the 14 re-read reports (`ss1/`): does the short-side reading of short-side-001 hold with the real VAL
   thresholds?

## Outcome (2026-11-19, ~13:30 UTC)
Both seed processes finished; the 14 old folds were re-read (`ss1/` reports). Read in `docs/results/rolling-single-002.md`:
the candidate fails (CI contains 0; negative without the best 5 % of days), no H1 / H3 rule passes (the kept shorts
lose), and the short exp() reading of short-side-001 replicates. The next hypotheses (N1-N4) are in that note.
