# ROADMAP archive: step specs that are merged / done (moved verbatim from docs/ROADMAP.md, 2026-11-22; append-only, newest at the bottom)

PRs and results of these steps: `docs/status_archive.md` and the git history.

## Block 1 — hardening and integration — DONE 2026-11-07 (plan and per-package outcome: `docs/hardening/PLAN.md`; PRs #26-#39 in `docs/STATUS.md`)
Order: `T4b -> I1 -> H2 -> H1 -> I5 -> I2 -> H3 -> H4 -> I3 -> H5 -> H6`. T4b is a ROADMAP fix that goes first (below).
I4 is split: its rolling half becomes step F2, its panel half is parked (Later); I7 (ar-stream) is not opened (PLAN.md
"Final campaign order" gives the reasons and the reopen condition). The T4b text below is kept as the record of the fix
(merged, #29).

### T4b — T-sharp per head vs the predicted move (`fix` + `test`, S, builder) — FIRST, before I1
- `eval/target_report.py` `_sharp` (~L95-103): replace the range reference `pred_high - pred_low` by each head's
  predicted move `|decode(mu_head, state, anchor_head) - last_close|`; the std in price units stays
  `decode(mu + std) - decode(mu)`. `last_close` is passed explicitly (do not rely on the close head being trained);
  the job passes `lc[:, LAST_COLUMNS.index("last_close")]`. Report keys: `share_std_below_move` (replaces
  `share_width_below_range`), the zero-std baseline, `meets_target` per head = share >= 0.80 AND coverage in range;
  every head is judged, not only when high and low exist. `stats.metrics.width_below_share` is reused (C2).
- Mark the high / low class AUC in the report and `health_report.md` as "information only (direction suspended)".
- Synthetic test with a KNOWN answer: random-walk paths with known conditional mean / sd; the oracle prediction gives
  share 1.0 on high / low and ~0 on close; a zero std passes the share but fails coverage; a too-wide std fails share.
- Behaviour change: report output keys only (no training change); no recorded result used the old definition.
- Acceptance: the new tests green; `pytest -q` green; `health_report.md` column renamed.

Then I1 ... H6 per `docs/hardening/PLAN.md`.

### T5r — read the baseline (`docs`/`exp`, S-M, researcher: the stronger model, because it decides what to try next)
- Inputs that already exist on `colab-results` (read with `git show origin/colab-results:<path>`, do not copy): `runs/baseline-001-s0-r2/`
  (`baseline_summary.json`, `health_report.md`, `test_report.json`; 2026-11-07 @44f6e71, 5 epochs, T4, 972 s; its
  `reg_target_mode` is `return`, scale 1.0, but `price_norm_mode` `logret_std`: the labels are log(future / last_close),
  so the high / low decodes were WRONG until T5r (B-051, fixed with the `logret_close` codec; HF-13 was right)). Older: `baseline-001-s0`
  (@55e2db4, pre-T4b, history only), `trial-002`, smoke `baseline-smoke-001`. Known so far: best epoch 1 of 5 (val
  -5.062 -> -4.947 while train -5.75 -> -6.12: overfit after epoch 1, B-038); test rank IC high 0.393 / low 0.436 /
  close 0.057 (shuffled ~0); T-sharp share 0.84 / 0.89 (a B-051 misdecode; 0.012 / 0.272 after the fix, fourth R-REVIEW note) but 80 % coverage 0.89 / 0.93 / 0.88 (> 0.85: intervals too
  wide); confidence decile Spearman -0.81 / -1.00 / -0.99; ~169 s per epoch (median), 2,015,804 / 239,297 / 239,772
  windows. NOT yet known: whether high/low IC beats the last-bar-range volatility baseline (rule C1/D), and any T-profit
  number (the battery was wired in I3, after this run).
- What the r2 run cannot give: the battery (`battery_report.txt`, `battery_profit.json`), the persisted predictions and the
  volatility baseline inputs are not in its folder, and its checkpoint is NOT on Drive: the publish sent
  `.../baseline-001-s0-r2/train` to the Drive trash (B-050, fixed 2026-11-08; restore it from the trash).
  No entry recomputes a report from a checkpoint (`jobs/` has `baseline` and `device_smoke` only). DECISION: do NOT build a
  recompute entry; queue a FRESH baseline at current main (about 16 minutes on a T4): new run_dir `baseline-002-s0`, same
  config as `docs/jobs/baseline-001-s0.json` (5 epochs, seed 0; job file `docs/jobs/baseline-002-s0.json`), preceded by a
  fresh smoke job (`baseline-smoke-002`) on the same commit (rule E5, the commit has not run on the GPU). The planner queues it; the reading session starts when its folder is on
  `colab-results`. Compare it with r2 (same seed, same data; differences are code changes I2-I3 and nondeterminism).
- Compute in the reading: (1) the volatility baseline for high/low (rank IC of the last bar's range vs the truth, same
  rows) and report only the excess as skill; (2) the battery from the run folder (profit per use of the outputs with
  its trivial baselines and day-block CI, cost 0.0008 per side); (3) T-conf / T-sharp per head from `test_report.json`;
  (4) record whether B-020 (bar OPEN time) can be settled from the run's data; (5) the validation curve vs train
  (best epoch 1) as the input to M1/M2r.
- Output: `docs/results/baseline-001.md` (targets per head with trivial baselines, step time, GPU name, epochs, gate
  failures if known), bottleneck rows (B-022, B-025, B-026, B-038, B-043 updated). Acceptance: every number has its source
  file named; no statement of profit without the baseline rows and the CI.

### T3b — interactive notebook section + Drive cached env (`feat`, M, builder)
- Inputs: PLANNER_HANDOFF "Next" 2; the run steps of I3. A notebook section: a settings cell -> train -> test report
  shown inline, over `workflow/run.py` steps (no logic in the notebook, G6). A cached Python env on Drive: venv
  `--system-site-packages`, one tar keyed by Python + torch + requirements hash, rebuilt on a key change.
- Acceptance: a CPU test of the cache key and restore logic; the notebook runs the fixture end to end in a test.

### G1 — test environments E1 / E2 / E3 (`feat` + `test`, M, builder)
- `data_io/synthetic.py`: the E1 generator of `docs/MODEL_DEV_PROTOCOL.md` section 3 (planted signal with known IC,
  known heteroscedastic noise, intrabar paths, the exact conditional mean / sd stored next to the labels), written as a
  contract artifact; E2 = global label permutation; E3 = a 20-coin x 6-month training-era slice of the real artifact
  with its own purged split (through I1's loader). A new D-nnn for the generator defaults.
- Acceptance: tests show the oracle columns reproduce the labels' conditional moments within MC error; the oracle
  T-sharp share is 1.0 / 1.0 / ~0 (high / low / close); E2 keeps inputs byte-identical and labels permuted; E3 never
  includes a row after the full run's train end.

### G2 — gate tooling and gate job (`feat` + `test`, M, builder)
- `trainer/grad_report.py` (gate A, checks A1-A7 on the representative batch set; gradient code stays in `trainer/`,
  G5), `eval/gates.py` (gates B, C, rule D, leakage canary: verdicts from `target_report` per env x seed), job entry
  `gates` (E1 -> E2 -> E3 x seeds -> `gate_report.json` / `.md` keyed by the settings hash), and the job-side check:
  a full-training job refuses to start without a passing gate report for its settings hash (baseline arm exempt,
  recorded). The trainer trace switches to the representative batch set (switch, default today, D-nnn).
- Acceptance: on E1 a correct small model passes; a planted defect fails the right check (a detached head -> A2, a
  constant confidence -> B, a zero std -> C coverage, a label leak -> canary); CPU < 60 s per test.
