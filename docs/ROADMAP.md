# ROADMAP — from a verified port to a finished, honestly evaluated model (owner 2026-11-02; rewritten H0c 2026-11-07)

Read `docs/STATUS.md` first, then `docs/RULES.md` once, then ONLY your step below. A step that touches the model,
the trainer, the targets or a training job also reads `docs/MODEL_DEV_PROTOCOL.md` (rule C6: gates before any full
training). The port (`docs/BUILD_PLAN.md`, B1-B10) is done; R0-T4 are done (their text is in git history, STATUS rows
link the PRs).

Goal: IMPROVE the model, then FINISH it professionally: meaningful confidence, sharp uncertainty, skill kept or
raised, every claim next to its trivial baseline, an honest final evaluation (holdout opened once, only when the owner
asks, rule C5), a model card, a versioned release, runs anyone can reproduce, docs a fresh session can follow. Work in
PyTorch. The TF project (`yuosef772424/crypto-signal-prediction`) is the frozen reference.

## Targets (every model step reports these on val AND test, per head; numbers, not adjectives)
- **T-profit (NORTH STAR, owner decision 2026-11-07: "I do not want the close itself; I want the model to be able to
  make profits in any way"):** profit after realistic costs (default 0.0008 PER SIDE, i.e. 0.0016 per round trip, the
  battery default `D-009`; the owner may correct it) from ANY use of the model outputs: close-direction trades, limit
  orders at the predicted high / low, the uncertainty as a trade filter and as position size, cross-sectional
  market-neutral ranking. Measured by the evaluation battery (`eval/battery*.py`) through the adapter
  (`eval/battery_adapter.py`), per seed, on val AND test, each use next to its trivial baselines (always up / down,
  shuffled, constant-offset limit, equal weights, last-candle momentum) and a dependence-aware CI (day-block
  bootstrap, `stats.metrics.block_bootstrap_ci`). Every variant tried (model, switch, rule, threshold) counts toward
  multiple testing: the CI level is `1 - 0.05 / n_variants`. Exploratory until the owner asks for strict confirmation
  (rule C5, F4). A use of the outputs is profitable only if it beats every baseline row AND its CI excludes 0 on every
  seed.
- **The targets below are MEANS, not goals:** T-skill, T-conf and T-sharp are diagnostics that explain WHY a model does
  or does not make profit (where the edge comes from, whether the uncertainty can be used). A step may not claim
  success on them alone, and may not reject a configuration on them alone when T-profit says otherwise; the protocol
  gates (gradient, confidence, uncertainty) stay as health checks before full training.
- **T-skill:** per head, rank IC within timestamp (across coins) and per-coin Spearman of prediction vs truth, next to
  its trivial baseline (rule C1): shuffled prediction for every head, and for high / low also the last bar's range
  (a volatility baseline: only the excess over it is skill). No regression beyond seed spread vs the baseline.
- **T-conf (confidence is meaningful):** across confidence deciles, the absolute error of the head's own prediction
  falls monotonically (Spearman of decile vs mean |error| <= -0.8), and the top-minus-bottom error gap exceeds its
  bootstrap CI, measured within coin so coin scale cannot fake it. A constant confidence scores 0 (trivial baseline).
- **T-sharp (uncertainty is sharp around the predicted value; corrected by the owner 2026-11-07):** PER HEAD (close,
  high, low; pass/fail for each), the predictive std in price units is smaller than the head's own predicted move
  from the last close, `|decoded pred_price - last_close|`, in >= 80 % of samples (owner example: high predicted +5 %
  above the close -> uncertainty ~2-3 %). At the same time the empirical coverage of the head's 80 % interval stays
  within 75-85 %. A zero std scores 1.0 on the share (trivial), hence the coverage condition. The earlier text
  ("width < pred_high - pred_low") was a wrong interpretation; `eval/target_report.py` computes the per-head version since T4b.
  Warning measured in H0c: even the true conditional distribution fails this on the close head (share 0.00; high /
  low 1.00), see `docs/MODEL_DEV_PROTOCOL.md` gate C; owner decision O-1 (CLOSED 2026-11-07: close-head T-sharp is informational only,
  never a target or a gate; high and low heads keep it).
- **Direction:** SUSPENDED for the high and low heads (owner): their sign / hit rate / up-down accuracy / class AUC
  is never evidence and never a selection metric. Direction is judged on the close head only, next to its trivial
  baselines (shuffled, constant, momentum sign, reversal sign).

## Approach
Every improvement is a registered switch whose default reproduces today (rule B1, decision file for every new
default). Before any full training the configuration passes the protocol gates (gradient, confidence, uncertainty) on
synthetic, shuffled-label and small real data, 3 seeds. One full run compares the baseline vs all switches on; if the
targets improve, ablate one switch at a time; if not, ablate plus diagnose (model doctor). Tests stay fast (E3).

Step header: change type, size (S < ~150 changed lines, M ~150-500, L more), agent kind (**builder** = clear spec,
the cheaper agent model suffices; **researcher** = open question, a stronger agent model, says why). `r` = a Colab run
plus its reading in the next session.

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

## Block 2 — read the baseline, then build the gates

### T5r — read the baseline (`docs`/`exp`, S-M, researcher: the stronger model, because it decides what to try next)
- Inputs that already exist on `colab-results` (read with `git show origin/colab-results:<path>`, do not copy): `runs/baseline-001-s0-r2/`
  (`baseline_summary.json`, `health_report.md`, `test_report.json`; 2026-11-07 @44f6e71, 5 epochs, T4, 972 s; its
  `reg_target_mode` is `return`, scale 1.0, so the decodes are right: HF-13 settled for this run). Older: `baseline-001-s0`
  (@55e2db4, pre-T4b, history only), `trial-002`, smoke `baseline-smoke-001`. Known so far: best epoch 1 of 5 (val
  -5.062 -> -4.947 while train -5.75 -> -6.12: overfit after epoch 1, B-038); test rank IC high 0.393 / low 0.436 /
  close 0.057 (shuffled ~0); T-sharp share 0.84 / 0.89 but 80 % coverage 0.89 / 0.93 / 0.88 (> 0.85: intervals too
  wide); confidence decile Spearman -0.81 / -1.00 / -0.99; ~169 s per epoch (median), 2,015,804 / 239,297 / 239,772
  windows. NOT yet known: whether high/low IC beats the last-bar-range volatility baseline (rule C1/D), and any T-profit
  number (the battery was wired in I3, after this run).
- What the r2 run cannot give: the battery (`battery_report.txt`, `battery_profit.json`), the persisted predictions and the
  volatility baseline inputs are not in its folder, and its checkpoint lives only on Drive (`.../baseline-001-s0-r2/train`).
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

### TR1 — trainer switches (`feat`, M, builder)
Registered options in `FitSettings` / `Trainer`, defaults reproduce today (D-nnn), each with a fixture test (default
= outputs unchanged; on = changes what it should):
1. `grad_clip_norm` (None; TF used 1.0, B-018).
2. weight-decay groups: `all` (today) vs `exclude_norm_bias_embedding`, and the decay value as a setting (today's
   1e-4 x lr 1e-3 is a no-op, H-M6).
3. EMA of weights (`ema_decay` None); evaluation, selection and the checkpoint's best weights use the EMA copy when on.
4. init registry: `torch` (today) vs `keras` (glorot uniform for Linear / Conv, orthogonal recurrent, zero bias; B-007).
5. LR warmup + cosine via the existing `LrSettings` exposed in the job settings (no new formula).
6. log-only: train / val / trace seconds split (HF-37), and an eval()-mode metric on a fixed train subset per epoch
   (the train / val gap the memorization study needs, H-M8).
- Acceptance: resume parity test still green with each switch on; pins + decision file.

### G3r — gates on the baseline configuration (`exp`, S-M, researcher)
- Run the `gates` job on today's default (E1 -> E2 -> E3, seeds 0-2; CPU for E1, GPU for E3). Read: which checks fail,
  on which seeds; for each failure the fix of the protocol table becomes a card (switch from TR1 or an architecture
  card for T6c). Output: docs/results/gates-baseline.md.
- Acceptance: every gate check has a value per env x seed; every failure has a named next action.

## Block 3 — explain the memorization (protocol section 5)

### M1 — memorization study, CPU part (`exp`, M, researcher)
- Card first. On the real artifact and the T5r checkpoint (`baseline-002-s0` on Drive): H-M8 (eval-mode train metric vs val; defines the effect
  size), H-M1 counts (distinct timestamps, label autocorrelation, cross-coin correlation, ESS), H-M2 (timestamp R^2,
  gap decomposition, month-id probe), H-M3 probes (coin id from inputs / tokens / embedding), H-M5 (boundary overlap,
  B-020), H-M9 (duplicate / stale windows). Re-runnable code goes into a tool with a test (not a notebook).
- Output: docs/results/memorization-m1.md, one bottleneck row per hypothesis (verified / rejected / open).
- Acceptance: every hypothesis tested here has its number and the expected-if-true value next to it.

### M2r — memorization study, short GPU arms (`exp` + small `feat`, M, researcher)
- A coin-split option in `SplitPlan` (default off, D-nnn, own `feat` commit). Short arms (fractional epochs, one
  change each, seeds where it decides): H-M1 stride-h subsample, H-M3 `level_passthrough=False`, H-M4 coin split vs
  time split + ridge baseline, H-M6 weight decay 1e-2 / 1e-1 with exclusions and LR 3e-4, H-M7 shuffled-label curve and
  d_model 16 / 32 / 64, 1 vs 2 layers.
- Output: the verified cause(s) and the switch each implies (input to T6c); failed ideas get a TF failure-registry row.
- Acceptance: the M1 effect size is explained (or the remaining unexplained part stated), each claim with seed spread.

### TR2 — input pipeline and step time (`feat`, S-M, builder)
- Switches, default today with equal outputs proven (same batches, same order): pinned memory + non-blocking copy,
  a one-batch prefetch, the whole training tensor on the GPU when it fits, AMP fp16 with GradScaler (default off).
- A GPU timing job: steps / s with each switch, train time without validation (HF-37). Output: B-008 closed.
- Acceptance: a test shows identical losses for 2 epochs on the fixture with each equal-output switch on vs off (CPU).

### T5r-s1 — baseline seed 1 (`exp`, S, builder)
- Job `baseline-001-s1` (exists in `docs/jobs/`) on the post-hardening commit with TR2's fastest equal-output pipeline;
  the baseline arm is exempt from the gates (protocol section 2) but its gate report is attached. Output: seed spread
  added to `docs/results/baseline-001.md`. Acceptance: both seeds on one commit (C4), spread per target stated.

## Block 4 — improve the model (switches, then one comparison, then ablation)

### T6a — switches 1-3 (`feat`, M, builder)
1. `input_path="direct_wide"` in the input-variant registry (H3): no decomposition and no learned projection before
   the deep blocks; inputs lifted to d_model by a fixed information-preserving map (B-002; check with the probe).
2. Selection metrics (C3): a profit-based selection metric (battery profit of the close / limit / sigma-sized uses on
   VAL, with its baselines) is a T6a candidate (T-profit); not built in I5. `val_rank_ic_close` (close-head within-timestamp rank IC) and `val_t_skill_excess` (mean
   per-head rank IC minus its trivial baseline: shuffled for close, last range for high / low); T-conf / T-sharp as
   tie-breakers. Default stays `val_raw_loss` (B-003). High / low direction never enters (rule D).
3. Confidence methods registry: today's loss; ranking loss on own |error| within coin without the diversity penalty
   (B-005); a dedicated error-predictor head on detached residuals. Batches stratified by coin when a ranking method is on.
- Acceptance (T6a, T6b, T6c alike): per switch on the fixture, default = parity outputs unchanged and switch on = runs
  and changes what it should; the E1 gate run (CPU, 3 seeds) recorded per switch; pins + D-nnn for every new default.

### T6b — switches 4-6 (`feat`, M, builder)
4. NIG floor: tail treatment of the standardised target + a smooth beta floor (B-001); report the `unc_max=20` cap.
5. Interval scale: `eval/calibrator.py` fitted on a calibration half of val, measured on the other half and on test.
6. Regression loss option `czar` (bounded / Cauchy-type loss harvested from PR #25, BRANCH_AR_STREAM section 4),
   default the NIG NLL; quantile heads as a confidence / interval method (protocol section 4).
- Inputs: G3r and M1 / M2r results (which switches the evidence asks for first). Acceptance: as T6a.

### T6c — architecture switches named by the evidence (`feat`, M, builder after a researcher card; skip with a reason)
- Only changes that G3r, M1 / M2r or the doctor named (e.g. `linear_path` / level passthrough gated or off if they
  dominate; LayerNorm GRU if the hidden state saturates; a separate scale trunk if gate A6 shows mu / scale conflict;
  removing the leak path H-M3 finds). One switch per finding, default today, card first.

### T7r — gates, then baseline vs all switches on (`exp`, M, researcher)
- Gates on the all-on configuration (must pass, protocol section 2). Card first: predicted movement per target,
  accept rule. Job `all-001`: 3 seeds per arm (C4 minimum 2; owner decision O-5), one commit, health / doctor report.
- Acceptance (to call it improved): T-profit (battery, with baselines and CI) first; then T-conf and T-sharp (high, low; the close head's T-sharp is
  informational, O-1) closer to target on every seed,
  T-skill not below the baseline beyond seed spread, close direction never claimed without its trivial baselines.

### T8 — read T7r, ablate or diagnose, fix, propose the default (`exp`, M, researcher)
- Improved: all-on minus one switch, 1 seed each, to rank them; then 2 seeds for the final set. Not improved: the
  same ablation plus the doctor, a fix as a new switch, gates, re-run. Output: `docs/decisions/D-nnn` proposing the new
  default to the owner.

## Block 5 — the model doctor

### D1 — doctor part 1: activations, heads, training history (`feat`, M, builder)
`eval/doctor/` (one module per check family, a `DOCTOR_CHECKS` registry, G8): dead units (forward hooks on activation
layers), GRU hidden saturation, class-head saturation, attention entropy (transformer encoder only), NIG floors and the
`unc_max` cap (extends `eval/nig_floor.py`), prediction collapse (std pred / std label per head), Kendall
log-variance drift and train / val gap per epoch (from the TR1 history columns), per-coin coverage spread and
calibration per head (coverage at 50 / 80 / 95 %, PIT histogram, class ECE). Each check: value, threshold (protocol
section 6), verdict, trivial reference. Builds on `workflow/health_report.py`, `model/probe.py`, `eval/nig_floor.py`,
`eval/target_report.py`, `trainer/trace.py`. Acceptance: each check fires on a model with a planted defect (e.g. a
zeroed layer -> dead units, a huge logit bias -> class saturation) and stays quiet on the fixture default.

### D2 — doctor part 2: attribution, ablation, battery; run it (`feat` + `exp`, M, builder)
Passthrough dominance (share of readout variance and gradient from `linear_path` + level passthrough vs the encoder),
layer ablation importance (zero / permute a layer's output -> delta val loss and delta T-skill per head), input
permutation importance (per feature and per time block), gradient flow and update / weight ratios per layer (reuses
`trainer/grad_report.py`), the
per-layer probe (`model/probe.py`) and the I5 battery through its adapter; `doctor_report.json` / `.md` written by every
full training, called from `workflow/health_report.py` (one call, as T4). Run it on the T8 model (and the T7r
checkpoints); each finding -> a bottleneck row and, if it names a fix, a T6c switch (back through the gates).
Acceptance: every protocol section 6 check has a value and verdict on the T8 model; a planted-defect test per check.

## Block 6 — finish

### F1 — reproducible runs and versioning (`feat` + `docs`, M, builder)
- Semantic version in `pyproject.toml` + a CHANGELOG; a `reproduce` job that re-runs a recorded job spec (commit +
  settings + artifact fingerprint) and compares the outputs (CPU fixture: identical; GPU: documented tolerance);
  environment lock (requirements lock + torch / CUDA / GPU recorded, as rule E1); a results index under docs/results.
- Acceptance: a test reproduces the fixture baseline byte-identically from its recorded spec.

### F2 — walk-forward stability on pre-seal data (`feat` + `exp`, M, builder then researcher; owner decision O-2)
- The rolling half of I4 (rolling folds over the run steps, through the one `Trainer`), then the T8 model over >= 4
  expanding folds before the time seal: does skill / T-conf / T-sharp hold across folds (seed spread per fold)?

### F3 — model card and fresh-session docs (`docs`, S-M, builder)
- `MODEL_CARD.md` (under docs): intended use, data and its seals, architecture, training, every target with trivial
  baselines and seed spread, calibration, known failure modes (memorization findings, close-head T-sharp), limits.
  README quickstart a fresh session can follow from zero (setup, fixture run, smoke job, gate job, full job).
- Acceptance: a fresh session following only README + STATUS runs the fixture and the gate job without asking.

### F4 — final honest evaluation (`exp`, M, researcher with the owner) — ONLY when the owner asks (C5)
- A pre-registration card committed BEFORE opening: frozen commit, model, metrics, accept rules, the holdout
  definition (the 183 sealed coins excluded from the artifact and / or bars after the 2026-10-01 time seal; B-022:
  TF results are not a clean holdout). The holdout is opened ONCE; the result is published whatever it says.

### F5 — release v1.0.0 (`docs`, S, builder)
- Tag, release notes, the frozen model artifact on Drive with its hash and model card, archive merged branches
  (`feat/ar-stream` as `archive/feat-ar-stream`), STATUS / handoff final state.

## Later (in this order unless R-REVIEW reorders)
1. Slot panel (the panel half of I4): top-N coins as one sample, anonymous ordered slots. Check TF
   `failure_registry.csv` F-0035 first.
2. I7 ar-stream experimental package: only if the confidence gate fails for every method of protocol section 4, or
   the owner asks (`docs/hardening/BRANCH_AR_STREAM.md` lists its required fixes).
3. Strict validation of the market-neutral portfolio (realistic costs, forward data), only when the owner asks.

### R-REVIEW (every 5 merged PRs, S)
Per rule F3.

## Using the TF project for follow-up
- Read-only by default: `docs/research/failure_registry.csv` (check `reopen_if` before repeating an idea), the
  decisions and results folders, the holdout lock, the data pipeline, and `docs/research/hardening/bottlenecks.csv`
  (its open rows were carried over here as B-001..B-005).
- Allowed writes there: the T2 export tool, a failure-registry row when an idea tested here fails, and a note in
  its STATUS that model work moved here.
