# ROADMAP — from a verified port to a better model (owner, 2026-11-02)

Read `docs/STATUS.md` first, then `docs/RULES.md` once, then ONLY your step below. The port itself (`BUILD_PLAN.md`,
batches B1-B10) is done and stays the record of how each layer was proven equal to the TF model.

Goal: IMPROVE the model. Make confidence meaningful and uncertainty sharp. Keep or raise its skill. Work in PyTorch
from now on. The TF project (`yuosef772424/crypto-signal-prediction`) is frozen as the reference: it is read for
history, data export and old results, and new model work does not go there.

## Targets (every model step reports these on val AND test; numbers, not adjectives)
- **T-skill:** per head, rank IC (within timestamp, across coins) and per-coin Spearman of prediction vs truth. Also
  AUC for class heads, always next to its trivial baseline (rule C1). No regression beyond seed spread vs the baseline.
- **T-conf (confidence is meaningful):** across confidence deciles, the absolute error of the head's own prediction
  falls monotonically (Spearman of decile vs mean |error| <= -0.8). The top-minus-bottom error gap must exceed its
  bootstrap CI, measured within coin so coin scale cannot fake it. A constant confidence scores 0 (trivial baseline).
  NOT the TF "hit rate": for high/low it is ~99 % for any model.
- **T-sharp (uncertainty below the expected range most of the time):** in >= 80 % of samples the predicted
  uncertainty width (predictive std, in price units) is smaller than the predicted range (pred_high - pred_low).
  At the same time the empirical coverage of the 80 % interval stays within 75-85 %.

## Approach
Every improvement is a separate registered switch whose default reproduces the current model. One run compares
the baseline vs all switches on. If the targets improve, ablate one switch at a time. If not, ablate plus
diagnose. Tests stay fast (rule E3).

## Steps (one session each; `r` = a Colab run plus its reading in the next session)

### R0 — Make the rules checkable (`test`/`docs`, S)
Build the checks `docs/RULES.md` marks "built in R0":
- `tests/test_decisions.py`: pin a fingerprint of the defaults; a change without a new `docs/decisions/D-nnn` fails.
- `tests/test_docs_claims.py`, starting with a fix of `NOTES.md` ("no PyTorch code yet" is false).
- A trivial-baseline test per metric in `stats/metrics.py`.
- A grep test: no metric defined outside `stats/`.
- The CI duration budget.

Add a `CLAUDE.md` line check that STATUS is < 60 lines.

### T1 — Complete the trainer (`feat`, M)
- Epoch loop.
- Lambda / LR schedules: port the TF `trainer/schedules.py` formulas with unit parity tests against values computed
  from the TF source.
- Early stopping and best checkpoint by the registered selection metric.
- Checkpoint + resume: model, optimizer, Kendall state, `RunIdentity`; resume refuses a changed identity.
- Seeds through `core/seeds.py`.
- Fixture test: 3 epochs run; resume after epoch 1 equals an uninterrupted run.

### T2 — Real data in (`feat`, M; TF repo side + this repo)
- On the TF side, an export tool writes the real dataset (the one TF `main.ipynb` trains on) as a contract artifact
  (`DATA_ARTIFACT_CONTRACT.md`).
- Its manifest records the source commit, the coins, the sealed coins EXCLUDED (TF `tools/holdout_guard.py` list)
  and the max bar time < 2026-10-01 (rule D1).
- This repo's loader refuses an artifact without those fields.
- Run it once on Colab (Drive data).
- The TF-side tool is the only change to the TF repo in this roadmap (a `feat` PR there).

### T3 — Colab runner for this repo (`feat`, S-M)
Reuse the TF runner design (queue branch -> run -> results branch + Drive). Two fixes:
- The device is asserted and recorded (rule E1).
- No `/tmp` dependence, plus a `republish` command (rule E2).

Colab ships CUDA torch, so no reinstall should be needed. Verify that and record it.

### T4 — Diagnostics on the targets (`feat`, M)
- `target_report` (T-skill, T-conf, T-sharp, coverage), with tests on synthetic data with a known answer.
- `model/probe.py`, extended with the RAW input next to every layer.
- NIG floor-hit shares per parameter and head.
- A trainer trace: per-head loss, Kendall weights, schedule values, per-head gradient share.
- One function `health_report(run)` writes them all.

### T5r — Baseline on GPU
Job `baseline-001`: the current default model (GRU encoder + level passthrough, anti-memorization on, as TF
`main.ipynb`; check `ORACLE_REVIEW.md` V1 for the exact default and the D-013 close-target decision in the TF repo),
2 seeds, full training, `health_report`. Record step time, GPU name and the targets.

Next session: write `docs/results/baseline-001.md`, add bottleneck rows, and if the init difference (B-007) matters,
add a Keras-init switch.

### T6 — Switches (`feat`, M+M; may split in two sessions)
Each is a registered option, default = today:
1. `input_path="direct_wide"`: no decomposition and no learned projection before the deep blocks. The inputs are
   lifted to d_model by a fixed information-preserving map. Use the existing `input_skip` seam where it fits.
2. A selection metric on T-skill (T-conf / T-sharp as tie-breakers).
3. A confidence target that ranks the head's own |error| within coin, without the diversity penalty (B-005).
4. NIG floor: target tail treatment after standardization, plus a smooth beta floor (B-001).
5. Interval scale: the existing `eval/calibrator.py` fitted on val, so the 80 % interval covers ~80 %.

Tests per switch on the fixture: default = parity outputs unchanged; switch on = runs and changes what it should.

### T7r — One run: baseline vs all switches on
Card first: predicted movement, accept rule. Job `all-001`: 2 seeds per arm, one commit, `health_report`.

### T8 — Read T7, ablate or diagnose, fix
- Improved: an ablation job (all-on minus one switch, 1 seed each, to rank them).
- Not improved: the same ablation plus the diagnostics, then fix and re-run.

Then propose the new default to the owner (`docs/decisions/D-nnn`).

### Later (in this order unless R-REVIEW reorders)
1. Slot panel: top-N coins as one sample, anonymous ordered slots, outputs for all slots. The port's `PanelModel`
   is the starting point. Check TF `failure_registry.csv` F-0035 first.
2. Rolling train/test study: the seams `workflow/rolling.py` and `staleness.py` exist.
3. Strict validation of the market-neutral portfolio (realistic costs, forward data), only when the owner asks.

### R-REVIEW (every 5 merged PRs, S)
Per rule F3.

## Using the TF project for follow-up
- Read-only by default: `docs/research/failure_registry.csv` (check `reopen_if` before repeating an idea), the
  decisions and results folders, the holdout lock, the data pipeline, and `docs/research/hardening/bottlenecks.csv`
  (its open rows were carried over here as B-001..B-005).
- Allowed writes there: the T2 export tool, a failure-registry row when an idea tested here fails, and a note in
  its STATUS that model work moved here.
