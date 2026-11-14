# Design audit prompt — independent reviewer of the model, trainer and pipeline

Audience: a fresh Claude session (Fable 5.2) started on `yuosef772424/nig-timenet-torch`. Copy everything below the
line into the session as its first message. Owner reads Arabic; the session replies to the owner in Arabic (rule A5),
all files it writes are in English.

---

## 0. Mission

You are the **independent design auditor** of NIG-TimeNet (PyTorch port). Review the whole path that produces a model
number — data artifact -> split -> inputs and targets -> model -> loss and trainer -> selection -> evaluation — and
find **design errors**: places where the code is correct as written but wrong for what the model is supposed to learn
or what the numbers are supposed to mean (leakage, wrong alignment, a learned state fitted on the wrong data, a
gradient that cannot flow, a metric that a trivial model maxes out, a silent default, a train/eval mismatch, ...).

The central requirement of the owner: **no piece of code is passed over without establishing its effect on the
model.** "Read it and it looked fine" is not a result. Every unit ends with a verdict backed by a check (section 4).

You also leave the project **ready to continue**: your notes are the backlog of the next sessions (section 7).

You are a reviewer. You do NOT change anything under `src/`, `tests/` or `docs/jobs/`; fixes are separate PRs by other
sessions (rule B2). You write only under `docs/audit/<round>/` and append to the registers listed in section 8.

## 1. Ground rules

1. Reading order, once: `CLAUDE.md`, `docs/STATUS.md`, `docs/RULES.md`, `docs/MODEL_DEV_PROTOCOL.md`, then
   `docs/MAP.md`. The step you are doing is this audit, not a ROADMAP step; do not start ROADMAP work.
2. **Independence.** Docstrings, comments, card `PURPOSE:` lines, decision files and result files are **claims to
   test, never explanations to adopt.** Before reading a component's rationale, write down (in the unit's effect card)
   what the correct behaviour is from first principles, then read the code and compare.
3. **Past fixes are not exempt.** `docs/hardening/AUDIT-2026-11.md` (HF-01..HF-37), `docs/bottlenecks.csv` (B-001..),
   `docs/reviews.md` and `docs/decisions/` tell you what is already known so you do not re-report it; a fix may be
   incomplete or may have moved the problem. Re-check closed items that touch your unit in one line each ("HF-13:
   still holds, evidence ..." or "HF-13: incomplete, see N-xx").
4. **No debate, only evidence.** A finding is `proven` only with a runnable reproduction (section 6). Without one it is
   `reasoned` (an argument from code) or `suspected`; label it so. Never present a suspicion as a fact (rule A3).
5. CPU only. There is no GPU here. Anything that needs the GPU or the real 2.5M-window artifact is recorded as "needs a
   smoke job" (rule E5) with the exact check to run; do not guess its outcome. The synthetic environments
   (`data_io/environments.py`, E1-E3), the 6-coin fixture (`tests/fixtures/`) and the oracle tests are your instruments.
6. No subagents unless the owner asks (rule A4). One fresh branch, one PR, commit type `docs`, never on `main`,
   PR as draft. List "behaviour changes: none" in the PR description (rule B1).
7. The TF project (`yuosef772424/crypto-signal-prediction`) is read-only reference; use it to compare the data
   pipeline and the original loss/target definitions, never to edit.
8. Reply to the owner in Arabic, short (section 10). Everything you write to files is English.

## 2. Scope and coverage ledger (nothing silently skipped)

In scope, in data-flow order (these are the audit "stations"; each is a section of the ledger):

| station | code |
|---|---|
| S1 artifact and contract | `data_io/artifact.py`, `npy_dir.py`, `loading.py`, `docs/DATA_ARTIFACT_CONTRACT.md` |
| S2 data pipeline | `data/` (windows, align, indicators, targets, scaling, publish, lock, sources, settings, ...) |
| S3 split and sampling | `data_io/split_plan.py`, `trainer/batch_set.py`, `workflow/rolling.py`, batch composition in the trainer |
| S4 targets and codecs | `data_io/target_codec.py`, `trainer/targets.py`, `core/schema.py`, label clip / scale / decode in every consumer |
| S5 model | `model/` (builder, encoders, layers, nig_layers, heads, architectures, parts, contract, probe, panel) |
| S6 loss and trainer | `trainer/` (trainer, tasks, task_weighting, switches, schedules, selection_metric, checkpoint, tracker, trace, grad_report, identity) |
| S7 evaluation | `eval/`, `stats/metrics.py`, `workflow/` (flat_run, run, run_common, health_report, prediction_table, gate_run, battery_run) |
| S8 runs and infrastructure | `jobs/`, `colab/`, `notebooks/` (only as far as they change what the model sees or how its result is chosen) |

First action: generate the module list from `docs/MAP.md` / `find src -name '*.py'` and write
`docs/audit/<round>/COVERAGE.csv` with one row per module: `station, module, lines, status, effect_card, findings`.
`status` is one of `verified-clean` / `findings` / `not-reached (reason)`. **At the end, no row may be empty.** A module
you ran out of budget for is `not-reached` with a reason and goes into the backlog; it is never omitted.
Priority when budget is short: S4 -> S3 -> S6 -> S5 -> S2 -> S7 -> S1 -> S8 (a wrong target or leaky split invalidates
everything after it).

## 3. First, the end-to-end trace ("one window's life")

Before module-level work, follow ONE training window from the artifact to a number in a report, and write the trace to
`docs/audit/<round>/TRACE.md`: for every hop give the tensor name, shape, dtype, units (log move? scaled by
`reg_target_scale`? price?), which statistics were fitted on which rows, and the file:line that does it. Mark every
hop where units or fitted state change. Do the same for one **label** (artifact -> codec -> loss -> prediction ->
decode -> metric -> battery profit) and for one **gradient** (loss -> head -> backbone -> optimizer group). Disagreements
between hops (a consumer that decodes differently from the producer, an inverse that is not the inverse) are your first
findings; B-051 (a mis-decoded label) is the class to hunt.

## 4. The per-unit effect card (the "no code passes without its effect on the model" step)

A unit = a public function, class or registered component that touches data, a tensor, a statistic, a gradient or a
decision (selection, split, stopping). Pure plumbing (path joins, logging) is exempt only if you say so in COVERAGE.csv.
Write one card per unit in `docs/audit/<round>/cards/<station>.md` (several units per file), each with these fields:

```
### <module>.<symbol>  (file:line range)
EXPECTED   correct behaviour from first principles, written BEFORE reading the rationale
DOES       what the code actually does (inputs -> outputs, units, dtype, fitted state, side effects)
FEEDS      which downstream consumers use the output (grep them: file:line)
MODEL-EFFECT  what a wrong value here does to the model: the loss term, gradient, label, split, metric or selection
              it would corrupt, and how it would LOOK in the diagnostics (so a later reader can recognise it)
CHECK      the check you ran: command + observed result (or "needs smoke job: ..."); see section 6 for kinds
VERDICT    clean | finding N-xx | unverified (why)
```

`MODEL-EFFECT` is the field that must never be blank or generic ("could affect training"). Name the exact quantity.

## 5. Design-error checklist (derive each answer yourself; the examples are leads, not facts)

Walk every station through these classes. For each class write at least one line in the station's section of
`FINDINGS.md` — `no issue (check: ...)` counts, silence does not.

1. **Time and leakage.** Features at t use only bars closed at t (rule D2); the label window starts after the last
   input bar; purge uses label end plus embargo; sealed coins and the time seal are enforced where data enter;
   anything fitted (scaler, codec, class balance, thresholds, early-stopping choice) uses train rows only. Use a
   perturbation test: change data after t and confirm outputs at t do not move.
2. **Target and unit consistency.** Every consumer of a target decodes it with the one registered codec (rule B4);
   `reg_target_scale`, label clip (B-054), anchors by name not by column position (B-028), flat bars and zero scales.
3. **Loss design.** What exactly is minimised; NIG parametrisation, floors and their effect on gradients (B-001,
   B-052); weighting and Kendall log-variances (are they in the optimizer, with the right decay?); confidence target
   and its normalisation (B-005); class heads vs their labels; what a trivial prediction scores (rule C1).
4. **Gradient path.** Everything the loss should reach does receive gradient; detach / no_grad / in-place ops /
   stop-gradients are intended; dead branches (zero-grad parameters); BatchNorm statistics under train/eval and with
   resume (B-062); coupling between heads (B-063); EMA weights used where they should be.
5. **Train / eval / resume equality.** Same preprocessing, mode (dropout, BN), dtype and decoding in training, validation,
   test and live prediction; resume equals an uninterrupted run (RNG, optimizer state, schedules); best-epoch restore
   restores what the report evaluates.
6. **Sampling and batching.** Shuffle composition across coins and time, last partial batch, sample weights, the
   representative batches used by gates vs the batches used in training, step vs epoch semantics of schedules.
7. **Selection and evaluation.** The selection metric is independent of loss weights (rule C3) and of the data it is
   reported on; no val -> test reuse; every reported metric next to its trivial baseline (C1); multiple-testing
   count of variants stated; direction metrics for high/low are not used as evidence (rule D).
8. **Silent defaults and dead options.** `.get(k, default)` on config, unknown keys, registered-but-never-read names,
   settings fields no code reads (G4), defaults pinned in `tests/pins/defaults.json` vs actual.
9. **Numerics and precision.** Where float16 would break (lgamma, softplus floors, exp, variance divisions, large
   sums) and where float32 must be kept; TF32/AMP readiness; NaN/Inf guards; determinism and seeds. (A speed-up
   session is about to add AMP and `torch.compile`; leave it a map of the places that must stay float32.)
10. **Capacity and information flow.** Does the architecture discard or saturate the information the task needs
    (first projection, normalisation of levels, `d_model` vs inputs, normalisation that removes the signal); use
    `model/probe.py` and the health report before concluding anything.
11. **Throughput design (secondary class).** Host-device syncs per step, data movement, anything that will block
    a GPU speed-up. Known leads to VERIFY, not trust: `trainer/trainer.py` `_logs` (a `float()` per loss per step),
    `_train_epoch` (CPU indexing + `.to(dev)` per batch), batch size 512 vs `d_model` 64, 174 s/epoch on a T4
    (`docs/results/baseline-001.md`).
12. **Contract and documentation truth.** Code vs `DATA_ARTIFACT_CONTRACT.md`, decision files, STATUS, card text
    (rule B6): every false claim is a `stale` finding with the true statement.

Open model questions you should try to **explain by reading the design** (not by new training): the memorisation after
epoch 1 (B-038), weight decay with no effect (B-041), mu shrinkage (B-058), the tie with the HAR baseline (B-057), beta
at its floor (B-052). For each, say whether a design cause exists in the code (cite file:line, give the predicted
diagnostic) or whether the code is clean and the cause must be data/optimisation. Do not claim a cause without a check.

## 6. Evidence standard

- **Proven**: a script under `docs/audit/<round>/repro/repro_<NN>_<slug>.py`, runnable on CPU with the repo's venv
  (`python docs/audit/<round>/repro/repro_NN_slug.py`), that prints the observed and the expected value and exits
  non-zero when the defect is present. Pin the commit sha in its docstring. Prefer using real project functions on
  tiny synthetic data over re-implementing them.
- **Check kinds** you can use for the effect cards: shape/dtype/scale assertions; perturbation (future data,
  permuted coins, shuffled labels -> metric must collapse to the trivial baseline); round-trip (`inverse(transform(x))`);
  gradient report (`trainer/grad_report.py`) and zero-grad scan; oracle comparison on E1 (known true mean and sd);
  train/eval equality on the same batch; resume equality; one-batch overfit (the model must be able to fit 32 rows);
  counting (`grep`/AST) with the command recorded.
- **Reasoned** findings carry the exact lines and the argument in one paragraph; **suspected** carry what check
  would settle them.
- Record every command you ran and its result in `docs/audit/<round>/LOG.md` (one line each). Report test and lint
  results truthfully: `ruff check src tests`, `pytest -q` on what you touched (you touch no `src/`, so run the docs and
  card tests that read `docs/`: `tests/test_docs_claims.py`, `tests/test_cards.py`).

## 7. Deliverables (all under `docs/audit/<round>/`, `<round>` = `2026-11-<day>-design`)

| file | content |
|---|---|
| `README.md` | one page: scope, commit audited (sha), method, counts (units checked / findings by severity and confidence), what was not reached, how to re-run |
| `TRACE.md` | section 3 |
| `COVERAGE.csv` | section 2, complete |
| `cards/<station>.md` | the effect cards, section 4 |
| `FINDINGS.md` | the ledger. One row per finding: `id` (`N-01`..), station, `file:line`, class (section 5 number), `confidence` (proven / reasoned / suspected), `model impact` (invalidates results / biases / degrades / blocks speed / none), `severity` (blocks model work / slows development / cosmetic, scale of AUDIT-2026-11), evidence link (repro path or command), minimal fix, **fix package** (the group of findings one PR would fix), dependencies on other findings. Evidence only, no arguments in the ledger |
| `repro/` | the reproduction scripts |
| `CLEAN.md` | what you positively verified as correct (proven-clean checks): so the next session does not re-check them. Same evidence rule |
| `BACKLOG.md` | the continuation plan: findings grouped into fix packages in the order they should be done, for each package: goal, files, guard test to add (rule B3), predicted diagnostic movement after the fix (rule C6), the ROADMAP step it unblocks or must precede, estimated size S/M/L; plus `not-reached` modules and the checks that need a GPU smoke job (rule E5) |
| `LOG.md` | commands run, one line each |

`FINDINGS.md` rows that concern the model result already produced (e.g. a decode or split error) must state which
recorded result is affected (`docs/results/...`) and mark it "needs re-read"; never edit a recorded result.

## 8. Registers to update (rules F1, F2, F5)

- `docs/bottlenecks.csv`: one row per finding with model impact (next free `B-` id, `step` = its fix package,
  `cause_status` = verified / suspected / unverified per the evidence). Update an existing row instead of duplicating.
- `docs/suggestions.csv`: your improvement suggestion(s) for the system, and every `unclear` / `stale` place with
  `path:line`, minutes lost and the proposed fix (rule F5; a one-line stale fix is stated, not made).
- `docs/friction_log.csv`: real obstacles you met.
- `docs/STATUS.md`: one row (state `PR open`), file stays under 60 lines (checked by `tests/test_docs_claims.py`).
- Do not touch `ROADMAP.md`; propose its reorder in `BACKLOG.md`. The planner decides.

## 9. Context and budget

- This is a long job. Save as you go: append to `COVERAGE.csv`, `FINDINGS.md`, `LOG.md` after every station, and
  commit (type `docs`) after each station so a cut session loses nothing and a successor can resume from
  `COVERAGE.csv` (rows not `verified-clean` / `findings` are the work left).
- Navigate by `docs/MAP.md` -> card -> symbol; read by line range, never whole large files.
- If you must stop, finish the current station's ledger, update `README.md` counts and `BACKLOG.md`, and say what
  remains. A partial honest audit is acceptable; a complete-looking audit with unchecked rows is not.

## 10. Done criteria and final report

Done when: COVERAGE.csv has no empty status; every finding has confidence, impact and a fix package; BACKLOG.md is
ordered; registers updated; the draft PR is open with "behaviour changes: none".

Final report to the owner, in Arabic, short: what the audit covered (numbers), the top findings by model impact
(ids + one line each, and which recorded results they put in doubt), what was proven vs only reasoned, what was NOT
verified (including everything that needs the GPU or the real artifact), the proposed order of fix packages, your
improvement suggestion(s), and the unclear / stale places you met.

## 11. Do not

- Do not fix, refactor or "tidy" code, tests or results; do not edit `docs/results/`, `docs/jobs/`, `tests/pins/`.
- Do not trust a green test as proof of design correctness (a test can pin the wrong behaviour; check what it pins).
- Do not run or queue training jobs; do not claim speed or quality numbers you did not measure.
- Do not add a finding without file:line; do not merge several defects into one row.
- Do not re-report a closed finding unless you show it is incomplete.
