# BUILD_PLAN — PyTorch re-implementation of NIG-TimeNet v2

Read `docs/STATUS.md` first, then only your own batch below (see "Execution batches", section 0b) — within it,
every step in order. **One branch + one PR per BATCH, not per step** (owner decision, 2026-11-02: this is a
port of an already-fully-specified, already-oracle-verified reference, not exploratory research, so the
per-step PR/CI round-trip of the original draft was cut to one per batch to save time). This does **not** mean
"write the whole batch, then test once": inside a batch, **implement one step, run that step's acceptance
test(s) and get them green, THEN move to the next step** — never write step N+1 on top of an unverified step N.
The reason per-step tests stay mandatory even though this is "just a port, not new design": framework-to-
framework conversion bugs (weight-layout transposes, axis/reduction order, gate order, residual order) are
*silent* — the code runs and looks plausible but is numerically wrong — and a single end-to-end check at the end
of a whole batch cannot localize which of several ported layers is wrong, so a failure there costs more time to
debug than every small test combined would have cost to run. One change type per commit
(`feat`/`fix`/`refactor`/`docs`/`test`) inside the batch's PR is still expected (separate commits, one PR).
Update your row(s) in `docs/STATUS.md` in the same PR. Append every real obstacle to `docs/friction_log.csv`
(columns below) — this is not optional, it is how the plan self-corrects; a review step every 5 steps (now:
every 1-2 batches, see the table) reads it and reorders what follows if needed.

You are a **builder**, not a researcher: no "architecture X is better" claims, no edits to
`crypto-signal-prediction`'s `docs/research/failure_registry.csv` or recorded results, no touching the holdout
lock / time seal. If a step asks you to judge something not resolved below, STOP and write the question into
`docs/friction_log.csv` instead of guessing.

---

## 0. Verify-before-planning (read this before anything else)

### V1 — What `main.ipynb` actually trains by default (it is NOT what `reference/` exports)

Checked directly in `yuosef772424/crypto-signal-prediction` at the pinned commit
`aeafc564f25c78f252936affbd374f68aa49b8d9` (same commit `reference/data/provenance.json` records), and already
written up in `ORACLE_REVIEW.md` (MAJOR-1) in this repo:

| Question | Answer | Evidence |
|---|---|---|
| anti-memorization on or off by default? | **ON** (`anti_memorization=True`) | `workflow/settings.py:160` (`ModelSettings.anti_memorization: bool = True`); `main.ipynb` cell 10 passes `model=ModelSettings(anti_memorization=True, class_only=False)` explicitly — not left at a different value |
| GRU vs Transformer trunk? | Real default = **GRU encoder + level passthrough** (`encoder="gru"`, `level_passthrough=True`, `level_norm="batch"`, `stats_mode="symlog"`, `d_model=64`, `dropout=0.25`, `input_clip=4.0`, `input_noise_std=0.1`, `feature_dropout=0.1`, `linear_path_l2=1e-3`, `head_hidden=64`, `class_head_hidden=32`) | `model/config.py:68-84` (`ANTI_MEMORIZATION_CONFIG`); merged in by `workflow/run.py:267-268` (`if settings.model.anti_memorization: overrides.update(kit.anti_memorization_config)`) |
| Which price targets? Is `close` suspended? | Real default = **`close` suspended** → targets `(high, low)` only, not `(high, low, close)` | `workflow/settings.py:135-140` (`TargetSettings.suspended_targets` docstring: `None` = "close suspended unless target_mode starts with entry_range"); `workflow/run.py:247-249` (`suspended = () if entry_range else ("close",)`); `main.ipynb` cell 10 leaves `target_mode=None, suspended_targets=None` |
| Label smoothing? | Real default = **0.1** (not 0) | `workflow/settings.py:211` (`anti_memorization_label_smoothing: float = 0.1`); `workflow/run.py:316` (`label_smoothing=tr.anti_memorization_label_smoothing if _am else 0.0`) |

`reference/` (`export_reference.py`) instead builds `MODEL_CONFIG` (plain) with only `{d_model: 16, num_layers:
2}` overridden: **Transformer/GQA encoder, 3 targets (high/low/close), `label_smoothing=0`**. This is a real,
reachable configuration of the upstream code (nothing is wrong with the numbers in it — see `ORACLE_REVIEW.md`
"Verified correct"), but `reference/README.md:25-27`'s own claim that this is "exactly as `main.ipynb` does" is
false for `main.ipynb`'s current defaults. `NOTES.md` #9 already discloses the anti-memorization exclusion "as
requested" but never mentions the `close` suspension.

**Decided (owner, chat 2026-11-02): `close` is enabled by default in this repo, always.** This plan does **not**
mirror `main.ipynb`'s `close`-suspension in its own defaults, in either oracle path: `price_targets` defaults to
`(high, low, close)` everywhere in this repo (`ARCHITECTURES`' `default_config`, P32; the second oracle, P01;
the GRU/anti-memorization variant, P33b). This is a deliberate, recorded divergence from
`crypto-signal-prediction`'s current default (which suspends `close` because, there, close's direction has no
measured predictive signal at the TF project's own data/horizon — `workflow/run.py:243-245`'s comment) — not an
oversight and not something a later step may "fix" back to match the TF default without a new owner decision.
The row in the table above stays as the factual description of what `main.ipynb` itself does; it does not
describe this repo's own default anymore.

**Consequence for this plan, per the instruction that triggered this check:** step **P01** below regenerates a
**second** TF oracle, under the real default config (`anti_memorization=True`, GRU trunk, `label_smoothing=0.1`,
**3 targets — `close` kept enabled per the decision above, not suspended**), *before* any GRU/anti-memorization
architecture work happens later in M6. It does not block M2-M5: those build and prove parity for the
**Transformer** path against the oracle that already exists and is already audited (`reference/data/`,
`ORACLE_REVIEW.md` verdict PASS) — which already has `close` enabled, so no change was needed there. The
GRU/anti-memorization path is a second, clearly-separated **registered encoder variant** (M6, P33b) validated
against the *second* oracle P01 produces. Do not blend the two configs in one model or one test.

### V2 — Open `ORACLE_REVIEW.md` findings and how this plan handles them

| Finding | Status | How this plan handles it |
|---|---|---|
| MAJOR-1 (provenance claim vs real default) | Open (doc-only fix suggested, not done) | Handled structurally by V1 above + P01 + P33b, not by editing `reference/README.md` (out of scope: that repo content is not production code of this plan, and the audit PR is still open for the owner to act on) |
| MAJOR-2 — `weights.npz`/`grads.npz` store `trainer/uncertainty_weighted_loss/log_var_{high,low,close}_class` as shape `(1,)` instead of the declared `()` (bug in `export_reference.py`'s `save_npz`, `np.ascontiguousarray` on a 0-d array) | Open | Every oracle-loading step (P07 onward) calls `np.asarray(x).reshape(meta_shape)` before use/compare for exactly these 3 names; never assumes shape equals the `.npy` array's own `.shape`. A loader test (P07) pins this with a comment pointing at `ORACLE_REVIEW.md` MAJOR-2 so it is not "fixed" by accident later without updating the loader |
| MINOR-1 — confirmed 1-ULP cross-machine drift in one `losses.json` value (`task_losses_weighted.low_reg`) | Informational, not a bug | Covered by the tolerance table below (1 ULP ≈ 1.2e-7 relative, inside every `rtol` this plan uses) |

### Tolerance table (binding for every acceptance test below; do not loosen or tighten without a `docs/friction_log.csv` row explaining why)

| Artifact | Comparison | rtol | atol | Why |
|---|---|---|---|---|
| Loaded weights (`weights.npz` → `torch.nn.Parameter`) | exact after `reshape`/`dtype` cast only | 0 | 0 | No arithmetic happens; a diff means the loader is wrong |
| Per-layer / per-sublayer forward output (training=False, same weights, same input) | `torch.allclose` | 1e-5 | 1e-6 | Same tolerance `crypto-signal-prediction`'s own audit uses for a cross-kernel-order TF claim (`docs/research/hardening/audit_2026-10-31.md` C5, "rtol=1e-5, atol=1e-6"); covers GEMM/softmax summation-order differences between TF/oneDNN and PyTorch/MKL, not logic bugs |
| Full model forward (`outputs.npz`) | `torch.allclose` per output key | 1e-5 | 1e-6 | Same reasoning, compounded over more ops; if a key fails, bisect with the per-layer tests (P08-P15) before touching this tolerance |
| Loss scalars / per-task losses (`losses.json`, both scenarios) | `math.isclose` | 1e-5 | 1e-6 | Absorbs MINOR-1's 1 ULP with 4 orders of margin to spare |
| Gradients (`grads.npz`, PRIMARY scenario) | `torch.allclose`, after the MAJOR-2 reshape for the 3 Kendall entries | 1e-4 | 1e-6 | Gradients compound rounding over the whole graph; looser than forward on purpose |
| One optimizer step | gradients only (as above); **post-update weights are NOT compared** | n/a | n/a | AdamW/Adam have framework-specific bias-correction/epsilon placement; matching the update bit-for-bit is not a meaningful parity goal and is explicitly out of scope (see "Out of scope") |

Any test that needs a different tolerance states it explicitly in its own step and says why; it may never be
*looser* than this table without a `docs/friction_log.csv` row.

### 0b. Execution batches (one PR per batch; steps keep their own numbers and their own tests)

The step list (P00-P39) is unchanged below — every step still names its own files/interfaces/tests. What changed
is the PR/review granularity: steps are grouped into 8 batches, each one branch + one PR + one CI run, merged as
a unit. **Inside a batch, steps are still done in listed order, each one's acceptance tests green before the
next step starts** (see the rule at the top of this file). A review step (P05/P16/P20/P24/P30/P36/P39) always
ends its batch, so every batch boundary is already a natural checkpoint.

| Batch | Steps | Milestone | Why this grouping |
|---|---|---|---|
| B1 | P00-P05 | M1 | Pure scaffolding (skeleton, registry, schema, config, CI) — no oracle numerics involved yet, lowest-risk batch, fine to do in one PR |
| B2 | P06-P10 | M2 (part 1) | Data/oracle loaders + the first 3 layer types (InstanceNorm/SymLog, Decomp, PatchEmbedding) — each still gets its own parity test against `reference/data/`; split from B3 so a weight-layout mistake here (the first place `Conv1d`/`Dense` transpose issues can appear, P10) is caught before attention is built on top of it |
| B3 | P11-P16 | M2 (part 2) | Attention + TransformerBlock + readout + NIG heads + full assembly + the M2 review/exit-gate test (P15/P16) — the largest batch, deliberately not split further because P15's end-to-end oracle test is the thing that proves the whole encoder, and splitting it from the layers it assembles would just move the same debugging work across a PR boundary |
| B4 | P17-P20 | M3 | Loss + Kendall + one optimizer step + review — self-contained, depends only on B3's finished model |
| B5 | P21-P24 | M4 | Metrics + trainer loop + run identity + review — no oracle dependency beyond B3/B4's model and losses |
| B6 | P25-P30 | M5 | SplitPlan + TargetCodec + PredictionTable + fixture evaluation + Calibrator + review |
| B7 | P31-P36 (incl. P33b) | M6 | Model contract + `tiny_mlp` + the second-oracle GRU parity (P33b) + `input_skip` seam + probe report + review |
| B8 | P37-P39 | M7 | Rolling/staleness seams + final review |

`docs/STATUS.md`'s table gets one row per **batch** now (not per step); each batch's row lists its step range so
the per-step detail stays traceable back to this file.

---

## 1. Context this plan is informed by (read once, cite by shorthand below)

`crypto-signal-prediction` (source of truth, read-only) is mid-flight on its own hardening plan
(`docs/research/hardening/{audit_2026-10-31.md,PLAN_2026-11.md,STATUS.md}`, decision `D-012`) that names almost
the same structural problems the task brief gave as C1-C8, with a proposed fix for each (cited by section below).
This PyTorch repo does **not** import that code (hard constraint: data prep is not rewritten in torch) and must
not wait for those PRs to merge, but its own design mirrors the *shape* of those fixes in fresh Python, informed
by the TF side's field names so the two projects stay conceptually compatible:

- **C1** (head/target/loss in 4 vocabularies) → TF-side fix: `core/heads.py: HeadSpec(name, kind, output_keys:
  {role: key}, label_key, task_type, label_encoding, default_weight, eval_role)` (`audit_2026-10-31.md:95-106`,
  TF-side step `S21`). This plan's `model/heads.py` (P17) uses the same field names.
- **C2** (3 config systems, silent defaults) → `core/schema_check.py: validate(cfg, schema)`, unknown key = error
  with did-you-mean (`audit_2026-10-31.md:156-165`, `S06`). This plan's `core/config.py` (P04) does the same in
  frozen dataclasses (Python's own `__post_init__`, no separate schema dict needed since there is no legacy
  dict-config to validate against).
- **C3** (no run identity) → `RunIdentity{schema_version, settings_hash, seed, data_fingerprint, code_sha,
  parent_hashes}` (`audit_2026-10-31.md:212-221`, `S07`... actually `S03`/identity PRs). This plan's
  `trainer/identity.py` (P23) uses the same field names.
- **C4** (target codec scattered) → `TargetCodec` registry, `fit/encode/decode/eval_spec`
  (`audit_2026-10-31.md:259-266`, `S22`). This plan's `data_io/target_codec.py` (P26) mirrors it.
- **C5** (no `PredictionTable`) → `predict_table(...) -> PredictionTable`, mandatory columns `timestamp, asset,
  horizon_ns, prediction_made_at, label_available_at, identity_hash, stage, fold, train_end, schema_version`
  (`audit_2026-10-31.md:318-330`, `S09`). This plan's `core/prediction_schema.py` + `workflow/prediction_table.py`
  (P27) use the same columns.
- **C6** (data roles / leakage control as config, not data property) → `core/roles.py: Role = fit | calibration |
  selection | report | sealed`, `SplitPlan` (`audit_2026-10-31.md:394-412`, `S10`/`S12`). This plan's `core/roles.py`
  + `data_io/split_plan.py` (P25) use the same names.
- **C7** (metrics duplicated, selection = loss) → `stats/metrics.py` tier-1, `SelectionMetric` registry
  (`audit_2026-10-31.md:452-459`, `S05`/`S06`). This plan's `stats/metrics.py` + `trainer/selection_metric.py`
  (P21-P22) mirror it.
- **C8** (registries have no shared contract) → `core/registry.py: Registry(kind, contract)`, exact API already
  built and battle-tested on the TF side (`core/registry.py`, read in full during the oracle audit; used by
  `TASK_REGISTRY`/`HEAD_REGISTRY`). **This plan ports that file close to verbatim** (P02) rather than
  re-designing it — it is pure stdlib Python (no TF dependency) and the TF side's own
  `docs/research/hardening/friction_log.csv` row 3 already paid down one ergonomics bug (`get()` dict-API
  mismatch) that this port starts from fixed.
- **D-012** (owner decision, accepted 2026-11-02): shared-namespace `load_into(ns)` exec-loading is a structural
  problem to be removed gradually on the TF side — **this repo never adopts it in the first place**: every module
  here uses normal `from .x import y` imports from day one (no `_loader.py`, no `exec`).
  D-012 point 3 — "a radically different model, or any single part, plugs in by matching the input/output contract
  of the approved model" — is this plan's `core/model_contract.py` (M6, `S23` on the TF side,
  `docs/research/hardening/PLAN_2026-11.md:213-224`): `ModelContract` (input/output spec), `contract_of(model)`,
  `check_contract(model, contract)`, an `ARCHITECTURES` registry with `nig_timenet_v2` default and a `tiny_mlp`
  example. This plan's P31-P33 use those exact names.
- Token-cost evidence for "normal imports from day one, not shared namespace" (`docs/architecture_review.md` §5,
  "Token cost per typical task"): the TF repo pays 20-36k tokens per task *today* with the shared-namespace
  pattern, versus 9-15k estimated *after* migrating to plain imports (steps S0-S10 there). This repo starts at the
  "after" number by construction instead of paying to migrate into it.
- The raw-input-information-loss hypothesis in the task brief is the owner's own hypothesis already named on the
  TF side: `docs/research/MODEL_DEV_PROTOCOL.md` §2-3 ("Representation loses it? Probe skill peaks at layer k and
  drops after"), and a seam for it is already planned there as `S15`/`S07`
  (`docs/research/hardening/PLAN_2026-11.md:171-181`, encoder option `input_skip`, default `"none"`, registered
  variants `concat_each_block`, `dense_skip`, `late_raw`, `no_proj`; probe tool `layer_probe.csv` with a probe on
  the raw input next to every block). **This plan uses the identical option name and the identical 4 variant
  names** (M6, P34-P35) so a result on one side transfers to the other.

None of the above is imported or depended on at runtime; it is cited so this plan's names and shapes do not drift
from the project's own vocabulary. No experiment claims, no edits to `crypto-signal-prediction` anywhere in this
plan.

---

## 2. Repository layout this plan builds (fixed; do not rename without a `friction_log.csv` row + plan update)

```
nig-timenet-torch/
  pyproject.toml                  # package name: nig_timenet_torch, src layout
  src/nig_timenet_torch/
    core/
      registry.py                 # P02
      schema.py  constants.py     # P03
      config.py                   # P04
      roles.py                    # P25
      seeds.py                    # P23
      model_contract.py           # P31
      prediction_schema.py        # P27
    stats/
      metrics.py                  # P21
    data_io/
      npy_dir.py                  # P06  (reads the artifact contract pinned in P01)
      split_plan.py               # P25
      target_codec.py             # P26
    model/
      layers.py                   # P08-P13 (InstanceNorm, SymLog, CausalMultiScaleDecomp, PatchEmbedding,
                                   #           RMSNorm, RelativeGQAttention, SwiGLU, TransformerBlock, LastToken,
                                   #           AttentionPool)
      nig_layers.py                # P14 (NIGHead, OrderedMeans, NIGUncertainty, ConfidenceHead)
      heads.py                     # P14/P17 (HEAD_REGISTRY, HeadSpec, classification head)
      parts.py                     # P34 (ENCODERS registry, input_skip seam)
      builder.py                   # P15 (build_nig_timenet_v2 equivalent, wires layers.py+nig_layers.py+heads.py)
      architectures.py             # P32-P33 (ARCHITECTURES registry: nig_timenet_v2, tiny_mlp)
      probe.py                     # P35 (layer_probe_report)
    trainer/
      tasks.py                     # P17 (TASK_REGISTRY, evidential/regression/classification losses)
      task_weighting.py            # P18 (UncertaintyWeightedLoss)
      trainer.py                   # P18/P22 (Trainer, compute_losses, train_step/eval_step)
      identity.py                  # P23 (RunIdentity)
      selection_metric.py          # P22 (SelectionMetric registry)
    workflow/
      settings.py                  # P04/P23/P37 (RunSettings-equivalent incl. RollingSettings seam)
      prediction_table.py          # P27 (predict_table, writer/reader)
      rolling.py                   # P37 (seam only)
      staleness.py                 # P38 (seam only)
    eval/
      calibrator.py                # P29
  tests/
    fixtures/                      # P01: tiny_dataset.dataset (npy_dir), golden JSON schema dump
    oracle/                        # P07: oracle_loader.py + copies of this repo's own reference/data/* paths used read-only
    test_*.py                      # one file per module above, same stem
  docs/
    BUILD_PLAN.md                  # this file
    STATUS.md
    DATA_ARTIFACT_CONTRACT.md      # P01
    friction_log.csv               # every step appends
  .github/workflows/ci.yml         # P00/P05: CPU-only, `pip install -e .[dev]` + `pytest -q`
```

`docs/friction_log.csv` columns (identical to `crypto-signal-prediction`'s, so the format needs no invention):
`date,step,what_slowed_or_broke,where,minutes_lost,root_cause,fix_idea`.

`docs/STATUS.md` table columns: `Step | Title | State | PR | Note`, states
`todo / in progress / PR open / merged / closed (rejected) / skipped (reason)`. Review steps repeat every 5
merged steps (P05, P10, ... placed explicitly below, not just "every 5th number").

---

## 3. Milestone M1 — skeleton + registry + config + CI

### P00 — Repository skeleton
**Goal.** A pip-installable, CPU-only PyTorch package skeleton with CI, no model code yet.
**Files (create).** `pyproject.toml`; `src/nig_timenet_torch/__init__.py` (empty, just a docstring + `__version__`);
`.github/workflows/ci.yml`; `.gitignore` (`__pycache__/`, `*.egg-info/`, `.venv/`); `tests/__init__.py`.
**Interfaces.** None yet.
**Read first.** Nothing external; this is a fresh skeleton. Look at this repo's own `reference/export_reference.py`
header (lines 1-35) only to copy the determinism pattern (`TF_DETERMINISTIC_OPS`-equivalent for torch:
`torch.use_deterministic_algorithms(True)`, single-thread `torch.set_num_threads(1)`, seeded `torch.manual_seed`)
into a `core/seeds.py` **stub** (full version in P23) used by every later test.
**Commands.**
```bash
pip install -e ".[dev]"   # dev extra: pytest, numpy, torch (CPU wheel)
pytest -q                 # should collect 0 tests, pass
```
**Acceptance tests.** `tests/test_package_imports.py::test_import_package` — `import nig_timenet_torch` succeeds
and `nig_timenet_torch.__version__` is a string.
**Definition of done.** CI workflow runs on push to the PR branch and is green on an empty test suite.
**Do NOT.** Do not add a `data/`, `model/` etc. top-level package yet — only the skeleton. Do not vendor any
TensorFlow or `crypto-signal-prediction` code.

### P01 — Pin the data-artifact contract AND regenerate the oracle for the real default config (V1's "step 0")
**Goal.** Two independent deliverables in one step (both are "read and record reality", no design judgment):
(a) an exact, tested description of the on-disk dataset artifacts this project will consume; (b) a second TF
oracle under `main.ipynb`'s real default settings, so M6's GRU/anti-memorization work has something to check
against later without re-deriving it then.
**Files (create).** `docs/DATA_ARTIFACT_CONTRACT.md`; `tests/fixtures/tiny_dataset.dataset/` (an `npy_dir`
directory: per-key `.npy` files + `meta.pkl` + `manifest.json` — produced by the commands below, then committed
as-is); `tests/fixtures/tiny_split.json` (a JSON dump of `train`/`val`/`test` dict **keys**, array **shapes**
and **dtypes** only — not the data twice). In `crypto-signal-prediction`'s own checkout (NOT copied into this
repo, just run there to produce the second oracle): a small script
`reference/export_reference_default.py` analogous to this repo's existing `reference/export_reference.py`, writing
to `reference/data_default/` in **this** repo (nig-timenet-torch) with its own `README_default.md` stating
plainly "real `main.ipynb` default architecture/training settings: anti_memorization=True, GRU trunk,
label_smoothing=0.1 — EXCEPT price_targets, which this repo keeps at (high, low, close) by owner decision
(2026-11-02), overriding main.ipynb's own close-suspension; see BUILD_PLAN.md V1" — copy
`reference/export_reference.py` verbatim except: `MODEL_OVERRIDES = dict(mns["ANTI_MEMORIZATION_CONFIG"])`
(merged over `MODEL_CONFIG` the same way `workflow/run.py:267-268` does it, i.e. `merged =
dict(mns["MODEL_CONFIG"]); merged.update(mns["ANTI_MEMORIZATION_CONFIG"])`), `price_targets` taken from
`merged["price_targets"]` **unchanged, i.e. `(high, low, close)` — do NOT apply `workflow/run.py:247-249`'s
`close` suspension here**, and `build_target_configs(price_targets, label_smoothing=0.1)` (mirrors
`workflow/run.py:316`, `tr.anti_memorization_label_smoothing`). Do not change anything else (same seed, same
batch, same determinism flags).
**Interfaces.** None (data-gathering step), except the committed contract doc must contain, verbatim, a Python
dict literal per key of `dataset`/`train`/`val`/`test` with `{key: {"dtype": ..., "shape_meaning": ...}}` — this
literal is what P06 and P25-P27 implement against.
**Read first (exact ranges, in `crypto-signal-prediction` at `aeafc564f25c78f252936affbd374f68aa49b8d9`).**
- `data/storage.py:61-110` (`save_dataset_dir`/`load_dataset_dir`: the `npy_dir` on-disk format — one `.npy` per
  numpy-array key, `meta.pkl` for everything else, `manifest.json` written last as the completion marker).
- `workflow/run.py:109-165` (`load_dataset`, `apply_dataset_config`, `make_splits`, `retarget`): the `dataset`
  dict keys actually read (`timeframes`, `base_tf`, `window_sizes`, `feature_order`, `forecast_horizon`, `stride`,
  `reg_target_scale`, `price_norm_mode`, `price_pct_clip`, `split_dates`, `holdout_start`, `higher_tf_mode`) and
  the `train`/`val`/`test` keys (`X_{tf}` per timeframe, `y` nested dict, `reg_target_scale` stamped per split).
- `workflow/training_config.py:6-38` (`build_target_configs`): confirms `y[true_key]` naming (`y_{t}_reg`,
  `y_{t}_class`) matches what P01's split dump must record.
- `core/schema.py` (full file, 46 lines): `LAST_COLUMNS`, `TS_COL`, `LAST_DTYPE`, `TARGET_COLUMNS`, `TARGET_MODES` —
  copy these tuples verbatim into this repo's own `core/schema.py` in P03, citing this file + commit as the source.
- `tests/run_fixture.py` (full file, ~85 lines): `ohlcv(days, seed, start)` and `build_dataset(ns, two_tf=False,
  coins=6, days=70, holdout=None)` — the exact, already-existing, already-tested tiny fixture generator to run,
  unmodified.
**Commands (run inside a `crypto-signal-prediction` checkout at the pinned commit, with `requirements-ci.txt`
installed — this is the ONLY step in the whole plan that runs code in the other repo, and it only reads/exports,
never writes there).**
```bash
python - <<'PY'
import sys; sys.path.insert(0, "docs/research/audit")
import _nbload
from tests.run_fixture import build_dataset
import data, numpy as np, json

ns = _nbload.load_pipeline()
ds = build_dataset(ns, two_tf=False, coins=6, days=70)
data.save_dataset_dir(ds, "tiny_dataset.dataset")   # -> copy this directory into nig-timenet-torch/tests/fixtures/

train, val, test = ns["split_data"](ds, config=ns["CONFIG"])
def shapes(d):
    out = {}
    for k, v in d.items():
        out[k] = {"dtype": str(v.dtype), "shape": list(v.shape)} if hasattr(v, "shape") else \
                 ({kk: {"dtype": str(vv.dtype), "shape": list(vv.shape)} for kk, vv in v.items()}
                  if isinstance(v, dict) else repr(type(v)))
    return out
json.dump({"train": shapes(train), "val": shapes(val), "test": shapes(test)}, open("tiny_split.json", "w"), indent=2)
PY
python reference/export_reference_default.py   # from this repo's reference/, pointed at the other repo's ./upstream
```
**Acceptance tests.** `tests/test_data_contract.py::test_fixture_loads` — `data_io.npy_dir.load(...)` (P06, so this
test is written now and stays red/`xfail` until P06 lands) reads `tests/fixtures/tiny_dataset.dataset/` and gets
back a dict whose keys match `docs/DATA_ARTIFACT_CONTRACT.md` exactly (`set(d) == set(contract_keys)`).
`tests/test_data_contract.py::test_split_schema_matches_doc` — `tiny_split.json`'s keys/shapes match the contract
doc's dict literal.
**Definition of done.** `docs/DATA_ARTIFACT_CONTRACT.md` exists with the dict literal; `tests/fixtures/` is
committed; `reference/data_default/` exists in this repo with its own README, sha256 table (same pattern as
`reference/README.md`), and `provenance.json`; `ORACLE_REVIEW.md`-style sanity (sha256 of two consecutive runs of
`export_reference_default.py` match) is noted in `README_default.md`, not re-audited in full (that is optional
future work, not this step).
**Do NOT.** Do not modify `crypto-signal-prediction` (no commits, no pushes — you only ran a script there and
copied its output out). Do not build the GRU torch module here — that is P33b. Do not make the fixture bigger than
`tests/run_fixture.py`'s defaults; a bigger fixture is a `friction_log.csv` row, not a silent change.

### P02 — `core/registry.py`
**Goal.** Port the exact `Registry`/`Contract`/`Entry` API from `crypto-signal-prediction`'s `core/registry.py`
(already read in full; pure stdlib, no TF dependency, nothing to adapt) — this is every later registry in this
plan (`TASK_REGISTRY`, `HEAD_REGISTRY`, `ENCODERS`, `ARCHITECTURES`, `TARGET_CODECS`).
**Files (create).** `src/nig_timenet_torch/core/registry.py`; `tests/test_registry.py`.
**Interfaces (exact, unchanged from the source — copy, do not redesign).**
```python
class RegistryError(Exception): ...
class DuplicateNameError(RegistryError, ValueError): ...
class UnknownNameError(RegistryError, KeyError, ValueError): ...
class ContractError(RegistryError, ValueError): ...

@dataclass(frozen=True)
class Contract:
    kind: str
    required_meta: tuple[str, ...] = ()
    optional_meta: tuple[str, ...] | None = None
    validator: Callable[[str, Any, Mapping[str, Any]], Iterable[str] | None] | None = None
    def check(self, name, obj, version, meta) -> None: ...

@dataclass(frozen=True)
class Entry:
    name: str; version: Any; obj: Any; meta: Mapping[str, Any]

class Registry:
    def __init__(self, kind: str, contract: Contract | None = None, *, view: Callable[[Entry], Any] | None = None): ...
    def register(self, name: str, obj: Any, *, version: Any, **meta) -> Any: ...
    def decorator(self, name: str, *, version: Any, **meta) -> Callable[[Any], Any]: ...
    def unregister(self, name: str) -> Entry: ...
    def get(self, name: str, default: Any = ...) -> Any: ...
    def __getitem__(self, name: str) -> Any: ...
    def names(self) -> tuple[str, ...]: ...
    def __contains__(self, name) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def keys(self) -> tuple[str, ...]: ...
    def values(self) -> list[Any]: ...
    def items(self) -> list[tuple[str, Any]]: ...
    def entries(self) -> list[Entry]: ...
    def describe(self) -> list[dict]: ...
    def check_all(self) -> list[str]: ...
```
**Read first.** `crypto-signal-prediction/core/registry.py`, full file (208 lines) — already fully read during the
oracle audit; port it with only the module docstring's provenance line changed to point at the source commit.
**Commands.** `pytest tests/test_registry.py -q`.
**Acceptance tests.** Port the behavioural cases directly (write them fresh, they are cheap and exact):
`test_duplicate_name_raises`, `test_unknown_name_raises_with_did_you_mean`, `test_contract_required_meta_enforced`,
`test_decorator_registers_and_returns_object_unchanged`, `test_dict_style_reads_get_items_values`,
`test_check_all_reports_contract_violations`.
**Definition of done.** All 6 tests pass; `Registry` has zero imports beyond stdlib (`ast`-grep in CI: `grep -n
"^import\|^from" src/nig_timenet_torch/core/registry.py` shows only `difflib`, `dataclasses`, `typing`).
**Do NOT.** Do not add a `view=` default that differs from the source (`lambda e: e.obj`). Do not add an overwrite
flag — the source deliberately has none (duplicate registration always raises).

### P03 — `core/schema.py` + `core/constants.py`
**Goal.** Copy the pure data constants this project must agree with the TF side on, verbatim, with a provenance
comment and a pinning test (so nobody "fixes" a typo into a silent incompatibility).
**Files (create).** `src/nig_timenet_torch/core/schema.py`; `src/nig_timenet_torch/core/constants.py`;
`tests/test_schema_pinned.py`.
**Interfaces.**
```python
# schema.py
LAST_COLUMNS: tuple[str, ...]      # = ('last_high','last_low','last_close','timestamp','future_close','future_low_min','future_high_max')
LAST_COLUMN_INDEX: Mapping[str, int]
TS_COL: int
LAST_DTYPE: str                    # 'float64'
TARGET_COLUMNS: tuple[str, ...]    # = ('high','low','close')
TARGET_MODES: tuple[str, ...]      # = ("return","return_close","scaled","magnitude","volnorm","entry_range")
NO_RELATIVE_BASES: tuple[str, ...] # = ("entry_range",)

# constants.py
NIG_ALPHA_DEN_MIN: float           # = 1e-2
```
**Read first.** `crypto-signal-prediction/core/schema.py` (full file, 46 lines) and `core/constants.py` (full
file, 11 lines) at the pinned commit — both already read in full during the oracle audit.
**Commands.** `pytest tests/test_schema_pinned.py -q`.
**Acceptance tests.** `test_last_columns_matches_source` — asserts the literal tuple equals the one quoted in this
step (byte-for-byte); `test_nig_alpha_den_min_matches_source` — `NIG_ALPHA_DEN_MIN == 1e-2`. Both tests' docstrings
quote the source file + commit so a future source change is a deliberate, visible diff here too.
**Definition of done.** Both modules import with zero dependencies beyond `types.MappingProxyType` (stdlib).
**Do NOT.** Do not copy `PRICE_NORM_*` constants yet (not needed until a price-normalization mode is ported; add
them in the step that needs them, not speculatively).

### P04 — `core/config.py` (strict settings)
**Goal.** One strict, frozen-dataclass config primitive used by every later `*Settings`/`*Config` class in this
repo: unknown keys raise, no silent defaults via `.get(...)`.
**Files (create).** `src/nig_timenet_torch/core/config.py`; `tests/test_config.py`.
**Interfaces.**
```python
def from_dict(cls: type, d: Mapping[str, Any], *, strict: bool = True):
    """Build a frozen dataclass `cls` from `d`. strict=True (default): a key in `d` not in `cls`'s fields raises
    ValueError naming the unknown key + a did-you-mean (reuse core.registry's difflib pattern). Nested dataclass
    fields are built recursively. Missing keys use the dataclass field default (no second default source)."""
```
**Read first.** `crypto-signal-prediction/docs/research/hardening/audit_2026-10-31.md:156-165` (C2 structural fix
paragraph — read for the *shape* of the rule, not to copy code: there is no legacy dict-config here to migrate,
so this is a fresh, smaller primitive, not a port).
**Commands.** `pytest tests/test_config.py -q`.
**Acceptance tests.** `test_unknown_key_raises_with_did_you_mean`; `test_nested_dataclass_builds_recursively`;
`test_missing_key_uses_dataclass_default`; `test_strict_false_ignores_unknown_keys` (escape hatch, mirrors the
TF-side `load_config(strict=False)` lesson in `audit_2026-10-31.md:357` — used only by tests that intentionally
feed extra keys, never by production code).
**Definition of done.** Every `*Settings` dataclass introduced later in this plan (P23 `RunSettings`-equivalent,
P37 `RollingSettings`) is built through `from_dict`, never hand-rolled.
**Do NOT.** Do not build a general JSON-schema validator (`audit_2026-10-31.md`'s `core/schema_check.py` is for
TF's *existing* dict configs; this repo has none to retrofit — dataclasses + `__post_init__` are enough).

### P05 — Review step 1 + CI finalize
**Goal.** Close M1: make sure CI actually gates merges, and read back the friction log before M2 starts.
**Files (edit).** `.github/workflows/ci.yml` (require pytest + a `ruff`/`flake8` lint pass, both on CPU only,
matrix = single Python version — pick 3.11 to match `crypto-signal-prediction`'s CI, `requirements-ci.txt` header);
`docs/STATUS.md` (fill P00-P04 rows); `docs/friction_log.csv` (create with header row only if P00-P04 didn't need
one yet).
**Read first.** `docs/friction_log.csv` as it stands (every row appended by P00-P04).
**Commands.** `pytest -q && ruff check src tests`.
**Acceptance tests.** None new; this step's test is that CI is green on the PR that merges P00-P04's branches (if
they weren't already each merged individually — prefer merging each P0x on its own PR as instructed, this step
is a checkpoint, not a batch merge).
**Definition of done.** `docs/STATUS.md` has no `todo` row for P00-P04; `docs/friction_log.csv` has been read and,
if any row suggests reordering (e.g. "P02 took 3x the estimate"), a note is added to this file's own change log
(bottom of document) — do not silently reorder without saying so.
**Do NOT.** Do not add new source modules in a review step.

---

## 4. Milestone M2 — parity of each layer vs the oracle

All steps in this milestone target the **existing** `reference/data/` oracle (Transformer encoder,
`d_model=16`, `num_layers=2`, 3 targets), already verified in `ORACLE_REVIEW.md` (verdict PASS). None of them
touch `reference/data_default/` (that is P33b, M6).

### P06 — `data_io/npy_dir.py` (reads the P01 fixture, nothing else)
**Goal.** A pure-numpy reader for the `npy_dir` format pinned in P01, independent of torch.
**Files (create).** `src/nig_timenet_torch/data_io/npy_dir.py`; `tests/test_npy_dir.py`.
**Interfaces.**
```python
def load(path: str | Path) -> dict:
    """Read an npy_dir dataset (manifest.json + <key>.npy per array + meta.pkl) into a dict with the same keys
    the TF `load_dataset_dir` returns. Raises FileNotFoundError if manifest.json is missing (no partial read:
    `save_dataset_dir` on the TF side renames a temp dir atomically, so a missing manifest means "never finished
    writing", not "corrupt")."""
```
**Read first.** `docs/DATA_ARTIFACT_CONTRACT.md` (P01's output, the single source of truth for keys/shapes now);
`crypto-signal-prediction/data/storage.py:87-110` (`load_dataset_dir`, for the manifest/meta.pkl reading order —
read-only reference, not imported).
**Commands.** `pytest tests/test_npy_dir.py -q`.
**Acceptance tests.** `test_load_matches_contract` — loads `tests/fixtures/tiny_dataset.dataset/` and asserts
`set(result) == set(contract_keys)` and every array's dtype/shape matches `docs/DATA_ARTIFACT_CONTRACT.md`.
`test_missing_manifest_raises_file_not_found`.
**Definition of done.** `tests/test_data_contract.py::test_fixture_loads` (written `xfail` in P01) now passes;
remove its `xfail` marker in this PR.
**Do NOT.** Do not add a `pkl.gz` reader yet (the committed fixture is `npy_dir`; add the other format only if a
later step actually needs it, with its own fixture).

### P07 — Oracle loader (`tests/oracle/oracle_loader.py`)
**Goal.** One module every parity test in P08-P19 imports, so the MAJOR-2 shape fix and the tolerance table live
in exactly one place.
**Files (create).** `tests/oracle/__init__.py`; `tests/oracle/oracle_loader.py`; `tests/oracle/test_oracle_loader.py`.
**Interfaces.**
```python
TOLERANCE = {"forward": dict(rtol=1e-5, atol=1e-6), "loss": dict(rtol=1e-5, atol=1e-6),
             "grad": dict(rtol=1e-4, atol=1e-6)}  # the table in section 0, as code

def load_config() -> dict: ...                 # reference/data/config.json
def load_weights() -> dict[str, "torch.Tensor"]: ...   # reference/data/weights.npz + weights.json, keyed by attr_path;
                                                 # the 3 "trainer/...log_var_*" entries are reshaped () per MAJOR-2
def load_layer_outputs() -> dict[str, "torch.Tensor"]: ...  # reference/data/layer_outputs.npz
def load_layers_meta() -> list[dict]: ...       # reference/data/layers.json
def load_sublayers() -> dict[str, "torch.Tensor"]: ...      # reference/data/sublayers.npz, same MAJOR-2 handling
def load_sublayers_meta() -> list[dict]: ...
def load_outputs() -> dict[str, "torch.Tensor"]: ...        # reference/data/outputs.npz
def load_losses() -> dict: ...                  # reference/data/losses.json (both scenarios)
def load_grads() -> dict[str, "torch.Tensor"]: ...           # reference/data/grads.npz, same MAJOR-2 handling
def load_inputs() -> "torch.Tensor": ...        # reference/data/inputs.npz["input_sequence"]
def load_labels() -> dict[str, "torch.Tensor"]: ...          # reference/data/labels.npz

def assert_close(actual, expected, kind: str, msg: str = ""): ...  # torch.allclose with TOLERANCE[kind], raises
                                                                      # AssertionError with max-abs-diff on failure
```
**Read first.** This repo's own `reference/README.md` (full file — already written by the audit, especially the
"Files" and "Sub-layer internals" tables) and `ORACLE_REVIEW.md` MAJOR-2 + MINOR-1 sections (for the exact 3 key
names to reshape and the exact tolerance rationale).
**Commands.** `pytest tests/oracle/test_oracle_loader.py -q`.
**Acceptance tests.** `test_load_weights_reshapes_kendall_scalars` — the 3 `trainer/.../log_var_*` tensors come
back with `.shape == ()` (torch scalar), not `(1,)`. `test_load_layer_outputs_keys_match_layers_json` —
every `layers.json` `output_keys` entry has a matching array in `load_layer_outputs()`.
**Definition of done.** Every later parity test imports from this module only; no test in P08-P19 opens
`reference/data/*` directly.
**Do NOT.** Do not fix the MAJOR-2 bug in `reference/export_reference.py` itself from here (that file belongs to
the audit PR #1, a separate concern) — only work around it in the loader, with a comment citing
`ORACLE_REVIEW.md`.

### P08 — `InstanceNorm` + `SymLog`
**Goal.** First two layers, parity vs `layer_outputs.npz` key `001_instance_norm__0`/`__1`.
**Files (create).** `src/nig_timenet_torch/model/layers.py` (start the file; later steps append classes);
`tests/model/test_layers_instance_norm.py`.
**Interfaces.**
```python
class InstanceNorm(torch.nn.Module):
    def __init__(self, n_features: int, eps: float = 1e-4, affine: bool = True,
                 clip: float | None = None, stats_mode: str = "full", stats_clip: float = 8.0): ...
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]: ...  # (z, stats)

class SymLog(torch.nn.Module):
    def __init__(self, clip: float | None = 8.0): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```
**Read first.** `crypto-signal-prediction/model/input_norm.py`, full file (89 lines; the formulas: `mean`/`var`
over the time axis, `std = sqrt(var + eps)`, `z = (x - mean) / std`, optional `clip * tanh(z / clip)`, optional
affine `gamma`/`beta`, `stats = concat([mean.squeeze(1), log(std).squeeze(1)])`). This was already read in full
during the oracle audit.
**Commands.** `pytest tests/model/test_layers_instance_norm.py -q`.
**Acceptance tests.** `test_instance_norm_matches_oracle` — build `InstanceNorm(n_features=6, eps=1e-4)`, load its
`gamma`/`beta` from `oracle_loader.load_weights()["instance_norm.gamma"/"instance_norm.beta"]`, run on
`oracle_loader.load_inputs()`, assert both outputs close to `layer_outputs["001_instance_norm__0"]` /
`["001_instance_norm__1"]` at `TOLERANCE["forward"]`.
**Definition of done.** The one test above is green.
**Do NOT.** Do not implement `stats_mode="symlog"`/`"none"` yet (the oracle config uses `"full"`; add the other
modes only when a step actually needs them — P33b for the GRU/anti-memorization variant, which uses `"symlog"`).

### P09 — `CausalMultiScaleDecomp`
**Goal.** Parity vs `layer_outputs.npz` key `002_decomp__0`/`__1`.
**Files (edit).** `src/nig_timenet_torch/model/layers.py` (append); **create** `tests/model/test_layers_decomp.py`.
**Interfaces.**
```python
class CausalMultiScaleDecomp(torch.nn.Module):
    def __init__(self, n_features: int, kernel_sizes: tuple[int, ...] = (3, 5, 9, 17)): ...
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...  # (trend, seasonal); seasonal = x - trend
```
**Read first.** `crypto-signal-prediction/model/decomposition.py`, full file (49 lines): causal left-pad +
`AveragePooling1D` per kernel size, a learned softmax gate over `[last_step, mean_abs_diff]`, `trend =
einsum("bktf,bk->btf", trends, gate)`.
**Commands.** `pytest tests/model/test_layers_decomp.py -q`.
**Acceptance tests.** `test_decomp_matches_oracle` — weights `decomp.gate.kernel`/`.bias` (check the exact
`attr_path` string in `weights.json`, likely `decomp.gate.kernel`), input = `layer_outputs["001_instance_norm__0"]`
(the real upstream input to this layer per `layers.json`'s `inputs` field, not the raw model input), assert both
outputs vs `layer_outputs["002_decomp__0"/"__1"]` at `TOLERANCE["forward"]`.
**Definition of done.** Green test; note in the test docstring which `layers.json` entry supplied the "inputs"
field so the next reader does not have to re-derive it.
**Do NOT.** Do not hand-roll average pooling with a manual loop; use `torch.nn.functional.avg_pool1d` with
explicit left-padding (`torch.nn.functional.pad(x, (0,0,k-1,0), mode="replicate")` then pool) to match the TF
`tf.repeat(x[:, :1, :], k-1, axis=1)` edge behaviour (repeat the first time step, not reflect/zero-pad).

### P10 — `PatchEmbedding`
**Goal.** Parity vs `layer_outputs.npz` key `003_trend_seasonal` (plain concat) and `004_patch_embed`.
**Files (edit).** `model/layers.py` (append); **create** `tests/model/test_layers_patch_embed.py`.
**Interfaces.**
```python
class PatchEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, d_model: int, patch_len: int = 4, stride: int = 2): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...   # left-pad if needed, Conv1d(patch_len, stride) + learned abs pos emb
```
**Read first.** `crypto-signal-prediction/model/patches.py`, full file (41 lines): `num_patches(seq_len,
patch_len, stride)` padding formula (`pad = (stride - (seq_len - patch_len) % stride) % stride`), left-pad by
repeating the first step, `Conv1D(d_model, patch_len, stride, padding="valid")` then `+ pos_emb`.
**Commands.** `pytest tests/model/test_layers_patch_embed.py -q`.
**Acceptance tests.** `test_num_patches_matches_15_tokens_pad_0` — with `seq_len=32, patch_len=4, stride=2`,
`num_patches` returns `(15, 0)` (the exact numbers `reference/README.md:31` states). `test_patch_embed_matches_oracle`
— weights `patch_embed.proj.weight/bias` **transposed** for `torch.nn.Conv1d`'s `(out, in, kernel)` layout vs
Keras `Conv1D`'s `(kernel, in, out)` (see `NOTES.md` #7: `kernel.permute(2, 1, 0)`), `patch_embed.pos` loaded
directly (same `(1, n_tokens, d_model)` shape both frameworks use); input = concat of
`layer_outputs["002_decomp__0"]` and `["002_decomp__1"]` on the last axis (must equal `layer_outputs
["003_trend_seasonal"]` first, assert that as a sub-check); output vs `layer_outputs["004_patch_embed"]` at
`TOLERANCE["forward"]`.
**Definition of done.** Both tests green.
**Do NOT.** Do not assume Keras and PyTorch `Conv1d` weight layout match — they don't; get the permute right here
once, write it as a named helper `keras_conv1d_weight_to_torch(w)` in `model/layers.py` so every later conv (there
are none more in this architecture, but the helper is reusable) does not re-derive it.

### P11 — `RMSNorm` + `RelativeGQAttention`
**Goal.** Parity vs `sublayers.npz` keys `block_1/01_norm1_out`, `block_1/02_attn_q_proj`, `.../03_attn_k_proj`,
`.../04_attn_v_proj`, `.../05_attn_scores_pre_softmax`, `.../06_attn_weights`, `.../07_attn_out`.
**Files (edit).** `model/layers.py` (append); **create** `tests/model/test_layers_attention.py`.
**Interfaces.**
```python
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, epsilon: float = 1e-6): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...   # scale * x * rsqrt(mean(x^2, -1, keepdim) + eps)

class RelativeGQAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, max_rel_pos: int = 16,
                 causal: bool = False, window: int | None = None, dropout: float = 0.0): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    # exposes .last_scores_pre_softmax / .last_attn_weights as instance attributes after forward(), for the
    # sublayer test to read without a second forward pass (mirrors the TF export's tf.nn.softmax capture trick)
```
**Read first.** `crypto-signal-prediction/model/transformer.py:10-91` (`RMSNorm`, `RelativeGQAttention`): no bias
on `wq/wk/wv/wo`, GQA repeat (`k,v` repeated `num_heads // num_kv_heads` times along the head axis), relative
position bias table shape `(2*max_rel_pos+1, num_heads)` gathered by clipped relative distance, scores =
`q@k^T / sqrt(d_h) + bias`, optional causal/window mask (`causal=False, window=None` for this oracle, so the mask
branch is UNREACHABLE for this test but must still exist — see P34), softmax, `ctx = attn @ v`, `wo(ctx)`.
**Commands.** `pytest tests/model/test_layers_attention.py -q`.
**Acceptance tests.** One test per captured sublayer key, all against `oracle_loader.load_sublayers()` with input
`sublayers["block_1/00_block_input"]`: `test_norm1_matches_oracle`, `test_q_proj_matches_oracle` (weight
`block_1.attn.wq.kernel` **transposed** for `torch.nn.Linear`'s `(out, in)` vs Keras `Dense`'s `(in, out)` —
`kernel.T`, see `NOTES.md` #7), `test_k_proj_matches_oracle`, `test_v_proj_matches_oracle`,
`test_scores_pre_softmax_matches_oracle`, `test_attn_weights_matches_oracle`, `test_attn_out_matches_oracle`
(this last one is the full `RelativeGQAttention.forward` call, not a sub-step).
**Definition of done.** All 7 green. Write the `Linear` weight-transpose rule as a named helper
`keras_dense_weight_to_torch(w)` next to the conv helper from P10 — every later `Dense`→`Linear` port reuses it.
**Do NOT.** Do not implement attention via `torch.nn.functional.scaled_dot_product_attention` (it does not expose
pre-softmax scores, which the parity test needs); write the explicit matmul/softmax/matmul form.

### P12 — `SwiGLU` + `TransformerBlock`
**Goal.** Parity vs `sublayers.npz` keys `.../08_residual1`, `.../09_norm2_out`, `.../10_swiglu_out`,
`.../11_block_output`, and `layer_outputs.npz` keys `005_block_1`, `006_block_2`.
**Files (edit).** `model/layers.py` (append); **create** `tests/model/test_layers_transformer_block.py`.
**Interfaces.**
```python
class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, dff: int | None = None): ...   # dff default = round(8*d_model/3/8)*8
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...         # w2(silu(w1(x)) * w3(x))

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, max_rel_pos=16, causal=False, window=None,
                 dropout=0.1, attn_dropout=0.0): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    # x = x + drop1(attn(norm1(x))); x = x + drop2(ffn(norm2(x))); return x   -- dropout is nn.Identity in eval()
```
**Read first.** `crypto-signal-prediction/model/transformer.py:94-149` (`SwiGLU`, `TransformerBlock.call`): the
exact residual wiring (two additions, no other arithmetic) already verified byte-for-byte in `ORACLE_REVIEW.md`
("Verified correct", Check 1).
**Commands.** `pytest tests/model/test_layers_transformer_block.py -q`.
**Acceptance tests.** `test_residual1_matches_oracle`, `test_norm2_matches_oracle`, `test_swiglu_matches_oracle`
(weights `block_1.ffn.w1/w2/w3.kernel`, all transposed per P11's helper), `test_block_output_matches_oracle` (the
sublayer value), `test_transformer_block_forward_matches_layer_outputs` — **call `TransformerBlock.forward` once**,
in `.eval()` mode (no sub-stepping), compare to `layer_outputs["005_block_1"]`; repeat for `block_2` with its own
weights and input (`layer_outputs["004_patch_embed"]` feeds `block_1`; `block_1`'s output feeds `block_2` — read
`layers.json`'s `inputs` field to confirm, do not assume).
**Definition of done.** All green, including both blocks (not just `block_1`).
**Do NOT.** Do not call `.train()` anywhere in this test file — dropout must be identity throughout M2 (the oracle
is `training=False` end to end).

### P13 — Readout: `LastToken`, `AttentionPool`, `FiLM`/`GatedFusion` stubs, trunk assembly
**Goal.** Parity vs `layer_outputs.npz` keys `007_final_norm` .. `017_trunk_drop`.
**Files (edit).** `model/layers.py` (append `LastToken`, `AttentionPool`; add `FiLM`, `GatedFusion` as
**unused-for-now** classes so P34 does not have to touch this file again); **create**
`tests/model/test_layers_readout.py`.
**Interfaces.**
```python
class LastToken(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor: return x[:, -1, :]

class AttentionPool(torch.nn.Module):
    def __init__(self, d_model: int): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...   # softmax(einsum("btd,d->bt", x, q) / sqrt(d)) pooled

class FiLM(torch.nn.Module):          # not wired into the builder until P34; exists so the module doesn't change shape later
    def __init__(self, d_model: int): ...
    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor: ...

class GatedFusion(torch.nn.Module):   # same note as FiLM
    def __init__(self, d_model: int): ...
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
```
**Read first.** `crypto-signal-prediction/model/readout.py`, full file (89 lines, all 4 classes); `model/builder.py:
175-196` (the `_branch` function's readout section only: `final_norm` → `[last_token, attn_pool, stats_proj]` →
`readout_concat` → `readout_fc` → `linear_path` (Flatten + Dense with L2) → `add_linear_path` → the caller's
`trunk_norm`/`trunk_drop`, already read in full during the oracle audit).
**Commands.** `pytest tests/model/test_layers_readout.py -q`.
**Acceptance tests.** One test per `layers.json` index 7-17 (`final_norm`, `last_token`, `attn_pool`, `stats_proj`,
`readout_concat`, `flatten_window`, `readout_fc`, `linear_path`, `add_linear_path`, `trunk_norm`, `trunk_drop`),
each comparing against `layer_outputs[f"{i:03d}_{name}"]` at `TOLERANCE["forward"]`, chained (each test's input is
the previous layer's **oracle** output, not this port's own output, so one wrong layer does not cascade failures
across the whole readout — only the single offending test goes red).
**Definition of done.** 11 green tests.
**Do NOT.** Do not wire `FiLM`/`GatedFusion` into any forward pass yet — they are dead code here on purpose
(`level_film`/`two_tower` are off in both oracles this plan targets); P34 is the first consumer.

### P14 — `NIGHead`, `OrderedMeans`, `NIGUncertainty`, `ConfidenceHead`, classification head
**Goal.** Parity vs `sublayers.npz` `nig_{close,high,low}/*` and `layer_outputs.npz` indices 18-36.
**Files (create).** `src/nig_timenet_torch/model/nig_layers.py`; `src/nig_timenet_torch/model/heads.py` (just the
classification-head function for now; `HEAD_REGISTRY`/`HeadSpec` are P17); **create**
`tests/model/test_nig_layers.py`.
**Interfaces.**
```python
# nig_layers.py
class NIGHead(torch.nn.Module):
    def __init__(self, in_features: int, hidden: int = 128, dropout: float = 0.1,
                 nu_min: float = 0.1, alpha_min: float = 2.0, beta_min: float = 0.01): ...
    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...  # mu, nu, alpha, beta

class OrderedMeans(torch.nn.Module):
    def forward(self, mu_c: torch.Tensor, raw_h: torch.Tensor, raw_l: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    # returns (mu_high, mu_close, mu_low) = (mu_c + softplus(raw_h), mu_c, mu_c - softplus(raw_l))

class NIGUncertainty(torch.nn.Module):
    def __init__(self, unc_max: float | None = 20.0): ...
    def forward(self, nu, alpha, beta) -> tuple[torch.Tensor, torch.Tensor]: ...   # (epistemic, aleatoric)

class ConfidenceHead(torch.nn.Module):
    def __init__(self, in_features: int, hidden: int = 32): ...
    def forward(self, h, nu, alpha, beta) -> torch.Tensor: ...   # 0.05 + 0.93*sigmoid(d2(relu(d1(stop_grad(concat)))))

# heads.py
def classification_head_forward(x: torch.Tensor, fc: torch.nn.Linear, out: torch.nn.Linear) -> torch.Tensor: ...  # Dense(relu)->Dropout->Dense(sigmoid)
```
**Read first.** `crypto-signal-prediction/model/nig_layers.py`, full file (109 lines, all 4 classes — already
read in full during the oracle audit, including `core/constants.NIG_ALPHA_DEN_MIN` usage); `model/heads.py:60-71`
(`build_binary_classification_head`, the exact `Dense(relu)->Dropout->Dense(sigmoid)` shape).
**Commands.** `pytest tests/model/test_nig_layers.py -q`.
**Acceptance tests.** Per `nig_close` (then repeat parametrized over `nig_high`, `nig_low`):
`test_fc_out_matches_oracle` (`sublayers["nig_close/01_fc_out"]`), `test_raw_params_matches_oracle`
(`.../02_raw_params`, columns order `[mu, nu, alpha, beta]` — assert column 0 is linear `mu`, no floor, per
`reference/README.md:150-151`), `test_softplus_floors_match_oracle` (`.../04_softplus_nu/alpha/beta` +
`nu_min/alpha_min/beta_min` added), `test_final_outputs_match_layer_outputs` (the 4 outputs vs
`layer_outputs[f"021_nig_close__{0..3}"]`). Then `test_ordered_means_matches_oracle` (inputs = the 3 heads' raw
`mu`, output vs `layer_outputs["027_ordered_means__{0,1,2}"]`, note the exact unpack order from
`crypto-signal-prediction/model/builder.py:218-219`: `[close, high, low]` in, `(high, close, low)` out — do not
swap). `test_nig_uncertainty_matches_oracle` (vs `028_unc_close__0/__1`, floor via
`core.constants.NIG_ALPHA_DEN_MIN` imported from P03, `unc_max=20.0` clip). `test_confidence_head_matches_oracle`
(vs `030_conf_close`). `test_classification_head_matches_oracle` (vs `029_y_close_class_logits`).
**Definition of done.** All green for `close`; parametrize (`pytest.mark.parametrize`) over `high`/`low` and
confirm those pass too without copy-pasting the test bodies.
**Do NOT.** Do not apply any floor to `mu` — it is linear, by design (`reference/README.md:151`).

### P15 — Full model assembly (`model/builder.py`) + end-to-end parity
**Goal.** Wire P08-P14 into one `torch.nn.Module` and prove the **whole** forward pass matches `outputs.npz`.
**Files (create).** `src/nig_timenet_torch/model/builder.py`; **create** `tests/model/test_builder_end_to_end.py`.
**Interfaces.**
```python
def build_nig_timenet_v2(seq_len: int, n_features: int, *, d_model=128, num_layers=4, num_heads=4,
                          num_kv_heads=2, patch_len=4, stride=2, kernel_sizes=(3,5,9,17), max_rel_pos=16,
                          dropout=0.1, attn_dropout=0.0, causal=False, window=None,
                          price_targets=("high","low","close"), head_types: dict[str, list[str]] | None = None,
                          enforce_order=True, head_hidden=128, nu_min=0.1, alpha_min=2.0, beta_min=0.01,
                          unc_max=20.0, class_head_hidden=64, encoder: str = "transformer",
                          input_skip: str = "none") -> "NigTimeNetV2": ...
    # `encoder` and `input_skip` are accepted and validated here but only "transformer"/"none" are implemented
    # until P34 — any other value raises NotImplementedError naming the step that adds it, NOT a silent no-op.

class NigTimeNetV2(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]: ...   # same 24 keys as the TF model's outputs
    def load_oracle_weights(self, weights: dict[str, torch.Tensor], attr_paths: dict[str, str]) -> None: ...
        # assigns every parameter by the weights.json "attr_path" string, raising on any unmatched parameter
        # (no silent partial load)
```
**Read first.** `crypto-signal-prediction/model/builder.py`, full file (243 lines — already read in full during
the oracle audit): the exact call order (`nig_targets` built first per `price_targets` order, `OrderedMeans` only
when `{"high","low","close"} <= set(nig_targets)` and `enforce_order`, then classification heads via
`build_head_outputs` for every non-`nig_regression` entry of `head_types[t]`); `reference/data/config.json`'s
`constructor_kwargs` (the exact kwargs this oracle was built with — this test's `build_nig_timenet_v2` call must
pass the identical values).
**Commands.** `pytest tests/model/test_builder_end_to_end.py -q`.
**Acceptance tests.** `test_weight_count_matches_oracle` — `len(list(model.parameters())) == 72` (the model-only
count from `weights.json`, excluding the 3 `trainer/`-prefixed Kendall entries). `test_load_oracle_weights_exact`
— every parameter equals its `weights.npz` array exactly (`TOLERANCE` n/a — this is `rtol=0,atol=0` per the
tolerance table). `test_forward_matches_outputs_npz` — run on `oracle_loader.load_inputs()` in `.eval()`, assert
every one of the 24 output keys vs `oracle_loader.load_outputs()` at `TOLERANCE["forward"]`.
**Definition of done.** All 3 green. This is the milestone's exit criterion: **the torch model reproduces the
TF oracle's full forward pass** within the stated tolerance.
**Do NOT.** Do not special-case the comparison per output key with different tolerances to make a borderline key
pass — if one key needs a looser tolerance, that is a `friction_log.csv` row and a conscious, documented table
change in section 0, never a one-off `if key == "...": atol = ...` in the test.

### P16 — Review step 2 (M2 close-out)
**Goal.** Confirm M2's exit criterion (P15) actually holds, and that no earlier per-layer test was "fixed" by
loosening a tolerance.
**Files (edit).** `docs/STATUS.md`, `docs/friction_log.csv` (read + append a trend note), `docs/BUILD_PLAN.md`
(bottom change log, if P06-P15's actual effort diverged from the "≤1 session" assumption — say so, don't silently
replan).
**Commands.** `pytest tests/model -q` (full M2 suite, should be fully green); `grep -rn "atol\s*=\|rtol\s*=" tests/
| grep -v "oracle_loader.py\|1e-5\|1e-6\|1e-4\|0, 0"` — must return nothing (no undocumented tolerance drift).
**Acceptance tests.** The grep above is the test.
**Definition of done.** Zero undocumented tolerances; `docs/STATUS.md` P06-P15 all `merged`.
**Do NOT.** Proceed to M3 if `test_forward_matches_outputs_npz` (P15) is not green — that is a hard gate, not a
"known issue".

---

## 5. Milestone M3 — parity of loss + one train step

### P17 — `trainer/tasks.py` (`TASK_REGISTRY`, evidential/regression/classification losses) + `model/heads.py` (`HEAD_REGISTRY`, `HeadSpec`)
**Goal.** Port the loss math and register both the task types and the head types, using `core.registry.Registry`
(P02) from the start (C1/C8, section 1) — no "4 vocabularies" here on day one.
**Files (create).** `src/nig_timenet_torch/trainer/tasks.py`; **edit** `src/nig_timenet_torch/model/heads.py`
(add `HeadSpec`, `HEAD_REGISTRY`); **create** `tests/trainer/test_tasks.py`, `tests/model/test_heads_registry.py`.
**Interfaces.**
```python
# model/heads.py
@dataclass(frozen=True)
class HeadSpec:
    name: str; kind: str; output_keys: dict[str, str]; label_key: str; task_type: str
    label_encoding: str; default_weight: float; eval_role: str
HEAD_REGISTRY = Registry("head", Contract("head", required_meta=("output_keys", "label_key", "task_type")))

# trainer/tasks.py
TASK_REGISTRY = Registry("task", Contract("task", required_meta=("stat_keys", "supports_sample_weight")))

def evidential_task_loss(y_true, outputs, target_cfg, scheduled_vars, sample_weight=None) -> tuple[torch.Tensor, dict]: ...
def regression_task_loss(y_true, outputs, target_cfg, scheduled_vars, sample_weight=None) -> tuple[torch.Tensor, dict]: ...
def classification_task_loss(y_true, outputs, target_cfg, scheduled_vars, sample_weight=None) -> tuple[torch.Tensor, dict]: ...
def nig_regularizer(error, nu, alpha, beta, normalize=False, stop_grad=True) -> torch.Tensor: ...
def compute_confidence_calibration_loss(y_true, y_pred, confidence, sample_weight=None) -> torch.Tensor: ...
```
**Read first.** `crypto-signal-prediction/trainer/tasks.py`, full file (317 lines — already read in full during
the oracle audit): the NIG NLL formula (`nig_base`), `_floor_alpha`, `nig_regularizer`, the calibration loss, and
the registration calls at the bottom (`register_task_type("evidential", ..., supports_sample_weight=True,
config_keys=(...))` — this plan's `TASK_REGISTRY.register` calls use the same `config_keys`/`stat_keys` lists).
**Commands.** `pytest tests/trainer/test_tasks.py tests/model/test_heads_registry.py -q`.
**Acceptance tests.** `test_evidential_loss_matches_oracle_nig_base` — recompute `nig_base`/`nig_pen`/`mae` for one
target on the oracle's `outputs.npz` + `labels.npz`, compare to `losses.json`'s `stats` entries for that target at
`TOLERANCE["loss"]`. `test_classification_loss_matches_oracle_accuracy`. `test_unknown_task_type_raises` (registry
contract). `test_duplicate_head_registration_raises`.
**Definition of done.** Both test files green.
**Do NOT.** Do not implement `beta_nll`, `huber_weight`, `normalize_reg`, or `alpha_floor_straight_through` — the
oracle's config uses none of them (all default/off); add them only when a step's acceptance test needs them.

### P18 — `trainer/task_weighting.py` (Kendall) + `trainer/trainer.py::compute_losses`
**Goal.** Parity vs `losses.json`'s `total_loss`/`raw_loss`/`task_losses_weighted` for **both** scenarios, and vs
`grads.npz` for the PRIMARY scenario.
**Files (create).** `src/nig_timenet_torch/trainer/task_weighting.py`; `src/nig_timenet_torch/trainer/trainer.py`
(just the `compute_losses` function + a minimal `Trainer.__init__` holding `scheduled_vars`/`loss_weights`/
`uncertainty_layer`; the full `train_step`/`eval_step` loop is P22); **create** `tests/trainer/test_losses_end_to_end.py`.
**Interfaces.**
```python
class UncertaintyWeightedLoss(torch.nn.Module):
    def __init__(self, task_names: list[str]): ...   # one scalar nn.Parameter per name, init 0.0, name "log_var_{t}"
    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor: ...  # sum 0.5*exp(-log_var)*L + 0.5*log_var

def compute_losses(model: torch.nn.Module, x, y, targets_cfg: dict, scheduled_vars: dict[str, float],
                    loss_weights: dict[str, float], uncertainty_layer: UncertaintyWeightedLoss | None,
                    kendall_task_names: list[str], fixed_task_names: list[str]
                    ) -> tuple[torch.Tensor, torch.Tensor, dict, dict, dict]: ...  # total, raw, per_task, stats, constraints
```
**Read first.** `crypto-signal-prediction/trainer/task_weighting.py`, full file (63 lines — already read in full);
`trainer/trainer.py:113-167` (`_compute_losses`, already read in full during the oracle audit: `raw_loss =
sum(per_target_losses)`, Kendall applied only to `kendall_task_names`, `fixed_task_names` added raw, plus
`model.losses`-equivalent L2 regularization term — see next paragraph).
**L2 regularization note (ties to `NOTES.md` #6, already confirmed correct on the TF side: `l2 * sum(w^2)`, no
0.5 factor).** PyTorch has no automatic per-layer `kernel_regularizer`; this port computes it explicitly:
`reg = nig_close_fc_l2 * (nig_close.fc.weight**2).sum() + ... + linear_path_l2 * (linear_path.weight**2).sum()`
for exactly the two regularized layers the oracle config names (`NIGHead.fc` at `l2=1e-5` — the `NIGHead.__init__`
default, not overridden anywhere; `linear_path` at `l2=1e-4` from `MODEL_CONFIG`), summed once, added to both
`total_loss` and `raw_loss` exactly as `trainer/trainer.py:162-165` does.
**Commands.** `pytest tests/trainer/test_losses_end_to_end.py -q`.
**Acceptance tests.** `test_total_loss_matches_oracle_primary_scenario` (`scheduled_vars = {"lambda_reg": 0.05,
"lambda_calib": 0.1}`, `TOLERANCE["loss"]` against `losses.json`'s `"schedules_end ..."` `total_loss.value`).
`test_total_loss_matches_oracle_start_scenario` (`lambda_reg=lambda_calib=0.0`). `test_regularization_term_matches_oracle`
(`model_regularization_losses.value`). `test_grads_match_oracle_primary_scenario` — `torch.autograd.grad(total,
model.parameters())`, compared per-parameter (via the same `attr_path` keying as P15) to `grads.npz` at
`TOLERANCE["grad"]`, with the MAJOR-2 reshape from `oracle_loader` applied to the 3 Kendall grads.
**Definition of done.** All 4 green — this is M3's exit criterion.
**Do NOT.** Do not try to match `losses.json`'s `low_reg` task loss any tighter than `TOLERANCE["loss"]` — that is
the known, confirmed, cross-machine 1-ULP value (MINOR-1); if your run is further off than `1e-5` relative, that
is a real bug, not MINOR-1, and must be root-caused before this step is done.

### P19 — One optimizer step (gradients only, per the tolerance table's explicit carve-out)
**Goal.** Prove the gradient computation is wired identically through a real optimizer call, without chasing
bit-exact post-update weights (explicitly out of scope, see section 0's tolerance table).
**Files (create).** `tests/trainer/test_one_optimizer_step.py`.
**Interfaces.** None new — reuses `compute_losses` (P18) and a plain `torch.optim.AdamW`.
**Read first.** Nothing new beyond P18 — this step is a usage test, not a new module.
**Commands.** `pytest tests/trainer/test_one_optimizer_step.py -q`.
**Acceptance tests.** `test_optimizer_step_runs_and_grads_match_oracle` — build the optimizer with *some*
reasonable hyperparameters (the exact values do not matter, state them in the test: `lr=5e-5, weight_decay=1e-4`,
matching `TRAINER_DEFAULT_CONFIG["optimizer"]` on the TF side, cited for traceability only, not for parity), call
`.zero_grad(); loss.backward(); optimizer.step()`, then assert **before** the `.step()` call that
`param.grad` for every parameter matches `grads.npz` at `TOLERANCE["grad"]` (i.e. re-run P18's grad check right
before stepping, in the same test, to prove the step didn't silently need different gradients), and **after** the
step only assert that every parameter changed (`not torch.equal(old, new)`) and no `NaN`/`Inf` appeared — no
numeric comparison of the updated weights to anything TF-side.
**Definition of done.** Green test; its docstring states explicitly "post-update weights are not and must not be
compared to the TF oracle" so a future contributor does not add that comparison back in.
**Do NOT.** Do not vendor or reimplement TF's AdamW bias-correction formula to chase a bit-match — this is the one
explicitly out-of-scope numeric target in the whole plan.

### P20 — Review step 3 (M3 close-out)
**Goal/Files/Commands.** Same shape as P05/P16: update `docs/STATUS.md`, read+append `docs/friction_log.csv`,
confirm `pytest tests/trainer -q` is fully green, confirm no tolerance drift (same grep as P16).
**Definition of done.** M3's exit criterion (P18) still green; `docs/STATUS.md` P17-P19 `merged`.
**Do NOT.** Start M4 before this passes.

---

## 6. Milestone M4 — trainer, metrics, selection metric, run identity

### P21 — `stats/metrics.py`
**Goal.** One owner for every metric this project computes (C7, section 1): AUC, rank-IC, Pearson/Spearman,
Wilson, pinball, ECE, MAE, pure numpy, no torch/model/trainer imports.
**Files (create).** `src/nig_timenet_torch/stats/metrics.py`; `tests/stats/test_metrics.py`.
**Interfaces.**
```python
def auc(y_true: np.ndarray, y_score: np.ndarray) -> float: ...                 # handles ties; NaN if one class is empty
def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray | None = None) -> float: ...  # per-timestamp cross-sectional Spearman if groups given, else plain
def pearson(a: np.ndarray, b: np.ndarray) -> float: ...
def spearman(a: np.ndarray, b: np.ndarray) -> float: ...
def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]: ...
def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float: ...
def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float: ...
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float: ...
```
**Read first.** `crypto-signal-prediction/docs/research/hardening/audit_2026-10-31.md:452-459` (C7 structural fix
— for the documented tie/NaN policy intent only; there is no single existing TF module to port since the TF side
itself has this duplicated, which is exactly the problem being avoided here by building it once, correctly, from
the start).
**Commands.** `pytest tests/stats/test_metrics.py -q`.
**Acceptance tests.** `test_auc_matches_sklearn` — cross-check `auc` against `sklearn.metrics.roc_auc_score` on
random + tie-heavy arrays (mirrors the TF-side guard's own cross-check, `audit_2026-10-31.md:460-461`).
`test_auc_nan_policy_documented` — a single-class input returns `float("nan")`, not an exception (state this in
the docstring). `test_rank_ic_per_timestamp_matches_manual_groupby`. `test_ece_bins_sum_to_total_count`.
**Definition of done.** All green; every function's docstring states its tie/NaN policy in one line (this is the
"documented tie/NaN policy" the brief and the TF audit both ask for).
**Do NOT.** Do not import `torch` in this module — it must stay usable on plain numpy arrays from any later
package (mirrors the TF side's tier-1 rule: `stats/` "never imports data/model/trainer").

### P22 — `trainer/selection_metric.py` + `trainer/trainer.py::Trainer` (full loop)
**Goal.** A generic trainer (any number of registered tasks) whose checkpoint/early-stopping selection is a
**named, fixed** metric from `stats/metrics.py`, independent of loss weights (C7's second half).
**Files (create).** `src/nig_timenet_torch/trainer/selection_metric.py`; **edit** `trainer/trainer.py` (add
`Trainer.train_step`/`Trainer.eval_step`, wrapping P18's `compute_losses`); **create**
`tests/trainer/test_selection_metric.py`, `tests/trainer/test_trainer_loop.py`.
**Interfaces.**
```python
SELECTION_METRICS = Registry("selection_metric", Contract("selection_metric", required_meta=("higher_is_better",)))
# registered at module import: "val_raw_loss" (default, higher_is_better=False), "val_auc_<head>", "val_pinball_<head>"

class Trainer:
    def __init__(self, model, tasks_cfg: dict, optimizer: torch.optim.Optimizer,
                 selection_metric: str = "val_raw_loss"): ...   # unknown name raises UnknownNameError from P02
    def train_step(self, x, y) -> dict: ...   # one optimizer step, returns scalar logs
    def eval_step(self, x, y) -> dict: ...    # no grad, returns scalar logs + selection metric value
```
**Read first.** `crypto-signal-prediction/docs/research/hardening/PLAN_2026-11.md:105-109` (`S06` Part A — the
exact independence test this step's acceptance test mirrors: "changing a loss weight does not change the
selection metric value for fixed outputs").
**Commands.** `pytest tests/trainer/test_selection_metric.py tests/trainer/test_trainer_loop.py -q`.
**Acceptance tests.** `test_unknown_selection_metric_raises`. `test_changing_loss_weight_does_not_change_selection_metric`
— fix model outputs/labels, compute the selection metric at `loss_weight=1.0` and `loss_weight=5.0` for one task,
assert equal. `test_train_step_reduces_loss_on_fixture` — a few steps on `tests/fixtures/tiny_dataset.dataset`
(via P06) reduce `total_loss` (sanity, not a parity test — no TF oracle exists for a multi-step train loop).
**Definition of done.** All green.
**Do NOT.** Do not default `selection_metric` to anything depending on loss weights (default must be
`"val_raw_loss"`, the TF side's own default, per `crypto-signal-prediction/workflow/settings.py:203`).

### P23 — `core/seeds.py` (full) + `trainer/identity.py` (`RunIdentity`)
**Goal.** C3 (section 1): a persisted run identity that makes resume-with-a-changed-objective fail loudly instead
of silently corrupting a run.
**Files (edit).** `src/nig_timenet_torch/core/seeds.py` (replace P00's stub with the real
`seed_everything(seed: int) -> None`: `random.seed`, `numpy.random.seed`, `torch.manual_seed`,
`torch.use_deterministic_algorithms(True)`, `torch.set_num_threads(1)`); **create**
`src/nig_timenet_torch/trainer/identity.py`; **create** `tests/trainer/test_identity.py`,
`tests/core/test_seeds.py`.
**Interfaces.**
```python
@dataclass(frozen=True)
class RunIdentity:
    schema_version: int; settings_hash: str; seed: int; data_fingerprint: str; code_sha: str
    parent_hashes: tuple[str, ...] = ()
    def to_json(self) -> dict: ...
    @classmethod
    def from_json(cls, d: dict) -> "RunIdentity": ...

def check_resume(saved: RunIdentity, current: RunIdentity, allow_changes: tuple[str, ...] = ("epochs",)) -> None:
    """Raises with a field-level diff if any field outside `allow_changes` differs. `epochs`/`seed`'s run-count
    equivalents are not RunIdentity fields in this design (RunIdentity has no `epochs` field at all, so there is
    nothing to allow-list there by default; the parameter exists for parity with the TF-side design and future
    fields, and is exercised by a test that adds a throwaway field)."""
```
**Read first.** `crypto-signal-prediction/docs/research/hardening/audit_2026-10-31.md:212-221` (C3 structural fix
— field names only, this is a fresh implementation, no TF code to port: there is no equivalent Python object to
copy, `workflow/identity.py` does not exist yet on the TF side either, per `docs/research/hardening/STATUS.md`
row S-for-identity being pre-existing audit PRs 0-4, already merged before this plan started, i.e. the TF-side
identity concept IS live but as a different object shape not read here — mirror only the field *names* from the
audit doc, not any code).
**Commands.** `pytest tests/trainer/test_identity.py tests/core/test_seeds.py -q`.
**Acceptance tests.** `test_resume_with_changed_settings_hash_raises`. `test_resume_with_only_allowed_change_passes`.
`test_seed_everything_gives_identical_first_batch` — two `Trainer`s built with the same seed on the same fixture
produce bit-identical first-batch indices and identical first-step loss; different seeds differ.
**Definition of done.** All green.
**Do NOT.** Do not make `RunIdentity` mutable (frozen dataclass only — it is a fingerprint, not run state).

### P24 — Review step 4 (M4 close-out)
Same shape as P05/P16/P20. **Definition of done.** `pytest tests/stats tests/trainer -q` fully green;
`docs/STATUS.md` P21-P23 `merged`.

---

## 7. Milestone M5 — SplitPlan + PredictionTable + evaluation parity on the fixture

### P25 — `core/roles.py` + `data_io/split_plan.py`
**Goal.** C6 (section 1): roles and leakage control as a property of the data, not a recipe re-derived per
caller.
**Files (create).** `src/nig_timenet_torch/core/roles.py`; `src/nig_timenet_torch/data_io/split_plan.py`;
**create** `tests/data_io/test_split_plan.py`.
**Interfaces.**
```python
# core/roles.py
class Role(enum.Enum):
    FIT = "fit"; CALIBRATION = "calibration"; SELECTION = "selection"; REPORT = "report"; SEALED = "sealed"

# data_io/split_plan.py
@dataclass(frozen=True)
class SplitPlan:
    train_idx: np.ndarray; val_idx: np.ndarray; test_idx: np.ndarray; calibration_idx: np.ndarray
    label_end: np.ndarray; feature_end: np.ndarray    # per-sample, aligned to the full dataset's row order
    role_of: dict[Role, np.ndarray]                   # Role -> index array, additive view over the above

def purge_embargo_split(label_end: np.ndarray, feature_start: np.ndarray, train_end, val_start, val_end,
                         test_start, embargo: "pd.Timedelta") -> SplitPlan: ...
```
**Read first.** `crypto-signal-prediction/docs/research/hardening/audit_2026-10-31.md:394-412` (C6 structural fix
+ its mandatory guard test design — `tests/test_split_purity.py`'s exact assertion:
`max(train.label_end) < min(val.feature_start)` and likewise val/test, including a deliberately leaky splitter
that must be caught); `docs/DATA_ARTIFACT_CONTRACT.md` (P01) for where `label_end`/`feature_end`-equivalent
timestamps come from in the consumed artifacts (if the pinned fixture does not carry them explicitly, derive them
from `timestamp` + `forecast_horizon` the same way the TF side does, and say so in this module's docstring).
**Commands.** `pytest tests/data_io/test_split_plan.py -q`.
**Acceptance tests.** `test_purity_train_before_val_before_test` (the exact assertion above, on the P01 fixture).
`test_leaky_splitter_is_caught` — construct a deliberately leaky `SplitPlan` (val overlapping train's `label_end`)
and assert the purity check (a standalone function, not just the constructor) raises. `test_calibration_role_disjoint_from_fit`.
**Definition of done.** All green.
**Do NOT.** Do not implement `rolling_splits` here — that is M7 (P37); this step is the single global split only.

### P26 — `data_io/target_codec.py` (`TargetCodec` registry)
**Goal.** C4 (section 1): one place that encodes/decodes a target, registered, `fit/encode/decode/eval_spec`.
**Files (create).** `src/nig_timenet_torch/data_io/target_codec.py`; **create** `tests/data_io/test_target_codec.py`.
**Interfaces.**
```python
TARGET_CODECS = Registry("target_codec", Contract("target_codec", required_meta=("supports_relative",)))

class ReturnCodec:   # registered under "return"
    def fit(self, train: dict) -> dict: ...          # -> state (nothing learned for plain return; still implements the interface)
    def encode(self, raw: np.ndarray, state: dict) -> np.ndarray: ...
    def decode(self, pred: np.ndarray, state: dict, anchors: np.ndarray) -> np.ndarray: ...
    def eval_spec(self, state: dict) -> dict: ...
```
**Read first.** `crypto-signal-prediction/docs/research/hardening/audit_2026-10-31.md:259-266` (C4 structural fix
+ its guard test design: `decode(encode(x)) == x` property test).
**Commands.** `pytest tests/data_io/test_target_codec.py -q`.
**Acceptance tests.** `test_return_codec_round_trip` — `decode(encode(x)) == x` on synthetic candle data (property
test, `hypothesis` library optional — plain parametrized random seeds are enough). `test_unregistered_mode_raises`.
**Definition of done.** Green; only `"return"` is registered (the oracle's price targets use plain returns — see
`reference/data/config.json`'s target config). Do not register `"entry_range"`/`"scaled"`/etc. speculatively.
**Do NOT.** Do not let `decode` read anything off `state` that `fit` did not put there — the whole point of this
component is that its state is self-contained and inspectable.

### P27 — `core/prediction_schema.py` + `workflow/prediction_table.py`
**Goal.** C5 (section 1): one canonical, schema-checked prediction table, one batched predict function.
**Files (create).** `src/nig_timenet_torch/core/prediction_schema.py`;
`src/nig_timenet_torch/workflow/prediction_table.py`; **create** `tests/workflow/test_prediction_table.py`.
**Interfaces.**
```python
# core/prediction_schema.py
PREDICTION_TABLE_COLUMNS = ("timestamp", "asset", "horizon_ns", "prediction_made_at", "label_available_at",
                            "identity_hash", "stage", "fold", "train_end", "schema_version")
def validate_table(df: "pd.DataFrame") -> None: ...   # raises listing every missing/extra/mistyped column

# workflow/prediction_table.py
def predict_table(model: torch.nn.Module, x: torch.Tensor, meta: dict, identity: "RunIdentity",
                   batch_size: int = 512) -> "pd.DataFrame": ...
def write_table(path, df: "pd.DataFrame", identity: "RunIdentity") -> None: ...   # + path + ".meta.json"
def read_table(path) -> "pd.DataFrame": ...
```
**Read first.** `crypto-signal-prediction/docs/research/hardening/audit_2026-10-31.md:318-330` (C5 structural fix
+ both guard tests: the O(1)-model-calls performance guard and the leakage guard
`label_available_at >= timestamp`, monotone per asset).
**Commands.** `pytest tests/workflow/test_prediction_table.py -q`.
**Acceptance tests.** `test_predict_table_schema_valid`. `test_predict_table_is_batched_not_per_asset` — wrap the
model's `forward` in a call counter, run on a 5-asset and a 50-asset fixture slice, assert the call count is equal
for both (O(1) in asset count, mirrors `audit_2026-10-31.md:331-334`'s exact guard). `test_no_row_with_label_available_at_gt_train_end_in_train_fold`
(mirrors C5's leakage test, using P25's `SplitPlan`).
**Definition of done.** All green.
**Do NOT.** Do not add `decision_*`/`outcome_*` optional columns yet (C5 names them for a feedback loop that is
out of scope here — see "Out of scope" at the end).

### P28 — Evaluation parity on the fixture
**Goal.** Exercise P06 (data) + P15 (model) + P26 (codec) + P27 (prediction table) together end to end on the
committed fixture — this is an **internal consistency** test, not a cross-framework parity test (no TF oracle
exists for this exact pipeline composition).
**Files (create).** `tests/integration/test_eval_on_fixture.py`.
**Commands.** `pytest tests/integration/test_eval_on_fixture.py -q`.
**Acceptance tests.** `test_full_pipeline_runs_on_fixture` — load `tests/fixtures/tiny_dataset.dataset` (P06),
build a `SplitPlan` (P25), build the oracle-config model (P15) with **freshly initialized** weights (not the
oracle's — there is nothing to compare numerically here), run `predict_table` (P27), decode with `ReturnCodec`
(P26), compute `stats.metrics.auc`/`mae` (P21) on the result, assert the table validates (`validate_table`) and
every metric is a finite float (not NaN, not inf) — this is a wiring smoke test, explicitly not a numeric claim
about model quality.
**Definition of done.** Green; the test's docstring states explicitly that it asserts wiring, not model skill.
**Do NOT.** Do not assert any specific metric value or threshold here — that would be an implicit, undocumented
experiment claim, which this plan's own rules forbid.

### P29 — `eval/calibrator.py` (`Calibrator`, role-checked)
**Goal.** C6's second half: a registered, role-checked learned-state component, `Calibrator` as the first
instance.
**Files (create).** `src/nig_timenet_torch/eval/calibrator.py`; **create** `tests/eval/test_calibrator.py`.
**Interfaces.**
```python
class Calibrator:
    def fit(self, scores: np.ndarray, labels: np.ndarray, *, role: Role) -> dict: ...  # raises if role != Role.CALIBRATION
    def apply(self, scores: np.ndarray, state: dict) -> np.ndarray: ...
```
**Read first.** `crypto-signal-prediction/docs/research/hardening/audit_2026-10-31.md:394-412` (the
`fit(data, *, role_required)` pattern, "a calibrator requires `calibration`, never `fit`").
**Commands.** `pytest tests/eval/test_calibrator.py -q`.
**Acceptance tests.** `test_fit_on_wrong_role_raises` — `Calibrator().fit(..., role=Role.FIT)` raises.
`test_fit_on_calibration_role_succeeds_and_applies`.
**Definition of done.** Green; isotonic regression only (mirrors the TF side's `calibration_method="isotonic"`
default) — Platt is not implemented (not needed until a step asks for it).
**Do NOT.** Do not let any other component in this repo call `.fit` on a `Calibrator` with `train`-role data —
enforced by the raise above, not by convention.

### P30 — Review step 5 (M5 close-out)
Same shape as prior reviews. **Definition of done.** `pytest tests/data_io tests/workflow tests/eval
tests/integration -q` fully green; `docs/STATUS.md` P25-P29 `merged`.

---

## 8. Milestone M6 — model contract + plug-in example + input_skip seam + probe report

### P31 — `core/model_contract.py`
**Goal.** D-012 point 3 / TF-side `S23` (section 1): a declared input/output contract any architecture must
satisfy, checked mechanically, not by convention.
**Files (create).** `src/nig_timenet_torch/core/model_contract.py`; **create** `tests/core/test_model_contract.py`.
**Interfaces.**
```python
@dataclass(frozen=True)
class InputSpec:
    name: str; shape: tuple[int | None, ...]; dtype: str   # None = batch dim

@dataclass(frozen=True)
class OutputSpec:
    name: str; shape: tuple[int | None, ...]; dtype: str; head_kind: str   # head_kind: how to decode it

@dataclass(frozen=True)
class ModelContract:
    inputs: tuple[InputSpec, ...]; outputs: tuple[OutputSpec, ...]

def contract_of(model: torch.nn.Module, sample_input: torch.Tensor) -> ModelContract: ...  # runs one forward pass to read real shapes
def check_contract(model: torch.nn.Module, contract: ModelContract, sample_input: torch.Tensor) -> None: ...
    # raises with a precise field-level diff on any mismatch
```
**Read first.** `crypto-signal-prediction/docs/research/hardening/PLAN_2026-11.md:213-224` (`S23`'s exact
description: "`ModelContract` = input spec ... + output spec ... extracted from the APPROVED model's plan;
`contract_of(model)` reads it from any built ... model; `check_contract(model, contract)` raises with a precise
diff").
**Commands.** `pytest tests/core/test_model_contract.py -q`.
**Acceptance tests.** `test_contract_of_matches_nig_timenet_v2` — `contract_of` on P15's model returns the 24
expected output names with the right `head_kind` per `HeadSpec.kind` (P17). `test_check_contract_raises_on_mismatch`
— feed a contract with one wrong output shape, assert the diff message names that exact field.
**Definition of done.** Green.
**Do NOT.** Do not make `ModelContract` require a specific architecture's internals — it must describe only the
boundary (names/shapes/dtypes), never reach into a model's layers.

### P32 — `ARCHITECTURES` registry + `nig_timenet_v2` registered
**Goal.** Make P15's model the first, default entry of a pluggable registry.
**Files (create).** `src/nig_timenet_torch/model/architectures.py`; **create** `tests/model/test_architectures.py`.
**Interfaces.**
```python
ARCHITECTURES = Registry("architecture", Contract("architecture", required_meta=("build_fn", "default_config")))
# ARCHITECTURES.register("nig_timenet_v2", build_nig_timenet_v2, version=1, build_fn=build_nig_timenet_v2,
#                         default_config={...the oracle's constructor_kwargs...})
```
**Read first.** Nothing new — this is wiring P15 + P02.
**Commands.** `pytest tests/model/test_architectures.py -q`.
**Acceptance tests.** `test_default_architecture_builds_and_matches_contract` — `ARCHITECTURES.get("nig_timenet_v2")`
builds, `check_contract` (P31) passes. `test_architecture_default_is_nig_timenet_v2` (a `ModelSettings`-equivalent
default field, mirrors `S23`'s `ModelSettings.architecture` default).
**Definition of done.** Green; P15's end-to-end oracle test (already green) still passes unchanged — this step
only adds a registration, it does not touch `build_nig_timenet_v2`'s body.
**Do NOT.** Regress P15's parity tests — run them in this step's CI too, not just the new ones.

### P33 — `tiny_mlp` example architecture (plug-in proof)
**Goal.** Prove, per D-012 point 3, that "a radically different model ... plugs in by matching the contract,
then trains with the same trainer and is evaluated by the same evaluation code" — with a second, trivial
architecture, not a design decision about the real model.
**Files (edit).** `model/architectures.py` (append); **create** `tests/model/test_tiny_mlp_plugin.py`.
**Interfaces.**
```python
class TinyMLP(torch.nn.Module):
    def __init__(self, seq_len: int, n_features: int, price_targets=("high","low","close"), hidden: int = 32): ...
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]: ...   # same 24-key output shape as nig_timenet_v2, via the same HEAD_REGISTRY heads
# ARCHITECTURES.register("tiny_mlp", build_tiny_mlp, version=1, ...)
```
**Read first.** `crypto-signal-prediction/docs/research/hardening/PLAN_2026-11.md:213-224` (the `tiny_mlp`
requirement, same paragraph as P31).
**Commands.** `pytest tests/model/test_tiny_mlp_plugin.py -q`.
**Acceptance tests.** `test_tiny_mlp_matches_contract` (P31's `check_contract`). `test_tiny_mlp_one_epoch_smoke` —
trains one epoch on `tests/fixtures/tiny_dataset.dataset` through the **same** `Trainer` (P22) and the **same**
`predict_table`/evaluation path (P27-28) used for `nig_timenet_v2`, asserts it runs to completion with finite
loss (again a wiring smoke test, not a quality claim).
**Definition of done.** Green; `docs/model_plugin_howto.md` written (one page: "how to add a model: implement
`forward` returning the `HeadSpec`-named keys, register it, done") — the brief's "document how to add a model" is
this file.
**Do NOT.** Make `TinyMLP` architecturally interesting — it must stay trivial (a flatten + a couple of `Linear`
layers); its only job is proving the seam, not being a candidate model.

### P33b — Second oracle parity: GRU/anti-memorization encoder variant (closes V1)
**Goal.** Validate a GRU-trunk, level-passthrough encoder against the **second** oracle P01 produced
(`reference/data_default/`), closing the loop V1 opened. This is the only step in the whole plan that targets
`reference/data_default/` instead of `reference/data/`.
**Files (edit).** `model/layers.py` (add a minimal GRU-trunk forward path mirroring
`crypto-signal-prediction/model/builder.py`'s `encoder == "gru"` branch: `RMSNorm` → `torch.nn.GRU` → residual
`Dropout` add, repeated `num_layers` times — read `model/builder.py:166-172` for the exact residual wiring);
`model/layers.py` (add `level_passthrough`'s `SymLog`-based extra channel concat, already have `SymLog` from
P08); **create** `tests/model/test_gru_encoder_vs_default_oracle.py`.
**Read first.** `crypto-signal-prediction/model/builder.py:107-196` (the whole `_branch` function, this time
reading the `encoder=="gru"` and `level_passthrough`/`level_norm` branches that P08-P15 skipped because the first
oracle never exercises them); `reference/data_default/config.json`'s `constructor_kwargs` (P01's output — the
exact values this step's model must be built with: `encoder="gru"`, `stats_mode="symlog"`,
`level_passthrough=True`, `level_norm="batch"`, `d_model=64`, etc.).
**Commands.** `pytest tests/model/test_gru_encoder_vs_default_oracle.py -q`.
**Acceptance tests.** Same shape as P15's end-to-end test but pointed at `reference/data_default/`:
`test_weight_count_matches_default_oracle`, `test_forward_matches_default_oracle_outputs` at `TOLERANCE["forward"]`.
**Definition of done.** Green. `ARCHITECTURES` (P32) gains no new top-level entry here — `encoder` stays a
constructor kwarg of `nig_timenet_v2`/`build_nig_timenet_v2` (mirrors the TF side: anti-memorization is a config
preset over the same builder, not a different architecture name) — register the anti-memorization preset as a
named `default_config` variant (`ARCHITECTURES.entry("nig_timenet_v2").meta["anti_memorization_config"] = {...}`
or an equivalent constant in `model/architectures.py`, not a second registry entry.
**Do NOT.** Do not merge this step's GRU path into `build_nig_timenet_v2`'s default call signature — it must be
opt-in via explicit kwargs, default behavior (P15/P32) must be unaffected (same rule as the TF side: "a new
option's default must reproduce the old behavior exactly").

### P34 — `input_skip` seam on the encoder
**Goal.** The exact seam named in the brief and already named on the TF side (`S15`): registered variants, default
`"none"` = today's graph unchanged (checkpoint-compatible).
**Files (create).** `src/nig_timenet_torch/model/parts.py`; **edit** `model/builder.py` (accept and wire
`input_skip`, currently a `NotImplementedError` placeholder from P15); **create**
`tests/model/test_input_skip.py`.
**Interfaces.**
```python
INPUT_SKIP_VARIANTS = Registry("input_skip", Contract("input_skip", required_meta=("build_fn",)))
# registered: "none" (identity, default), "concat_each_block", "dense_skip", "late_raw", "no_proj"
# exact semantics per crypto-signal-prediction/docs/research/hardening/PLAN_2026-11.md:171-177:
#   concat_each_block: normalized input tokens concatenated to every TransformerBlock input through its own small projection
#   dense_skip: DenseNet-style — each block sees all previous block outputs + input
#   late_raw: normalized last-k bars concatenated into the readout/trunk
#   no_proj: blocks run at the input width, no step/patch projection (only sane when n_features is small)
```
**Read first.** `crypto-signal-prediction/docs/research/hardening/PLAN_2026-11.md:171-181` (`S15`/`S16r`, full
paragraph, already quoted above — read again here for the causality requirement: "no future bar reaches step t").
**Commands.** `pytest tests/model/test_input_skip.py -q`.
**Acceptance tests.** `test_input_skip_none_matches_p15_oracle_exactly` — with `input_skip="none"`, P15's full
end-to-end oracle test still passes bit-for-bit identical (same weight shapes, same names) — this is the
checkpoint-compatibility guarantee, written as a regression test here. `test_each_variant_builds_and_runs` —
parametrized over the 4 non-default variants, asserts the model builds and a forward pass produces finite
output (no oracle exists for these — they are new paths). `test_causality_no_future_leakage` — for each variant,
zero out all time steps after `t` in the input and assert the output at position `<= t` is unchanged (the
standard causality probe, mirrors `tests/test_no_lookahead.py`'s intent on the TF side).
**Definition of done.** All green.
**Do NOT.** Do not make any variant the default — `"none"` stays default forever per the checkpoint-compatibility
rule; do not run or claim results from any variant (that is `E-arch-001` on the TF side, an experiment, out of
scope here).

### P35 — Per-layer linear-probe report
**Goal.** The tool that actually tests the owner's hypothesis (brief: "early Dense projection may destroy
information") — a report, not a verdict; mirrors TF-side `S07`'s `layer_probe.csv`.
**Files (create).** `src/nig_timenet_torch/model/probe.py`; **create** `tests/model/test_probe.py`.
**Interfaces.**
```python
def layer_probe_report(model: torch.nn.Module, x: torch.Tensor, y: np.ndarray, *,
                        probe_points: list[str]) -> "pd.DataFrame":
    """For each name in `probe_points` (module names inside `model`, e.g. "raw_input", "instance_norm",
    "patch_embed", "block_1", "block_2", "trunk_norm"), fits a ridge probe (closed-form, numpy only — no sklearn
    dependency needed) from that layer's flattened activation to `y`, 5-fold CV, reports mean R^2 (or AUC if `y`
    is binary) +/- std. "raw_input" probes the model's raw input directly (flattened last-k bars), so it is
    always in the table next to every learned layer, per the owner's hypothesis."""
```
**Read first.** `crypto-signal-prediction/docs/research/MODEL_DEV_PROTOCOL.md` §2-3 (full file, 81 lines,
already read in full: "linear-probe skill per layer", "Probe skill peaks at layer k and drops after -> the later
layers or the readout destroy information"); `docs/research/hardening/PLAN_2026-11.md:116-129` (`S07`'s
`layer_probe.csv` columns and the exact probe points: "RAW input ... next to `step_proj`/`patch_embed`, each
TransformerBlock, readout, trunk").
**Commands.** `pytest tests/model/test_probe.py -q`.
**Acceptance tests.** `test_probe_report_includes_raw_input_row`. `test_probe_report_runs_on_fixture` — on
`tests/fixtures/tiny_dataset.dataset` with P15's model, produces a `DataFrame` with one row per `probe_points`
entry, each a finite number — again wiring, not a claim about where information is lost (that reading is M7-and-
beyond research work, explicitly out of scope for this plan to perform or assert).
**Definition of done.** Green; `docs/model_plugin_howto.md` (from P33) gets one added paragraph: "how to add a
probe point" (name a module, it appears in the next report run).
**Do NOT.** Do not draw or write any conclusion from the fixture's probe numbers (6 synthetic coins, 70 days — far
too small to mean anything) — this step delivers the *tool*, never a *finding*.

### P36 — Review step 6 (M6 close-out)
Same shape as prior reviews, plus: confirm P15's and P33b's oracle parity tests are both still green (two
oracles, two green suites) after P31-P35's refactors. **Definition of done.** `pytest tests/model -q` (now the
largest suite) fully green; `docs/STATUS.md` P31-P35 (and P33b) `merged`.

---

## 9. Milestone M7 — rolling-study driver seam

### P37 — `RollingSettings` + `workflow/rolling.py` (seam only)
**Goal.** Per the hard constraint ("keep room — seams only, not implementations"): the *shape* of a rolling
retrain loop, with no experiment run through it.
**Files (create).** `src/nig_timenet_torch/workflow/rolling.py`; **edit** `workflow/settings.py` (add
`RollingSettings`, default `None` = today's single split, unchanged); **create**
`tests/workflow/test_rolling_seam.py`.
**Interfaces.**
```python
@dataclass(frozen=True)
class RollingSettings:
    mode: str = "expanding"            # "expanding" | "sliding"
    train_span: "pd.Timedelta" = ...
    test_span: "pd.Timedelta" = ...
    step: "pd.Timedelta" = ...
    embargo: "pd.Timedelta" = ...
    max_folds: int | None = None
    retrain_every_k: int = 1
    seeds: tuple[int, ...] = (0,)

def rolling_folds(dataset, settings: RollingSettings) -> list[SplitPlan]: ...   # uses P25's SplitPlan.purge_embargo_split per fold
def run_rolling_study(dataset, settings: RollingSettings, train_fn, predict_fn, parent_identity) -> list["pd.DataFrame"]:
    """Loops `rolling_folds`; each fold = `train_fn`/`predict_fn` with its own `RunIdentity(fold=i, seed=s,
    data_cut=fold.test_start, parent=parent_identity.settings_hash)` (P23) and its own `predict_table` write
    (P27). NOT CALLED by any test with real training — see acceptance tests."""
```
**Read first.** `crypto-signal-prediction/docs/research/hardening/PLAN_2026-11.md:159-166` (`S13`, the
`RollingSettings` field list and the fold-loop description, already quoted in section 1).
**Commands.** `pytest tests/workflow/test_rolling_seam.py -q`.
**Acceptance tests.** `test_rolling_settings_default_is_none_equivalent_to_single_split` — no behavior change
when `RollingSettings` is absent (checkpoint/config compatibility, same "new option defaults to old behavior"
rule used everywhere else in this plan). `test_rolling_folds_produces_non_overlapping_purged_folds` — on 2-3 tiny
synthetic folds (numbers only, `train_fn`/`predict_fn` are `lambda *_: None` stubs — this test proves the fold
*geometry* is correct, not that training works), assert each fold's `SplitPlan` passes P25's purity check and
folds advance monotonically in time.
**Definition of done.** Green; `run_rolling_study` exists and type-checks but no test calls it with a real
`train_fn` (that would be `E-roll-001`, an experiment, explicitly out of scope).
**Do NOT.** Do not run or write a card for an actual rolling-retrain experiment — this step is the scaffolding
only, per the hard constraint.

### P38 — `workflow/staleness.py` (seam only)
**Goal.** The companion seam: a metric-vs-distance-from-train_end computation and a train-once-vs-retrain-every-k
comparison *function*, with no experiment run.
**Files (create).** `src/nig_timenet_torch/workflow/staleness.py`; **create** `tests/workflow/test_staleness_seam.py`.
**Interfaces.**
```python
def staleness_curve(prediction_tables: list["pd.DataFrame"], metric_fn: Callable[["pd.DataFrame"], float]
                     ) -> "pd.DataFrame":
    """One row per (days_since_train_end, metric value), `metric_fn` typically a `stats.metrics` function applied
    per-row-group. Pure function over already-written PredictionTables (P27) — does not train anything."""

def label_permutation_null(table: "pd.DataFrame", metric_fn, n_perm: int = 200, seed: int = 0) -> np.ndarray:
    """Shuffles the label column `n_perm` times (same `seed`-derived RNG every call, for reproducibility) and
    returns the null distribution of `metric_fn` — the standard null this plan's own tolerance/acceptance
    philosophy (section 0) and the TF side's `MODEL_DEV_PROTOCOL.md` §4 both require before any "better" claim."""
```
**Read first.** `crypto-signal-prediction/docs/research/hardening/PLAN_2026-11.md:159-169` (`S13`/`S14r`'s
`staleness.py` description, already quoted in section 1); `MODEL_DEV_PROTOCOL.md` §4 ("Evidence rules": "every
diagnostic difference is reported with an effect size and a null").
**Commands.** `pytest tests/workflow/test_staleness_seam.py -q`.
**Acceptance tests.** `test_staleness_curve_runs_on_synthetic_tables`. `test_label_permutation_null_is_reproducible_with_seed`
(same seed → identical null array; different seed → different).
**Definition of done.** Green.
**Do NOT.** Do not call either function from a step that "concludes" anything about retraining frequency — there
is no real multi-fold trained model anywhere in this plan to feed them meaningfully yet; they are tested on
synthetic `PredictionTable`s only.

### P39 — Final review: STATUS wrap-up, risks, scope, open decisions
**Goal.** Close the plan out as a document, not just as code.
**Files (edit).** `docs/STATUS.md` (final pass, every row `merged`/`skipped (reason)`); `docs/friction_log.csv`
(final read-through, one summary row if a pattern repeated); `docs/BUILD_PLAN.md` (append the "Risks", "Out of
scope" and "Open owner decisions" sections below if they are not already the tail of this file — they are, see
below; this step is to re-read them against what actually happened and correct anything that drifted, with a
dated note, never a silent rewrite of history).
**Commands.** `pytest -q` (whole suite); `ruff check src tests`.
**Acceptance tests.** Full green suite is the test.
**Definition of done.** Every P-step P00-P38 is `merged` or `skipped (reason)` in `docs/STATUS.md`; this document
and `docs/STATUS.md` agree.
**Do NOT.** Do not add new scope here — this step is a close-out, not a new milestone.

---

## 10. STATUS table (seed for `docs/STATUS.md`; keep the two files in sync — this copy is read-only reference)

| Step | Title | State | PR | Note |
|---|---|---|---|---|
| P00 | Repo skeleton | todo | | |
| P01 | Data-artifact contract + second (real-default) oracle | todo | | needs the other repo's checkout at the pinned commit; V1 |
| P02 | core/registry.py | todo | | |
| P03 | core/schema.py + constants.py | todo | | |
| P04 | core/config.py | todo | | |
| P05 | Review 1 + CI finalize | todo | | M1 close |
| P06 | data_io/npy_dir.py | todo | | needs P01 |
| P07 | Oracle loader | todo | | needs P02 (none strictly, but keep order) |
| P08 | InstanceNorm + SymLog | todo | | needs P07 |
| P09 | CausalMultiScaleDecomp | todo | | needs P08 |
| P10 | PatchEmbedding | todo | | needs P09 |
| P11 | RMSNorm + RelativeGQAttention | todo | | needs P10 |
| P12 | SwiGLU + TransformerBlock | todo | | needs P11 |
| P13 | Readout layers | todo | | needs P12 |
| P14 | NIG head stack | todo | | needs P13 |
| P15 | Full model assembly + e2e parity | todo | | needs P08-P14; M2 exit gate |
| P16 | Review 2 | todo | | M2 close |
| P17 | tasks.py + heads.py registries | todo | | needs P02, P15 |
| P18 | Kendall + compute_losses | todo | | needs P17; M3 exit gate (with P19) |
| P19 | One optimizer step | todo | | needs P18 |
| P20 | Review 3 | todo | | M3 close |
| P21 | stats/metrics.py | todo | | |
| P22 | selection_metric + Trainer loop | todo | | needs P18, P21 |
| P23 | seeds.py + identity.py | todo | | |
| P24 | Review 4 | todo | | M4 close |
| P25 | roles.py + split_plan.py | todo | | needs P01 |
| P26 | target_codec.py | todo | | needs P02 |
| P27 | prediction_schema.py + prediction_table.py | todo | | needs P15, P25 |
| P28 | Evaluation parity on fixture | todo | | needs P06,P15,P26,P27 |
| P29 | Calibrator | todo | | needs P25 |
| P30 | Review 5 | todo | | M5 close |
| P31 | model_contract.py | todo | | needs P17 |
| P32 | ARCHITECTURES registry | todo | | needs P15, P31 |
| P33 | tiny_mlp plugin | todo | | needs P32 |
| P33b | GRU/anti-memorization parity (closes V1) | todo | | needs P01 (second oracle), P08-P15 (reused layers) |
| P34 | input_skip seam | todo | | needs P15 |
| P35 | Per-layer probe report | todo | | needs P15 |
| P36 | Review 6 | todo | | M6 close |
| P37 | RollingSettings + rolling.py seam | todo | | needs P23, P25, P27 |
| P38 | staleness.py seam | todo | | needs P27 |
| P39 | Final review | todo | | plan close |

---

## 11. Risks

- **The second oracle (P01) may not be reproducible byte-for-byte** the way the first one is, for the same
  cross-machine float reasons as MINOR-1 — mitigated by the same tolerance table, not by chasing bit-exactness.
- **GRU weight layout (PyTorch `nn.GRU` vs Keras `GRU`)** differs in gate ordering (PyTorch: reset, update, new;
  Keras: reset, update, new is the same order by default but the *weight matrix concatenation* order differs) —
  P33b must verify this explicitly with a tiny single-layer GRU unit test before trusting the full-model parity
  test; if it does not match, the fix is a documented permutation helper (same pattern as P10/P11's
  `keras_*_weight_to_torch` helpers), never a silent "close enough" tolerance widening.
- **`npy_dir` fixture format drift**: if `crypto-signal-prediction`'s `data/storage.py` format changes (e.g. the
  TF-side hardening plan's own `S10`/`S12` work touches it), this repo's `data_io/npy_dir.py` and
  `docs/DATA_ARTIFACT_CONTRACT.md` go stale silently (there is no cross-repo CI). Mitigation: `docs/
  DATA_ARTIFACT_CONTRACT.md` names the exact source commit; re-verify it by hand if a later task ever bumps the
  pinned commit — this plan does not set up automated cross-repo drift detection (explicitly out of scope, below).
- **`core.registry.Registry`'s dict-API ergonomics already caused one friction incident on the TF side**
  (`crypto-signal-prediction/docs/research/hardening/friction_log.csv` row 3: `get()` needing a default). P02
  ports the *already-fixed* version, so this specific issue should not repeat, but any *new* consumer-ergonomics
  mismatch is still possible and should be a `docs/friction_log.csv` row here, not a silent registry edit.

## 12. Deliberately out of scope (do not build these under this plan)

- Rewriting any part of `data/` (the pipeline) in Python/torch — hard constraint; this repo only ever reads
  artifacts `data/` already produced.
- Any experiment run (`E-arch-001`, `E-roll-001`, `E-slot-001` equivalents), any claim of the form "architecture A
  beats architecture B", any edit to `crypto-signal-prediction`'s `docs/research/failure_registry.csv` or any
  recorded result there.
- Slot panel (cross-coin attention over fixed anonymous slots) — only "keep room" was asked; no seam is built for
  it in this plan because no concrete interface point has been identified yet that would not be speculative. If
  it is wanted next, it is its own follow-up plan, informed by this one's `ARCHITECTURES`/`ENCODERS` registries.
- NIG uncertainty fixes (calibration, floor tuning) beyond the `Calibrator` component itself (P29) — the T3
  arms/experiments on the TF side (`S11r`) are explicitly experiments, not this plan's job.
- Cascaded stages (teacher/distillation) — `S23`'s "optional teacher mode" is noted in section 1 but not built;
  `ModelContract` (P31) is the seam a future cascaded-stage plan would use.
- Matching TF's optimizer update arithmetic bit-for-bit (P19's explicit carve-out).
- Any cross-repo CI/automation that would re-verify `docs/DATA_ARTIFACT_CONTRACT.md` automatically against
  `crypto-signal-prediction`'s current `master` — this plan pins one commit and documents it; keeping that pin
  fresh is a future, separate decision.
- A `MAP.md`/zone-map navigation system for this repo (unlike `crypto-signal-prediction`) — this repo is small and
  flat enough from day one (plain imports, no shared namespace) that the token-cost problem `MAP.md` solves there
  does not yet exist here; revisit only if the repo grows past the point `src/nig_timenet_torch/` fits in one
  skim.

## 13. Open owner decisions (listed, not decided here)

1. Should `reference/README.md`'s provenance claim (ORACLE_REVIEW.md MAJOR-1) actually be edited, or is the
   existing oracle intentionally scoped to the Transformer/MODEL_CONFIG path forever, with the GRU/anti-
   memorization path staying a separate, secondary oracle (`reference/data_default/`, P01) from now on? This
   plan assumes the latter (two oracles, clearly labeled) but the owner may prefer to fix the README instead and
   retire the ambiguity a different way.
2. Should the `ARCHITECTURES` registry (P32-33) eventually gain a `gru_anti_memorization` named entry distinct
   from `nig_timenet_v2` with config overrides, or stay one architecture with a config preset (as P33b assumes,
   mirroring the TF side's own choice to keep it one builder with `ANTI_MEMORIZATION_CONFIG` as an overlay)?
3. Slot panel and cascaded stages (section 12): worth a seam now, or only once a concrete first consumer exists?
4. Should this repo eventually get its own `MODEL_DEV_PROTOCOL.md`-equivalent (experiment cards, model-health
   report) once real training experiments start here, or should every such experiment stay recorded on the
   `crypto-signal-prediction` side even once training happens in torch? (Affects where P35's probe report and any
   future health-report tooling ultimately get read from.)
5. `requirements`/Python version pin for this repo: this plan assumes CPU-only PyTorch, Python 3.11 (matching
   `crypto-signal-prediction`'s CI) — not yet confirmed against any GPU/Colab runtime requirement for this repo
   specifically (the hard constraint says GPU runs "go through Colab" but does not specify whether that means this
   repo needs Colab-specific packaging; P00 assumes not).

---

## Change log of this plan itself

- 2026-11-02: initial version, written by the planning session referenced in this repo's PR history.
- 2026-11-02 (P05, B1 review of `docs/friction_log.csv`): no reordering needed. Notes carried forward: P01's
  `shapes()` snippet must recurse (test is `{asset: split}`); P06/P25 must treat `price_norm_mode`,
  `price_pct_clip`, `holdout_start`, `higher_tf_mode` as optional dataset keys (absent from the fixture; see
  `docs/DATA_ARTIFACT_CONTRACT.md` notes); P25 must decide whether the dataset's `split_dates` or the pipeline
  CONFIG's dates are authoritative (the P01 split used CONFIG's); torch is pinned `2.15.0+cpu` and must be installed
  from the CPU index before `pip install -e ".[dev]"`.
