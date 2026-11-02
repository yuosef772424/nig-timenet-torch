# ORACLE_REVIEW — independent audit of `reference/` (NIG-TimeNet v2 TF oracle)

Scope: `reference/export_reference.py` + `reference/README.md` + `reference/data/*`, upstream
`yuosef772424/crypto-signal-prediction` pinned at `aeafc564f25c78f252936affbd374f68aa49b8d9` (verified: this
is the exact commit of the `./upstream` checkout used below, and matches `reference/data/provenance.json`).

Method: cloned upstream at the pinned commit into a fresh git worktree at `./upstream`, built a clean venv
(`uv venv -p python3.12`, `numpy==2.4.6`, `tensorflow-cpu~=2.21.0` wheel), re-ran `reference/export_reference.py`,
diffed every output file against the committed `reference/data/`, and independently re-derived outputs/losses
from scratch (new scripts, not reusing `export_reference.py`'s code) using only the real upstream model/trainer
objects. Versions actually used: TensorFlow 2.21.0, NumPy 2.4.6, Keras 3.15.1 (matches the README's provenance
table exactly); Python was 3.12.3, not the documented 3.12.15 (3.12.15 is not installable via `uv python
install` in this sandbox — closest available is 3.12.13).

## Findings

### MAJOR-1 — "exactly as main.ipynb does" is false for the current default `RunSettings`

**Where:** `reference/README.md:25-27` (provenance claim) vs. `upstream/workflow/settings.py:160` (
`ModelSettings.anti_memorization: bool = True`), `upstream/workflow/settings.py:135-140` (
`TargetSettings.suspended_targets` docstring: `None` = "close suspended unless target_mode starts with
entry_range"), `upstream/workflow/run.py:247-249` (`suspended = () if entry_range else ("close",)`),
`upstream/workflow/run.py:267-268` (`if settings.model.anti_memorization: overrides.update(kit.anti_memorization_config)`),
`upstream/workflow/run.py:316` (`label_smoothing=tr.anti_memorization_label_smoothing if _am else 0.0`), and
`upstream/main.ipynb` cell 10 (`settings = RunSettings(..., model=ModelSettings(anti_memorization=True,
class_only=False), target=TargetSettings(target_mode=None, suspended_targets=None, ...), ...)`).

**What's wrong:** `reference/README.md` states the oracle's model is "Built through the upstream entry point,
exactly as `main.ipynb` does: `model.build_model_fn(32, 6, config={"d_model": 16, "num_layers": 2})`". This is
not what `main.ipynb` actually builds with its own (unedited) default `RunSettings`:

- `ModelSettings.anti_memorization` defaults to `True`, and `main.ipynb`'s own settings cell passes
  `anti_memorization=True` explicitly. `workflow.run.plan_model` then merges the **entire**
  `ANTI_MEMORIZATION_CONFIG` (`upstream/model/config.py:68-84`: `encoder="gru"` — not `"transformer"` — plus
  `d_model=64`, `stats_mode="symlog"`, `level_passthrough=True`, `level_norm="batch"`, `input_clip=4.0`,
  `input_noise_std=0.1`, `feature_dropout=0.1`, `dropout=0.25`, `linear_path_l2=1e-3`, `head_hidden=64`,
  `class_head_hidden=32`) over `MODEL_CONFIG`, not merely `{d_model: 16, num_layers: 2}`. The real default model
  is a **GRU encoder with level passthrough**, not the Transformer/GQA encoder the oracle exports.
- `TargetSettings.suspended_targets=None` (the notebook's own default) resolves to `suspended=("close",)`
  (`workflow/run.py:247-249`), so the real default `price_targets` is `(high, low)` — **not** `(high, low,
  close)` as the oracle has. `close` is trained/evaluated by the oracle but is not part of the real default run.
- Classification `label_smoothing` is `tr.anti_memorization_label_smoothing = 0.1` in the real default run
  (`workflow/settings.py:211`), not `0.0` as the oracle explicitly sets (`export_reference.py:273`).

This is not a hidden bug on the builder's part — `NOTES.md` item 9 discloses that "anti-memorization config
... [is] not covered ... as requested" — but `reference/README.md`'s own wording still asserts the stronger,
contradicted claim ("exactly as main.ipynb does"), and never mentions that `close` is suspended by default
(`NOTES.md` #9 doesn't mention target suspension at all). Anyone taking the README at face value would validate
a PyTorch port against the wrong architecture (Transformer instead of the real default GRU) and the wrong
target set (3 targets instead of 2) while believing they match production.

**Reproduce:**
```bash
cd upstream && python3 - <<'PY'
import json
nb = json.load(open("main.ipynb"))
print("".join(nb["cells"][10]["source"]))   # model=ModelSettings(anti_memorization=True, class_only=False)
PY
grep -n "anti_memorization: bool" workflow/settings.py
grep -n "suspended = () if entry_range" workflow/run.py
grep -n "overrides.update(kit.anti_memorization_config)" workflow/run.py
```

**Fix recommendation:** reword the Provenance section to state plainly that the oracle exercises
`build_nig_timenet_v2` via `build_model_fn` with **`MODEL_CONFIG` defaults** (Transformer encoder, 3 targets,
`label_smoothing=0`), which is a legitimate, real, reachable configuration of the upstream code, but is
*not* what `main.ipynb`'s own default `RunSettings` currently trains (anti-memorization GRU model, 2 targets,
`label_smoothing=0.1`). Cross-reference `NOTES.md` #9 so the two documents agree.

### MAJOR-2 — `weights.npz`/`grads.npz` store the 3 Kendall `log_var_*` scalars with the wrong shape

**Where:** `reference/export_reference.py:45-53` (`save_npz`, specifically line 52:
`np.lib.format.write_array(buf, np.ascontiguousarray(arrays[key]), allow_pickle=False)`).

**What's wrong:** `np.ascontiguousarray` on a 0-d (scalar) array silently promotes it to shape `(1,)` in NumPy
2.4.6. The only scalar-shaped tensors in this export are the 3 trainer Kendall weights
(`trainer/uncertainty_weighted_loss/log_var_{high_class,low_class,close_class}`, declared `shape: []` in both
`weights.json` and `grads.json`, consistent with `tf.keras.layers.Layer.add_weight(shape=())` — verified live
that Keras 3.15.1/TF 2.21.0 really does keep these as 0-d, `v.numpy().shape == ()`). But the corresponding
arrays actually stored in `weights.npz` and `grads.npz` have shape `(1,)`, not `()`. The numeric *values* are
correct (e.g. `grads.npz["trainer/uncertainty_weighted_loss/log_var_high_class"] == [0.17111754]`, matching
what a from-scratch recompute gives); only the rank/shape metadata is wrong, and it disagrees with the
`weights.json`/`grads.json` files shipped right next to it. A PyTorch port's test harness that asserts
`loaded.shape == tuple(meta["shape"])` (the natural thing to do, since that is exactly what the JSON promises)
will fail on these 3 of ~150 entries.

**Reproduce:**
```python
import numpy as np
a = np.array(0.0, dtype=np.float32)
print(a.shape, np.ascontiguousarray(a).shape)   # () -> (1,)   <- the bug

with np.load("reference/data/weights.npz") as wz:
    print(wz["trainer/uncertainty_weighted_loss/log_var_high_class"].shape)   # (1,), weights.json says []
with np.load("reference/data/grads.npz") as gz:
    print(gz["trainer/uncertainty_weighted_loss/log_var_high_class"].shape)   # (1,), grads.json says []
```

**Fix recommendation:** in `save_npz`, special-case 0-d arrays (e.g. `arr = np.asarray(arrays[key]); buf_arr =
arr if arr.ndim else arr.reshape(())` after, not through, `np.ascontiguousarray`, or just skip
`np.ascontiguousarray` for 0-d input since it is already contiguous) and re-export. This changes 6 of the
~2600+ scalars stored across `weights.npz`/`grads.npz`; everything else is unaffected (double-checked: no other
weight/grad array in this export is 0-d).

### MINOR-1 — cross-machine re-run is not byte-identical (confirms `NOTES.md` #10)

**Where:** `reference/README.md:117-132` (sha256 table; the claim is explicitly about "two consecutive runs"
on the same machine, which this audit cannot contradict) vs. an independent re-run on different hardware in
this sandbox (same NumPy/TensorFlow/Keras versions, Python 3.12.3 instead of the documented 3.12.15).

**What was found:** 11 of 13 data files (`config.json`, `inputs.npz`, `labels.npz`, `layer_outputs.npz`,
`layers.json`, `outputs.npz`, `weights.json`, `weights.npz`, `sublayers.json`, `sublayers.npz`, `grads.npz`)
reproduced byte-for-byte (sha256 matches `reference/README.md` exactly). Two did not:
- `provenance.json` differs only in the recorded Python patch version (expected; this sandbox cannot install
  the exact 3.12.15).
- `losses.json` differs in exactly **one** of ~150 recorded float32 values:
  `scenarios."schedules_end (PRIMARY...)".task_losses_weighted.low_reg` = `1.4175130128860474` (hex
  `3fb57111`) here vs. `1.417513132095337` (hex `3fb57112`) in the committed file — a 1-ULP difference. An
  independent from-scratch recompute of the same scenario (new script, real `GenericTrainer._compute_losses`,
  committed `weights.npz`/`inputs.npz`/`labels.npz`) reproduces this exact 1-ULP-different value and otherwise
  matches `losses.json` exactly, including `total_loss` and `raw_loss` bit-for-bit (the ULP perturbation in
  `low_reg` is absorbed by float32 rounding in the final `tf.add_n`). `grads.npz` (which is computed from the
  same tape in the same run) is still byte-identical to the README despite this.

This is exactly the risk `NOTES.md` #10 already flags ("another CPU may give last-bit differences... compare
with tolerances, not bitwise") — this audit turns that prediction into a confirmed, reproduced fact, isolated
to one specific value. Not a logic bug; informational.

**Reproduce:** rebuild the venv as in `reference/README.md`, run `python reference/export_reference.py`,
`sha256sum` the outputs against the table, and diff `losses.json` key-by-key against the committed file.

## Verified correct (no finding)

- **Check 1 (real upstream code only):** read every layer module (`model/{transformer,nig_layers,heads,
  readout,decomposition,patches,input_norm,builder,config}.py`) and the trainer/workflow loss path
  (`trainer/{trainer,tasks,task_weighting,config}.py`, `workflow/{run,settings,training_config}.py`). The script
  only calls upstream functions/layers (`model.load_into`, `build_model_fn`/`build_nig_timenet_v2`,
  `trainer.build_config`, `GenericTrainer._compute_losses`) or wraps `tf.nn.softmax`/`tf.nn.softplus` to *record*
  their I/O (`export_reference.py:147-163`), never re-implements them. The one piece of arithmetic written in
  the script itself (the two residual `+` additions of `TransformerBlock.call`, disclosed in
  `reference/README.md:152-153`) is verified below.
- **Check 2 (config.json rebuilds the same model):** independently rebuilt `build_nig_timenet_v2` from
  `config.json`'s `constructor_kwargs` alone (fresh script, not `export_reference.py`) — the 37-layer list
  (names, order, classes) and all 72 model weights (names/shapes/dtype/trainable) matched `layers.json` /
  `weights.json` exactly, confirming the README's layer-order table (rows 0-36) is correct, including the
  non-obvious Keras graph ordering (`class_fc_{close,high,low}` before `nig_{close,high,low}`, which does not
  match the Python call order in `builder.py` but is what Keras's own `.layers` property returns).
- **Check 3 (layer outputs/sublayers from the right inputs):** independently recomputed 11 values across 6
  layers (`instance_norm` z+stats, `decomp` trend+seasonal, the `trend_seasonal` concat, `patch_embed`,
  `block_1`'s full output via a single direct layer call, and all 4 `nig_close` outputs) from the committed
  weights/inputs — all byte-identical to `layer_outputs.npz`, exceeding the requested 5 spot-checks. The
  `export_sublayers` internal self-checks (`export_reference.py:218-220`, `246-250`) also ran and passed during
  the re-run.
- **Check 4 (losses/grads from upstream loss code, stated weights):** confirmed `loss_weights` are all `1.0`
  (`config['targets'][t].get('loss_weight', 1.0)`, never set otherwise) and `scheduled_vars` are `lambda_reg=
  0.05, lambda_calib=0.1` in the PRIMARY scenario, matching `TrainSettings()` end values. Recomputed
  `total_loss`/`raw_loss` from scratch (see MINOR-1) — matched `losses.json` exactly (`5.208889484405518` /
  `6.332683563232422`), confirming `_compute_losses` (not a reimplementation) drives the numbers. Also
  confirmed Keras 3's `l2` regularizer is `l2 * sum(w²)` with no `0.5` factor and `Dense(activation="gelu")`
  defaults to the exact erf form (`approximate=False`) — both match `NOTES.md` #6/#7.
- **Check 5 (sha256 re-run):** see MINOR-1 — 11/13 data files byte-identical cross-machine; the 2 exceptions are
  explained and are not logic bugs.

## Answers to `NOTES.md` OPEN items

1. **Head types.** Confirmed: `workflow.run.plan_model` (`run.py:278-280`) always sets
   `head_types={t: [...]}` explicitly for every active target regardless of `anti_memorization`/`class_only`
   (`class_only=False` by default), using `["nig_regression", "binary_classification"]` — i.e. the oracle's
   `MODEL_CONFIG['head_types']` default is correct and is what `main.ipynb` sets for `head_types` specifically
   (independent of the architecture/target-set issue in MAJOR-1). `builder.DEFAULT_HEAD_TYPES` (NIG-only) is
   only reached by calling `build_nig_timenet_v2` directly with no `head_types`, which `build_model_fn` never
   does. A NIG-only variant does need a second export; no inconsistency found.
2. **Which scenario to validate against.** Use the PRIMARY (`schedules_end`) scenario — it is the only one that
   exercises every loss term (evidential regularizer, confidence-calibration loss, and therefore the
   `ConfidenceHead` gradient path); the `start` scenario has `lambda_reg=lambda_calib=0` and would hide bugs in
   those code paths entirely. Verified by recomputation (see Check 4) that PRIMARY's numbers are correct.
3. No concern: `training=False` is the only sensible scope for a byte-reproducible oracle; confirmed Dropout
   layers are identity at `training=False` by reading `tf.keras.layers.Dropout` usage sites, and flood_level
   only applies `if training` (`trainer/trainer.py:136`), so it's correctly inert here.
5. **Kendall `log_var_*` ownership.** They are structurally training-loop state, not model state: they live on
   `UncertaintyWeightedLoss`, a layer owned by `GenericTrainer` (`trainer/trainer.py:64-66`), not by `net`
   (`export_reference.py` itself separates `net.weights` from `gt.uncertainty_layer.weights`, prefixing the
   latter `trainer/`). Recommend the port keep them outside the model checkpoint, mirroring upstream's own
   separation — but see MAJOR-2 first (their exported shape is currently wrong).
6. Confirmed correct: live-checked in Keras 3.15.1 that `regularizers.l2(c)(w) == c * sum(w**2)` exactly
   (no 0.5 factor).
7. Confirmed correct: live-checked `tf.keras.activations.gelu` source — default `approximate=False` is the
   exact `0.5*x*(1+erf(x/sqrt(2)))` form.
8. Not independently re-checked (would require installing `tensorflow-cpu` itself and re-diffing, which this
   audit's venv did not do — it also installed the full `tensorflow` wheel, like the original export). Still
   open; low risk since both this audit and the original export used the same non-`-cpu` wheel, so this is at
   least self-consistent.
9. **Scope exclusions.** Correct as a description of what's technically absent, but see MAJOR-1: the stated
   reason ("as requested") describes a scope decision, while `reference/README.md`'s provenance wording still
   claims equivalence to `main.ipynb`'s defaults, which is false once `anti_memorization`'s architecture change
   and the default `close` target suspension are taken into account. Recommend `NOTES.md` #9 explicitly name
   both (currently only the architecture variants are named, not target suspension).
10. **Confirmed, not just predicted** — see MINOR-1: a real cross-machine re-run (same NumPy/TF/Keras versions)
    produced a 1-ULP difference in one `losses.json` value. The recommendation to compare with tolerances, not
    bitwise, is validated.

## Verdict: **PASS, with required follow-ups**

The oracle's core promise holds: every number in `reference/data/` is genuinely produced by calling real,
unmodified upstream code (model, trainer, loss) — nothing is re-implemented — and this audit independently
reproduced the model (rebuild-and-compare), 11 layer/sub-layer outputs, and the total/raw loss from scratch,
all byte- or value-identical to the committed files (mod one disclosed cross-machine ULP). It is safe to use
as a numeric ground truth for porting `build_nig_timenet_v2` under the **exact config `config.json` records**
(Transformer encoder, `d_model=16`, `num_layers=2`, 3 targets, `label_smoothing=0`).

Two things must be fixed/acknowledged before relying on it further:
- **MAJOR-2** (shape bug) should be fixed and the 3 affected files re-exported — trivial, isolated, no
  numeric content changes.
- **MAJOR-1** (provenance wording) should be corrected in `reference/README.md` so nobody assumes this
  matches `main.ipynb`'s actual default training run (it doesn't: different encoder, different target set,
  different label smoothing). This does not require a new export, only fixing the claim — unless the intent
  really is to also cover the anti-memorization/GRU default, in which case a second export is needed (same
  conclusion as `NOTES.md` #1 draws for the head-types question, extended to the whole architecture).
