"""Export a deterministic TensorFlow reference ("oracle") of NIG-TimeNet v2 under main.ipynb's REAL default settings
(BUILD_PLAN.md P01 / V1): anti_memorization=True (ANTI_MEMORIZATION_CONFIG over MODEL_CONFIG, as workflow/run.py:267-268),
GRU trunk, label_smoothing=0.1 (workflow/run.py:316) -- EXCEPT price_targets, kept at (high, low, close) by owner
decision (2026-11-02), i.e. main.ipynb's close-suspension (workflow/run.py:247-249) is NOT applied.
Verbatim copy of export_reference.py except: MODEL_OVERRIDES, label_smoothing, OUT (= reference/data_default).

Everything is computed by the UPSTREAM code (./upstream, read only), loaded through its shared-namespace loaders:
  * model    : model.load_into(ns)                      -> build_model_fn / MODEL_CONFIG
  * targets  : workflow.load_into(ns, only=(...))       -> build_target_configs, TrainSettings (loss schedules)
  * losses   : trainer.load_into(ns)                    -> build_config, GenericTrainer._compute_losses
No upstream code is copied here. Run:  python reference/export_reference.py
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU only
os.environ["PYTHONHASHSEED"] = "0"

import sys

sys.dont_write_bytecode = True                   # never write __pycache__ into ./upstream (read only)

import contextlib
import io
import json
import random
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UPSTREAM = ROOT / "upstream"
OUT = HERE / "data_default"
SEED = 0

# ─────────────────────────────── the small model (constructor overrides over MODEL_CONFIG) ───────────────────────────────
SEQ_LEN = 32
N_FEATURES = 6
BATCH = 8
MODEL_OVERRIDES = None  # = dict(mns["ANTI_MEMORIZATION_CONFIG"]), set in main() once the upstream model namespace is loaded


# ─────────────────────────────── deterministic writers ───────────────────────────────
def save_npz(path, arrays):
    """np.savez equivalent with fixed zip timestamps/order (np.savez stamps the current time -> not byte-identical)."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for key in arrays:  # insertion order is the documented order
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[key]), allow_pickle=False)
            zf.writestr(info, buf.getvalue())


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=False)
        f.write("\n")


def jsonable(v):
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def f32(x):
    """A float32 scalar as JSON: exact float value + its bit pattern (hex) so nothing is lost to printing."""
    a = np.asarray(x, dtype=np.float32).reshape(())
    return {"value": float(a), "f32_hex": a.view(np.uint32).item().to_bytes(4, "big").hex()}


# ─────────────────────────────── setup ───────────────────────────────
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)  # python + numpy + backend (Keras 3) in one call


def load_upstream():
    sys.path.insert(0, str(UPSTREAM))
    import model as model_pkg        # noqa: E402  (shared-namespace packages of upstream)
    import trainer as trainer_pkg    # noqa: E402
    import workflow as workflow_pkg  # noqa: E402

    quiet = contextlib.redirect_stdout(io.StringIO())
    with quiet:
        mns = model_pkg.load_into({"__name__": "nig_ref_model"}, exclude=("diagnostics", "layer_report", "selftests"))
        tns = trainer_pkg.load_into({"__name__": "nig_ref_trainer"}, exclude=("smoke_test", "kfold"))
        wns = workflow_pkg.load_into({"__name__": "nig_ref_workflow"}, only=("settings", "training_config"))
    return mns, tns, wns


def upstream_sha():
    return subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()


# ─────────────────────────────── graph helpers ───────────────────────────────
def as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def inbound_names(layer):
    """Names of the layers feeding `layer` in the functional graph (with output index), in call order."""
    if not layer._inbound_nodes:
        return []
    node = layer._inbound_nodes[0]
    names = []
    for t in node.input_tensors:
        op, _node_idx, tensor_idx = t._keras_history
        names.append(f"{op.name}" + (f"[{tensor_idx}]" if len(as_list(op.output)) > 1 else ""))
    return names


def attr_paths(net):
    """id(variable) -> attribute path inside its top-level layer, e.g. 'block_1.attn.wq.kernel' (stable, unlike the
    auto-numbered Keras sublayer names such as dense_14)."""
    out = {}

    def walk(obj, prefix, seen):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        for v in obj._trainable_variables + obj._non_trainable_variables if hasattr(obj, "_trainable_variables") else []:
            out.setdefault(id(v), f"{prefix}.{v.name}")
        for attr, val in sorted(vars(obj).items()):
            if attr.startswith("_"):
                continue
            if isinstance(val, tf.keras.layers.Layer):
                walk(val, f"{prefix}.{attr}", seen)
            elif isinstance(val, (list, tuple)):
                for j, item in enumerate(val):
                    if isinstance(item, tf.keras.layers.Layer):
                        walk(item, f"{prefix}.{attr}[{j}]", seen)

    for layer in net.layers:
        walk(layer, layer.name, set())
    return out


@contextlib.contextmanager
def capture(module, fn_name):
    """Temporarily wrap module.<fn_name> (e.g. tf.nn.softmax) to record every (input, output) of the upstream call.
    Captures the exact tensors the upstream code computes; nothing of upstream is re-implemented."""
    orig = getattr(module, fn_name)
    calls = []

    def wrapped(x, *a, **kw):
        y = orig(x, *a, **kw)
        calls.append((np.asarray(x), np.asarray(y)))
        return y

    setattr(module, fn_name, wrapped)
    try:
        yield calls
    finally:
        setattr(module, fn_name, orig)


def export_sublayers(net, lo_arrays, lmeta):
    """Internals of every TransformerBlock and NIGHead, by calling the upstream sub-layer objects of the built model
    (training=False) on the exact tensor each receives inside the parent layer's call."""
    by_name = {m["name"]: m for m in lmeta}
    arrays, meta = {}, []

    def put(parent, step, val, how):
        key = f"{parent}/{step}"
        arrays[key] = np.asarray(val)
        meta.append({"block": parent, "step": step, "key": key, "shape": list(arrays[key].shape),
                     "dtype": str(arrays[key].dtype), "how": how})

    def exact(a, b, what):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or a.tobytes() != b.tobytes():
            raise AssertionError(f"{what}: recomputed value differs from layer_outputs (max abs diff "
                                 f"{np.max(np.abs(a - b)) if a.shape == b.shape else 'shape'})")

    for layer in net.layers:
        if type(layer).__name__ != "TransformerBlock":
            continue
        name = layer.name
        src = by_name[name]["inputs"][0]
        src_key = by_name[src]["output_keys"][0]
        x = tf.constant(lo_arrays[src_key])
        attn = layer.attn
        put(name, "00_block_input", x, f"layer_outputs[{src_key!r}] (output of {src})")
        n1 = layer.norm1(x)
        put(name, "01_norm1_out", n1, "block.norm1(x)")
        put(name, "02_attn_q_proj", attn.wq(n1), "block.attn.wq(norm1_out)  (before reshape to heads)")
        put(name, "03_attn_k_proj", attn.wk(n1), "block.attn.wk(norm1_out)  (before reshape / GQA repeat)")
        put(name, "04_attn_v_proj", attn.wv(n1), "block.attn.wv(norm1_out)  (before reshape / GQA repeat)")
        with capture(tf.nn, "softmax") as sm:
            a_out = attn(n1, training=False)
        if len(sm) != 1:
            raise AssertionError(f"{name}: expected 1 softmax call inside attention, got {len(sm)}")
        mask = "causal mask" if attn.causal else "no causal mask (causal=False)"
        mask += "" if attn.window is None else f", window={attn.window}"
        put(name, "05_attn_scores_pre_softmax", sm[0][0],
            "input of tf.nn.softmax captured during block.attn(norm1_out, training=False): q·kᵀ/√dh + relative bias, "
            f"{mask}; shape (B, heads, T, T)")
        put(name, "06_attn_weights", sm[0][1],
            "output of the same captured tf.nn.softmax call (attention dropout is identity at training=False)")
        put(name, "07_attn_out", a_out, "block.attn(norm1_out, training=False)  (after wo)")
        r1 = x + layer.drop1(a_out, training=False)
        put(name, "08_residual1", r1, "x + block.drop1(attn_out, training=False)  (the addition of TransformerBlock.call)")
        n2 = layer.norm2(r1)
        put(name, "09_norm2_out", n2, "block.norm2(residual1)")
        f = layer.ffn(n2)
        put(name, "10_swiglu_out", f, "block.ffn(norm2_out)  (SwiGLU, after w2)")
        out = r1 + layer.drop2(f, training=False)
        put(name, "11_block_output", out, "residual1 + block.drop2(swiglu_out, training=False)")
        own_key = by_name[name]["output_keys"][0]
        exact(out, lo_arrays[own_key], f"{name} recomputed block output")
        exact(layer(x, training=False), lo_arrays[own_key], f"{name}(x) direct call")

    for layer in net.layers:
        if type(layer).__name__ != "NIGHead":
            continue
        name = layer.name
        src = by_name[name]["inputs"][0]
        src_key = by_name[src]["output_keys"][0]
        h = tf.constant(lo_arrays[src_key])
        put(name, "00_head_input", h, f"layer_outputs[{src_key!r}] (output of {src})")
        hid = layer.fc(h)
        put(name, "01_fc_out", hid, "head.fc(h)  (Dense hidden, swish)")
        p = layer.out(layer.drop(hid, training=False))
        put(name, "02_raw_params", p, "head.out(head.drop(fc_out, training=False)); columns = [mu, nu, alpha, beta] "
                                      "before softplus/floor")
        for j, part in enumerate(("mu", "nu", "alpha", "beta")):
            put(name, f"03_raw_{part}", p[:, j:j + 1], f"raw_params[:, {j}:{j + 1}]")
        with capture(tf.nn, "softplus") as spl:
            res = layer(h, training=False)
        if len(spl) != 3:
            raise AssertionError(f"{name}: expected 3 softplus calls in NIGHead, got {len(spl)}")
        cfg = layer.cfg
        for (sp_in, sp_out), part in zip(spl, ("nu", "alpha", "beta")):
            put(name, f"04_softplus_{part}", sp_out,
                f"output of tf.nn.softplus captured during head(h, training=False), before + {part}_min")
        for j, part in enumerate(("mu", "nu", "alpha", "beta")):
            floor = "no floor (mu is linear)" if part == "mu" else f"softplus + {part}_min={cfg[part + '_min']}"
            put(name, f"05_final_{part}", res[j], f"head(h, training=False)[{j}]: {floor}")
            exact(res[j], lo_arrays[by_name[name]["output_keys"][j]], f"{name} output {part}")
        for (sp_in, _), part, j in zip(spl, ("nu", "alpha", "beta"), (1, 2, 3)):
            exact(sp_in, p[:, j:j + 1], f"{name} softplus input {part} vs raw params")

    save_npz(OUT / "sublayers.npz", arrays)
    save_json(OUT / "sublayers.json", meta)


def main():
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)

    mns, tns, wns = load_upstream()
    set_seeds(SEED)

    # ── 1) model ──
    MODEL_OVERRIDES = dict(mns["ANTI_MEMORIZATION_CONFIG"])  # noqa: N806  (workflow/run.py:267-268)
    merged = dict(mns["MODEL_CONFIG"])
    merged.update(MODEL_OVERRIDES)
    net = mns["build_model_fn"](SEQ_LEN, N_FEATURES, config=dict(MODEL_OVERRIDES))
    price_targets = list(merged["price_targets"])

    # ── trainer config: the same targets/schedules main.ipynb uses (workflow.run.make_training_config) ──
    tr_defaults = wns["TrainSettings"]()
    targets = wns["build_target_configs"](price_targets, label_smoothing=0.1)
    user_cfg = {
        "targets": targets,
        "loss": {
            "use_uncertainty_weighting": tr_defaults.use_uncertainty_weighting,
            "schedules": {"lambda_reg": dict(tr_defaults.lambda_reg), "lambda_calib": dict(tr_defaults.lambda_calib)},
        },
    }
    with contextlib.redirect_stdout(io.StringIO()):
        tcfg = tns["build_config"](user_cfg)
        gt = tns["GenericTrainer"](net, tcfg)

    # ── 2) inputs + labels ──
    rng = np.random.default_rng(SEED)
    level = rng.normal(0.0, 2.0, size=(BATCH, 1, N_FEATURES))
    scale = rng.uniform(0.5, 2.0, size=(BATCH, 1, N_FEATURES))
    walk = np.cumsum(rng.standard_normal((BATCH, SEQ_LEN, N_FEATURES)), axis=1) * 0.3
    x = (level + scale * walk).astype(np.float32)
    labels = {}
    for t in price_targets:
        labels[f"y_{t}_reg"] = (rng.standard_normal((BATCH, 1)) * 0.5).astype(np.float32)
        labels[f"y_{t}_class"] = rng.integers(0, 2, size=(BATCH, 1)).astype(np.float32)  # {0,1} (feed-time form)
    save_npz(OUT / "inputs.npz", {"input_sequence": x})
    save_npz(OUT / "labels.npz", labels)

    # ── config.json ──
    save_json(OUT / "config.json", jsonable({
        "seed": SEED,
        "builder_call": "model.build_model_fn(seq_len, n_features, config=overrides)  "
                        "== build_nig_timenet_v2(seq_len, n_features, **{**MODEL_CONFIG, **overrides})",
        "seq_len": SEQ_LEN, "n_features": N_FEATURES, "batch": BATCH,
        "overrides": MODEL_OVERRIDES,
        "constructor_kwargs": {"seq_len": SEQ_LEN, "n_features": N_FEATURES, **merged},
        "model_name": net.name,
        "input_names": [t.name for t in as_list(net.inputs)],
        "trainer": {
            "targets": targets,
            "loss": {k: tcfg["loss"][k] for k in ("use_uncertainty_weighting",
                                                   "uncertainty_weighting_exclude_task_types", "schedules")},
            "constraints": [c.get("name") for c in tcfg["loss"]["constraints"]],
        },
    }))

    # ── 3) weights ──
    w_arrays, w_meta = {}, []
    owner = {}
    for layer in net.layers:
        for v in layer.weights:
            owner.setdefault(id(v), layer)
    apaths = attr_paths(net)
    for v in net.weights:
        lay = owner[id(v)]
        w_arrays[v.path] = v.numpy()
        w_meta.append({"name": v.path, "shape": list(v.shape), "dtype": str(v.dtype), "trainable": bool(v.trainable),
                       "attr_path": apaths.get(id(v)), "layer": lay.name, "layer_class": type(lay).__name__})
    for v in gt.uncertainty_layer.weights if gt.uncertainty_layer is not None else []:
        w_arrays["trainer/" + v.path] = v.numpy()
        w_meta.append({"name": "trainer/" + v.path, "shape": list(v.shape), "dtype": str(v.dtype),
                       "trainable": bool(v.trainable), "layer": gt.uncertainty_layer.name,
                       "layer_class": type(gt.uncertainty_layer).__name__,
                       "note": "Kendall log-variance of the trainer (not part of the model)"})
    assert len(w_arrays) == len(w_meta), "duplicate weight path"
    save_npz(OUT / "weights.npz", w_arrays)
    save_json(OUT / "weights.json", w_meta)

    # ── 4) every layer output (training=False), graph order ──
    probe_outputs, slots = [], []
    for i, layer in enumerate(net.layers):
        outs = as_list(layer.output)
        for k, o in enumerate(outs):
            probe_outputs.append(o)
            slots.append((i, layer, k, len(outs)))
    probe = tf.keras.Model(net.inputs, probe_outputs, name="probe")
    vals = probe([x], training=False)
    lo_arrays, lmeta = {}, []
    by_layer = {}
    for (i, layer, k, n), val in zip(slots, vals):
        key = f"{i:03d}_{layer.name}" + (f"__{k}" if n > 1 else "")
        lo_arrays[key] = np.asarray(val)
        by_layer.setdefault(i, []).append(key)
    for i, layer in enumerate(net.layers):
        lmeta.append({"index": i, "name": layer.name, "class": type(layer).__name__,
                      "inputs": inbound_names(layer),
                      "output_keys": by_layer[i],
                      "output_shapes": [list(lo_arrays[k].shape) for k in by_layer[i]]})
    save_npz(OUT / "layer_outputs.npz", lo_arrays)
    save_json(OUT / "layers.json", lmeta)

    # ── 5) final outputs ──
    outs = net(x, training=False)
    save_npz(OUT / "outputs.npz", {k: np.asarray(outs[k]) for k in sorted(outs)})

    # ── 6) + 7) losses and gradients with the upstream trainer ──
    y = {k: tf.constant(v) for k, v in labels.items()}
    xs = tf.constant(x)
    loss_report = {"training": False, "note": "GenericTrainer._compute_losses(x, y, sample_weight=None, training=False)",
                   "scenarios": {}}

    def run(sched_values, with_grads):
        for name, val in sched_values.items():
            gt.scheduled_vars[name].assign(val)
        train_vars = gt.trainable_variables
        with tf.GradientTape() as tape:
            total, raw, per_t, stats, cons = gt._compute_losses(xs, y, None, training=False)
        rep = {"scheduled_vars": {k: float(v.numpy()) for k, v in gt.scheduled_vars.items()},
               "total_loss": f32(total), "raw_loss": f32(raw),
               "task_losses_weighted": {t: f32(v) for t, v in per_t.items()},
               "loss_weights": {t: float(v.numpy()) for t, v in gt.loss_weights.items()},
               "model_regularization_losses": f32(tf.add_n(net.losses)) if net.losses else None,
               "stats": {k: f32(v) for k, v in stats.items()},
               "constraints": {k: f32(v) for k, v in cons.items()},
               "kendall_tasks": list(gt.kendall_task_names), "fixed_tasks": list(gt.fixed_task_names)}
        grads = tape.gradient(total, train_vars) if with_grads else None
        return rep, train_vars, grads

    start = {k: float(v["start"]) for k, v in tcfg["loss"]["schedules"].items()}
    end = {k: float(v["end"]) for k, v in tcfg["loss"]["schedules"].items()}
    rep_start, _, _ = run(start, False)
    rep_end, train_vars, grads = run(end, True)
    loss_report["scenarios"]["schedules_end (PRIMARY; grads.npz uses this)"] = rep_end
    loss_report["scenarios"]["schedules_start (epoch 0 values)"] = rep_start
    loss_report["primary"] = "schedules_end (PRIMARY; grads.npz uses this)"
    save_json(OUT / "losses.json", loss_report)

    trainer_ids = {id(v) for v in (gt.uncertainty_layer.weights if gt.uncertainty_layer is not None else [])}
    g_arrays, g_meta = {}, []
    for v, g in zip(train_vars, grads):
        name = ("trainer/" if id(v) in trainer_ids else "") + v.path
        g_meta.append({"name": name, "shape": list(v.shape), "grad_is_none": g is None})
        g_arrays[name] = np.zeros(v.shape, np.float32) if g is None else np.asarray(tf.convert_to_tensor(g))
    save_npz(OUT / "grads.npz", g_arrays)
    save_json(OUT / "grads.json", g_meta)

    # ── 8) sub-layer internals (TransformerBlock, NIGHead) ──
    export_sublayers(net, lo_arrays, lmeta)

    save_json(OUT / "provenance.json", {
        "upstream_sha": upstream_sha(), "python": sys.version.split()[0], "numpy": np.__version__,
        "tensorflow": tf.__version__, "keras": tf.keras.__version__ if hasattr(tf.keras, "__version__") else None,
    })
    print(f"wrote {len(list(OUT.iterdir()))} files to {OUT}")


if __name__ == "__main__":
    main()
