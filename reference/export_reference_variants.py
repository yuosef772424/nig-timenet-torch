"""Export small TF oracles for the model variants ported in batch B9 (BUILD_PLAN.md P40-P43):

  tcn         encoder="tcn" (causal dilated Conv1D residual blocks)
  multitf     multi-timeframe seq_len/n_features dicts (one encoder branch per timeframe, branch_concat)
  multiclass  multiclass_classification head (softmax) + upstream classification_task_loss (binary=False)
  panel       cross_asset PanelModel (shared encoder -> masked cross-coin attention -> per-coin heads),
              cross_attention True and False

Everything is computed by the UPSTREAM code (./upstream, read only), loaded through the shared-namespace loaders of
reference/export_reference.py (whose helpers are imported, not copied). Each case writes reference/data_variants/<case>/:
  config.json   kwargs used, upstream commit, versions
  weights.npz   every model variable keyed by attr path (NIG models: same attr paths as reference/data/weights.json;
                panel: 'encoder.<attr path>' for the encoder, Keras variable .path for the panel's own layers)
  inputs.npz    model inputs     outputs.npz   model outputs, training=False      [losses.json for multiclass]
Run:  .venv/bin/python reference/export_reference_variants.py
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import export_reference as er  # noqa: E402  (sets the deterministic TF env vars at import)
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

OUT = HERE / "data_variants"
SMALL = {"d_model": 16, "num_layers": 2}


def nig_weights(net):
    paths = er.attr_paths(net)
    return {paths[id(v)]: np.asarray(v) for v in net.weights}


MODEL_CONFIG = None  # upstream MODEL_CONFIG, set in main()


def full_kwargs(config):
    """What build_model_fn really passes to build_nig_timenet_v2: MODEL_CONFIG with `config` merged over it."""
    return {**MODEL_CONFIG, **config}


def write(case, cfg, weights, inputs, outputs, extra=None):
    d = OUT / case
    d.mkdir(parents=True, exist_ok=True)
    er.save_json(d / "config.json", er.jsonable({**cfg, "upstream_commit": er.upstream_sha(),
                                                  "tensorflow": tf.__version__, "numpy": np.__version__}))
    er.save_npz(d / "weights.npz", weights)
    er.save_npz(d / "inputs.npz", inputs)
    er.save_npz(d / "outputs.npz", {k: np.asarray(outputs[k]) for k in sorted(outputs)})
    if extra:
        er.save_json(d / "losses.json", er.jsonable(extra))


def case_tcn(mns, rng):
    er.set_seeds(er.SEED)
    kw = {**SMALL, "encoder": "tcn"}
    net = mns["build_model_fn"](32, 6, config=kw)
    x = rng.standard_normal((8, 32, 6)).astype(np.float32)
    out = net(tf.constant(x), training=False)
    write("tcn", {"seq_len": 32, "n_features": 6, "config": kw, "constructor_kwargs": full_kwargs(kw)}, nig_weights(net), {"input_sequence": x}, out)


def case_multitf(mns, rng):
    er.set_seeds(er.SEED)
    seq_len, n_features = {"1d": 32, "4h": 24}, {"1d": 6, "4h": 5}
    net = mns["build_model_fn"](seq_len, n_features, config=dict(SMALL))
    xs = {tf_: rng.standard_normal((8, seq_len[tf_], n_features[tf_])).astype(np.float32) for tf_ in seq_len}
    out = net({k: tf.constant(v) for k, v in xs.items()}, training=False)
    write("multitf", {"seq_len": seq_len, "n_features": n_features, "config": SMALL,
                       "constructor_kwargs": full_kwargs(SMALL)}, nig_weights(net), xs, out)


def case_multiclass(mns, tns, rng):
    er.set_seeds(er.SEED)
    head_types = {t: ["nig_regression", "binary_classification"] for t in ("high", "low")}
    head_types["close"] = ["nig_regression", "multiclass_classification"]
    kw = {**SMALL, "head_types": head_types, "n_classes": 4}
    net = mns["build_model_fn"](32, 6, config=kw)
    x = rng.standard_normal((8, 32, 6)).astype(np.float32)
    out = net(tf.constant(x), training=False)
    y = rng.integers(0, 4, size=(8,)).astype(np.int32)
    w = rng.uniform(0.5, 1.5, size=(8,)).astype(np.float32)
    loss_fn = tns["classification_task_loss"]
    losses = {}
    for ls in (0.0, 0.1):
        for weighted in (False, True):
            cfg = {"output_keys": {"logits": "y_close_class_logits"}, "binary": False, "label_smoothing": ls}
            loss, stats = loss_fn(tf.constant(y), out, cfg, {}, sample_weight=tf.constant(w) if weighted else None)
            losses[f"ls={ls},weighted={weighted}"] = {"loss": er.f32(loss), "accuracy": er.f32(stats["accuracy"])}
    write("multiclass", {"seq_len": 32, "n_features": 6, "config": kw, "constructor_kwargs": full_kwargs(kw)}, nig_weights(net),
          {"input_sequence": x, "y_close_class": y, "sample_weight": w}, out, extra=losses)


def case_panel(mns, rng):
    import importlib.util  # load cross_asset/model.py by path: the package __init__ pulls pandas (data side)

    spec = importlib.util.spec_from_file_location("upstream_cross_asset_model", er.UPSTREAM / "cross_asset" / "model.py")
    panel_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panel_mod)

    for attention in (True, False):
        er.set_seeds(er.SEED)
        base = mns["build_model_fn"](32, 6, config=dict(SMALL))
        enc = panel_mod.extract_encoder(base)
        cfg = {"d_model": 16, "n_cross_layers": 1, "num_heads": 4, "cross_attention": attention, "dropout": 0.25,
               "attn_dropout": 0.1, "ff_mult": 2, "class_head_hidden": 32, "head_hidden": 64}
        m = panel_mod.build_panel_model(enc, 32, 6, **cfg)
        day = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3], dtype=np.int32)
        pos = np.array([0, 1, 2, 0, 1, 0, 1, 2, 3, 0], dtype=np.int32)
        x = rng.standard_normal((len(day), 32, 6)).astype(np.float32)
        out = m({"x": tf.constant(x), "day": tf.constant(day), "pos": tf.constant(pos)}, training=False)
        enc_ids = {id(v) for v in enc.weights}
        weights = {f"encoder.{k}": v for k, v in nig_weights(base).items()
                   if any(id(w) in enc_ids for w in base.weights if er.attr_paths(base)[id(w)] == k)}
        for v in m.weights:
            if id(v) not in enc_ids:
                weights[v.path] = np.asarray(v)
        case = "panel" if attention else "panel_no_attention"
        write(case, {"seq_len": 32, "n_features": 6, "encoder_config": SMALL,
                      "encoder_constructor_kwargs": full_kwargs(SMALL), "panel_config": cfg}, weights,
              {"x": x, "day": day, "pos": pos}, out)


def main():
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    global MODEL_CONFIG
    mns, tns, _ = er.load_upstream()
    MODEL_CONFIG = dict(mns["MODEL_CONFIG"])
    rng = np.random.default_rng(er.SEED)
    case_tcn(mns, rng)
    case_multitf(mns, rng)
    case_multiclass(mns, tns, rng)
    case_panel(mns, rng)
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
