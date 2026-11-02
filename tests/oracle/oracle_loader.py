"""The ONE reader of the TF oracle in reference/data/ (BUILD_PLAN.md P07). Every parity test imports from here; no
test opens reference/data/* directly, so the tolerance table and the MAJOR-2 shape workaround live in one place.

MAJOR-2 (ORACLE_REVIEW.md): reference/export_reference.py's save_npz writes 0-d arrays as shape (1,)
(np.ascontiguousarray promotes them), so weights.npz / grads.npz store the 3 Kendall scalars
``trainer/uncertainty_weighted_loss/log_var_{high,low,close}_class`` as (1,) while weights.json / grads.json declare ().
Every array is therefore reshaped to its JSON-declared shape here, and ``_reshape`` asserts that ONLY those 3 names ever
needed it -- if the exporter is fixed (or another entry starts to differ), the test suite says so instead of hiding it.
Do not fix the exporter from here (it belongs to the audit PR); this module only works around it.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "reference" / "data"

#: BUILD_PLAN.md section 0 tolerance table, as code. Never loosen without a docs/friction_log.csv row.
TOLERANCE = {"forward": dict(rtol=1e-5, atol=1e-6), "loss": dict(rtol=1e-5, atol=1e-6),
             "grad": dict(rtol=1e-4, atol=1e-6)}

#: The only entries whose stored .npy shape differs from the JSON-declared shape (ORACLE_REVIEW.md MAJOR-2).
MAJOR2_NAMES = frozenset(f"trainer/uncertainty_weighted_loss/log_var_{t}_class" for t in ("high", "low", "close"))


def _json(name, data=DATA):
    return json.loads((data / name).read_text())


def _npz(name, data=DATA):
    with np.load(data / name, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _reshape(name, arr, shape):
    shape = tuple(shape)
    if arr.shape != shape:
        if name not in MAJOR2_NAMES:
            raise AssertionError(f"oracle entry {name!r}: stored shape {arr.shape} != declared {shape} "
                                 "(only the 3 MAJOR-2 Kendall scalars are expected to differ)")
        arr = np.asarray(arr).reshape(shape)
    return arr


def _t(a):
    return torch.from_numpy(np.array(a, copy=True))


def load_config(data=DATA) -> dict:
    return _json("config.json", data)


def load_weights_meta(data=DATA) -> list[dict]:
    return _json("weights.json", data)


def _key(meta_entry):
    """attr_path for model variables (stable, e.g. 'block_1.attn.wq.kernel'); the Keras path for trainer variables."""
    return meta_entry.get("attr_path") or meta_entry["name"]


@lru_cache(maxsize=4)
def _weights_np(data=DATA):
    raw = _npz("weights.npz", data)
    return {_key(m): _reshape(m["name"], raw[m["name"]], m["shape"]) for m in load_weights_meta(data)}


def load_weights(data=DATA) -> dict[str, torch.Tensor]:
    """weights.npz keyed by attr_path (trainer entries by their 'trainer/...' name), shapes as weights.json declares."""
    return {k: _t(v) for k, v in _weights_np(data).items()}


def load_layer_outputs(data=DATA) -> dict[str, torch.Tensor]:
    return {k: _t(v) for k, v in _npz("layer_outputs.npz", data).items()}


def load_layers_meta(data=DATA) -> list[dict]:
    return _json("layers.json", data)


def load_sublayers_meta(data=DATA) -> list[dict]:
    return _json("sublayers.json", data)


def load_sublayers(data=DATA) -> dict[str, torch.Tensor]:
    raw = _npz("sublayers.npz", data)
    return {m["key"]: _t(_reshape(m["key"], raw[m["key"]], m["shape"])) for m in load_sublayers_meta(data)}


def load_outputs(data=DATA) -> dict[str, torch.Tensor]:
    return {k: _t(v) for k, v in _npz("outputs.npz", data).items()}


def load_losses(data=DATA) -> dict:
    """losses.json (both scenarios; ``primary`` names the one grads.npz was taken in)."""
    return _json("losses.json", data)


def load_grads(data=DATA) -> dict[str, torch.Tensor]:
    """grads.npz keyed like load_weights (attr_path / 'trainer/...'), shapes as grads.json declares."""
    raw = _npz("grads.npz", data)
    by_name = {m["name"]: _key(m) for m in load_weights_meta(data)}
    return {by_name[m["name"]]: _t(_reshape(m["name"], raw[m["name"]], m["shape"])) for m in _json("grads.json", data)}


def load_inputs(data=DATA) -> torch.Tensor:
    return _t(_npz("inputs.npz", data)["input_sequence"])


def load_labels(data=DATA) -> dict[str, torch.Tensor]:
    return {k: _t(v) for k, v in _npz("labels.npz", data).items()}


def assert_close(actual, expected, kind: str, msg: str = ""):
    """torch.allclose with TOLERANCE[kind]; AssertionError with shape / max-abs-diff details on failure."""
    tol = TOLERANCE[kind]
    a = torch.as_tensor(actual).detach().to(torch.float64)
    e = torch.as_tensor(expected).detach().to(torch.float64)
    if a.shape != e.shape:
        raise AssertionError(f"{msg}: shape {tuple(a.shape)} != expected {tuple(e.shape)}")
    if not torch.allclose(a, e, **tol):
        diff = (a - e).abs()
        rel = diff / e.abs().clamp_min(1e-30)
        worst = tuple(int(i) for i in torch.unravel_index(diff.argmax(), diff.shape))
        raise AssertionError(f"{msg}: not close at {kind} tolerance {tol}: max abs diff {diff.max().item():.3e}, "
                             f"max rel diff {rel.max().item():.3e}, worst index {worst}")
