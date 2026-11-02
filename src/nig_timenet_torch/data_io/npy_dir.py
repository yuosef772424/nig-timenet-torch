"""Reader of the ``npy_dir`` dataset format (BUILD_PLAN.md P06; contract: docs/DATA_ARTIFACT_CONTRACT.md).

On-disk format, written by crypto-signal-prediction ``data/storage.py::save_dataset_dir`` (@ aeafc56, lines 61-85):
one ``<key>.npy`` per numpy-array key, ``meta.pkl`` (pickle) with every other key, ``manifest.json`` written last as
the completion marker (the TF side writes into a temp dir and renames it, so a missing manifest = never finished).
Reading order mirrors ``load_dataset_dir`` (lines 120-129): manifest, meta.pkl, arrays, then keys in ``key_order``.
Pure numpy: no torch import here.
"""

import json
import pickle
from pathlib import Path

import numpy as np

__all__ = ["load"]


def load(path: str | Path) -> dict:
    """Read an npy_dir dataset (manifest.json + <key>.npy per array + meta.pkl) into a dict with the same keys
    the TF `load_dataset_dir` returns. Raises FileNotFoundError if manifest.json is missing (no partial read:
    `save_dataset_dir` on the TF side renames a temp dir atomically, so a missing manifest means "never finished
    writing", not "corrupt")."""
    src = Path(path)
    man_path = src / "manifest.json"
    if not man_path.is_file():
        raise FileNotFoundError(f"incomplete or missing npy_dir dataset (no manifest.json): {src}")
    manifest = json.loads(man_path.read_text())
    if manifest.get("format") != "npy_dir" or manifest.get("version") != 1:
        raise ValueError(f"{src}: unsupported manifest format/version "
                         f"{manifest.get('format')!r}/{manifest.get('version')!r} (expected 'npy_dir'/1)")
    with open(src / "meta.pkl", "rb") as f:
        meta = pickle.load(f)  # trusted input: datasets written by the project's own pipeline
    arrays = {}
    for k, spec in manifest["arrays"].items():
        a = np.load(src / f"{k}.npy", allow_pickle=False)
        if list(a.shape) != list(spec["shape"]) or str(a.dtype) != spec["dtype"]:
            raise ValueError(f"{src}: {k}.npy is {a.dtype}{list(a.shape)}, "
                             f"manifest says {spec['dtype']}{spec['shape']}")
        arrays[k] = a
    missing = [k for k in manifest["key_order"] if k not in arrays and k not in meta]
    if missing:
        raise ValueError(f"{src}: keys {missing} listed in manifest key_order are neither arrays nor in meta.pkl")
    return {k: (arrays[k] if k in arrays else meta[k]) for k in manifest["key_order"]}
