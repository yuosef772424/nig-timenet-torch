"""Reader of the ``npy_dir`` dataset format (BUILD_PLAN.md P06; contract: docs/DATA_ARTIFACT_CONTRACT.md).

On-disk format, written by crypto-signal-prediction ``data/storage.py::save_dataset_dir`` (@ aeafc56, lines 61-85):
one ``<key>.npy`` per numpy-array key, ``meta.pkl`` (pickle) with every other key, ``manifest.json`` written last as
the completion marker (the TF side writes into a temp dir and renames it, so a missing manifest = never finished).
Reading order mirrors ``load_dataset_dir`` (lines 120-129): manifest, meta.pkl, arrays, then keys in ``key_order``.
Pure numpy: no torch import here.
"""

import json
import pickle
import shutil
from pathlib import Path

import numpy as np

__all__ = ["load", "localize", "SMALL_ARRAY_BYTES"]

#: with ``mmap=True`` arrays up to this size are read into memory (labels, base params, last_candles); bigger ones
#: stay on disk and only the pages that are read enter memory (same threshold as the TF project)
SMALL_ARRAY_BYTES = 64 * 1024 * 1024


def localize(src: str | Path, local_dir: str | Path) -> Path:
    """Copy a dataset folder to ``local_dir`` (for example from a mounted Drive to the VM disk) and return the copy.
    A copy that already matches (same manifest bytes, same file sizes) is reused. manifest.json is copied last, so
    an interrupted copy is never mistaken for a finished one. ``local_dir`` equal to ``src`` returns ``src``."""
    src, dst = Path(src), Path(local_dir)
    if not (src / "manifest.json").is_file():
        raise FileNotFoundError(f"incomplete or missing npy_dir dataset (no manifest.json): {src}")
    if dst.resolve() == src.resolve():
        return src
    files = [p for p in src.iterdir() if p.is_file()]
    same = ((dst / "manifest.json").is_file()
            and (dst / "manifest.json").read_bytes() == (src / "manifest.json").read_bytes()
            and all((dst / p.name).is_file() and (dst / p.name).stat().st_size == p.stat().st_size for p in files))
    if not same:
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True)
        for p in sorted(files, key=lambda q: q.name == "manifest.json"):
            shutil.copyfile(p, dst / p.name)
    return dst


def load(path: str | Path, *, mmap: bool = False, small_bytes: int = SMALL_ARRAY_BYTES) -> dict:
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
        a = np.load(src / f"{k}.npy", allow_pickle=False, mmap_mode="r" if mmap else None)
        if list(a.shape) != list(spec["shape"]) or str(a.dtype) != spec["dtype"]:
            raise ValueError(f"{src}: {k}.npy is {a.dtype}{list(a.shape)}, "
                             f"manifest says {spec['dtype']}{spec['shape']}")
        arrays[k] = np.array(a) if mmap and a.nbytes <= small_bytes else a
    missing = [k for k in manifest["key_order"] if k not in arrays and k not in meta]
    if missing:
        raise ValueError(f"{src}: keys {missing} listed in manifest key_order are neither arrays nor in meta.pkl")
    return {k: (arrays[k] if k in arrays else meta[k]) for k in manifest["key_order"]}
