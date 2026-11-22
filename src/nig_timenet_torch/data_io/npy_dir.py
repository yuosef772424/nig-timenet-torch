"""PURPOSE: The npy_dir dataset format (manifest.json + one .npy per array + meta.pkl): its reader (optionally memory-
    mapped), its writer and a local-disk copy helper.
TAGS: npy dir, dataset reader, dataset writer, mmap, localize, data format, manifest, sha256
PITFALLS: with mmap=True arrays up to small_bytes (64 MiB) are read into memory, only bigger ones stay memmaps; localize
    copies and save writes manifest.json last so an interrupted copy / write is never taken for a finished one; save
    writes into the given folder (the contract writer, data_io.artifact_writer, adds the temp folder + rename).

Reader of the ``npy_dir`` dataset format (BUILD_PLAN.md P06; contract: docs/DATA_ARTIFACT_CONTRACT.md).

On-disk format, written by crypto-signal-prediction ``data/storage.py::save_dataset_dir`` (@ aeafc56, lines 61-85):
one ``<key>.npy`` per numpy-array key, ``meta.pkl`` (pickle) with every other key, ``manifest.json`` written last as
the completion marker (the TF side writes into a temp dir and renames it, so a missing manifest = never finished).
Reading order mirrors ``load_dataset_dir`` (lines 120-129): manifest, meta.pkl, arrays, then keys in ``key_order``.
Pure numpy: no torch import here.
"""

import datetime as dt
import hashlib
import json
import pickle
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np

__all__ = ["load", "save", "localize", "sha256_file", "SMALL_ARRAY_BYTES"]

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


def sha256_file(path: str | Path) -> str:
    """sha256 hex digest of a file (read in 4 MiB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def save(dataset: dict, directory: str | Path, *, export: Callable[[dict[str, str]], dict] | None = None) -> dict:
    """Write ``dataset`` (a dict of numpy arrays and other values) as an npy_dir into the existing folder
    ``directory``: one ``<key>.npy`` per non-object array, ``meta.pkl`` with every other key, ``manifest.json`` LAST
    (key order, array shapes / dtypes, a fresh id and creation time). ``export(files)`` is called with
    ``{file name: sha256}`` of every written file and returns the manifest's ``export`` block (None = no block).
    Returns the manifest dict."""
    directory = Path(directory)
    arrays = {k: v for k, v in dataset.items() if isinstance(v, np.ndarray) and not v.dtype.hasobject}
    meta = {k: v for k, v in dataset.items() if k not in arrays}
    for k, a in arrays.items():
        np.save(directory / f"{k}.npy", a, allow_pickle=False)
    with open(directory / "meta.pkl", "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    files = {f"{k}.npy": sha256_file(directory / f"{k}.npy") for k in sorted(arrays)}
    files["meta.pkl"] = sha256_file(directory / "meta.pkl")
    manifest = {"format": "npy_dir", "version": 1, "id": uuid.uuid4().hex,
                "created": dt.datetime.now(dt.timezone.utc).isoformat(),
                "key_order": list(dataset), "arrays": {k: {"shape": list(a.shape), "dtype": str(a.dtype)}
                                                        for k, a in arrays.items()}}
    if export is not None:
        manifest["export"] = export(files)
    with open(directory / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    return manifest
