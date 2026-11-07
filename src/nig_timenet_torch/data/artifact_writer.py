"""Contract artifact writer

PURPOSE: Write a built dataset as a contract artifact (npy_dir plus manifest export block) that
    data_io.load_artifact accepts, dropping sealed coins and windows after the time seal.
TAGS: artifact writer, contract artifact, manifest, time seal, sealed coins, sha256
PITFALLS: windows at or after the time seal are dropped, never clipped (rule D1); the folder is renamed into place,
    manifest last, so a half-written artifact never loads.

Write a built dataset as a contract artifact (npy_dir + manifest ``export`` block) that
``data_io.artifact.load_artifact`` accepts.

Windows whose label bar is at or after the time seal are dropped (never clipped) and so is every sealed coin; the
manifest records the source commit, the coin list, the full sealed list, the lock hash, the latest label bar and a
sha256 of every file. The folder is written next to its destination and renamed, manifest last (completion marker).
"""

import datetime as dt
import hashlib
import json
import pickle
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.data.lock import HoldoutLock, is_sealed, load_lock
from nig_timenet_torch.data.timeframe import tf_timedelta
from nig_timenet_torch.data_io.artifact import SCHEMA

__all__ = ["write_artifact", "git_head", "restrict_to_time_seal"]

_SHA1 = re.compile(r"^[0-9a-f]{40}$")


def git_head(repo: str | Path | None = None) -> str:
    """HEAD commit of the checkout this package lives in (or of `repo`); raises when there is none."""
    repo = Path(repo) if repo else Path(__file__).resolve().parent
    try:
        out = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError("no git checkout to read the source commit from: pass OutputSettings.source_commit "
                           "(40 hex)") from e
    return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def restrict_to_time_seal(dataset: dict, lock: HoldoutLock) -> dict:
    """Drop sealed coins and windows whose label bar reaches the time seal; asset_bounds are re-tiled."""
    last = np.asarray(dataset["last_candles"])
    n = len(last)
    bar_ns = int(tf_timedelta(dataset["base_tf"]).value)
    label_ns = last[:, TS_COL].astype("int64") + int(dataset["forecast_horizon"]) * bar_ns
    keep = label_ns < lock.time_seal.value
    for b in dataset["asset_bounds"]:
        if is_sealed(b["name"], lock):
            keep[b["start"]:b["end"]] = False
    if keep.all():
        return dataset
    out, bounds, pos = {}, [], 0
    for b in dataset["asset_bounds"]:
        cnt = int(keep[b["start"]:b["end"]].sum())
        if cnt:
            bounds.append({**b, "start": pos, "end": pos + cnt})
            pos += cnt
    for k, v in dataset.items():
        out[k] = v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v
    out["asset_bounds"] = bounds
    return out


def _split_keys(dataset: dict) -> tuple[dict, dict]:
    arrays = {k: v for k, v in dataset.items() if isinstance(v, np.ndarray) and not v.dtype.hasobject}
    return arrays, {k: v for k, v in dataset.items() if k not in arrays}


def write_artifact(dataset: dict, out_dir: str | Path, *, source_commit: str = "", producer: str = "",
                   overwrite: bool = False, lock: HoldoutLock | None = None) -> dict:
    """Write `dataset` (the dict `build_dataset` returns) to `out_dir` and return the manifest."""
    lock = lock or load_lock()
    dataset = restrict_to_time_seal(dataset, lock)
    n = len(dataset["last_candles"])
    if n == 0:
        raise ValueError("no window is left after removing sealed coins and the sealed period")
    sha = source_commit or git_head()
    if not _SHA1.match(sha):
        raise ValueError(f"source_commit must be 40 hex characters, got {sha!r}")
    out = Path(out_dir).resolve()
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass OutputSettings.overwrite=True to replace it")
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    arrays, meta = _split_keys(dataset)
    for k, a in arrays.items():
        np.save(tmp / f"{k}.npy", a, allow_pickle=False)
    with open(tmp / "meta.pkl", "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    files = {f"{k}.npy": _sha256_file(tmp / f"{k}.npy") for k in sorted(arrays)}
    files["meta.pkl"] = _sha256_file(tmp / "meta.pkl")
    last = np.asarray(dataset["last_candles"])
    bar_ns = int(tf_timedelta(dataset["base_tf"]).value)
    ts = last[:, TS_COL].astype("int64")
    iso = lambda ns: pd.Timestamp(int(ns), tz="UTC").isoformat()  # noqa: E731
    manifest = {
        "format": "npy_dir", "version": 1, "id": uuid.uuid4().hex,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(),
        "key_order": list(dataset), "arrays": {k: {"shape": list(a.shape), "dtype": str(a.dtype)}
                                                for k, a in arrays.items()},
        "export": {
            "schema": SCHEMA, "source_commit": sha, "producer": producer or "nig_timenet_torch.data",
            "coins": [b["name"] for b in dataset["asset_bounds"]], "sealed_excluded": sorted(lock.sealed),
            "lock_sha256": lock.sha256, "time_seal": lock.time_seal.isoformat(),
            "max_bar_time": iso(ts.max() + int(dataset["forecast_horizon"]) * bar_ns),
            "max_window_time": iso(ts.max()), "rows": n,
            "base_tf": dataset["base_tf"], "timeframes": [str(t) for t in dataset["timeframes"]], "files": files,
            "data_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}}
    with open(tmp / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return manifest
