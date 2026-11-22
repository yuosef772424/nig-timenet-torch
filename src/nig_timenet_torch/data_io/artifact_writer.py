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

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.label_time import label_bar_open
from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.core.timeframe import tf_timedelta
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.data_io.artifact import SCHEMA, SHA1_RE
from nig_timenet_torch.data_io.lock import HoldoutLock, is_sealed, load_lock

__all__ = ["write_artifact", "git_head", "restrict_to_time_seal"]


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


def restrict_to_time_seal(dataset: dict, lock: HoldoutLock) -> dict:
    """Drop sealed coins and windows whose label bar reaches the time seal; asset_bounds are re-tiled."""
    last = np.asarray(dataset["last_candles"])
    n = len(last)
    bar_ns = int(tf_timedelta(dataset["base_tf"]).value)
    label_ns = label_bar_open(last[:, TS_COL].astype("int64"), dataset["forecast_horizon"], bar_ns)
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


def write_artifact(dataset: dict, out_dir: str | Path, *, source_commit: str = "", producer: str = "",
                   overwrite: bool = False, lock: HoldoutLock | None = None) -> dict:
    """Write `dataset` (the dict `build_dataset` returns) to `out_dir` and return the manifest."""
    lock = lock or load_lock()
    dataset = restrict_to_time_seal(dataset, lock)
    n = len(dataset["last_candles"])
    if n == 0:
        raise ValueError("no window is left after removing sealed coins and the sealed period")
    sha = source_commit or git_head()
    if not SHA1_RE.match(sha):
        raise ValueError(f"source_commit must be 40 hex characters, got {sha!r}")
    out = Path(out_dir).resolve()
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass OutputSettings.overwrite=True to replace it")
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    bar_ns = int(tf_timedelta(dataset["base_tf"]).value)
    ts = np.asarray(dataset["last_candles"])[:, TS_COL].astype("int64")
    iso = lambda ns: pd.Timestamp(int(ns), tz="UTC").isoformat()  # noqa: E731

    def export(files: dict[str, str]) -> dict:
        return {
            "schema": SCHEMA, "source_commit": sha, "producer": producer or "nig_timenet_torch.data",
            "coins": [b["name"] for b in dataset["asset_bounds"]], "sealed_excluded": sorted(lock.sealed),
            "lock_sha256": lock.sha256, "time_seal": lock.time_seal.isoformat(),
            "max_bar_time": iso(label_bar_open(ts.max(), dataset["forecast_horizon"], bar_ns)),
            "max_window_time": iso(ts.max()), "rows": n,
            "base_tf": dataset["base_tf"], "timeframes": [str(t) for t in dataset["timeframes"]], "files": files,
            "data_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}

    manifest = npy_dir.save(dataset, tmp, export=export)
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return manifest
