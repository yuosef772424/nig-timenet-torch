"""Loader of a contract artifact exported by the TF side (ROADMAP T2, rule D1; docs/DATA_ARTIFACT_CONTRACT.md).

The artifact is an ``npy_dir`` dataset (data_io/npy_dir.py) whose ``manifest.json`` carries an ``export`` block
written by crypto-signal-prediction ``tools/export_torch_artifact.py``. ``load_artifact`` REFUSES an artifact without
that block, or with any of the D1 fields missing or inconsistent: source commit, coin list, sealed coins excluded, max
bar time below the time seal. It does not trust the manifest alone: the sealed list must have the size pinned in this
module and the lock file hash must be the pinned one (``EXPECTED_SEALED_COUNT``, ``EXPECTED_LOCK_SHA256``); the coin
list must equal the dataset's own ``asset_bounds``, no coin may be on the sealed list, and the max bar time is
recomputed from ``last_candles`` and must not exceed what the manifest says.
Pure numpy/pandas: no torch import here.

    python -m nig_timenet_torch.data_io.artifact <artifact dir> [--verify-hashes]
"""

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.data_io import npy_dir

__all__ = ["ArtifactError", "ArtifactManifest", "Artifact", "SCHEMA", "TIME_SEAL", "EXPECTED_SEALED_COUNT",
           "EXPECTED_LOCK_SHA256", "read_manifest", "load_artifact", "verify_hashes"]

SCHEMA = "torch_artifact/1"
#: The time seal of the TF holdout lock (docs/DATA_ARTIFACT_CONTRACT.md): no bar, and no label, at or after it.
TIME_SEAL = pd.Timestamp("2026-10-01", tz="UTC")
#: Pinned HERE, not read from the manifest: the TF holdout lock (crypto-signal-prediction
#: docs/research/plan/lock/holdout.json, rule R0) seals this many coins and has this file sha256 (= its holdout.sha256).
#: A manifest that reports another count or hash was exported against a different lock and is refused. If the TF lock is
#: deliberately rebuilt, change both constants in a PR that says so (owner decision), never to make an artifact load.
EXPECTED_SEALED_COUNT = 183
EXPECTED_LOCK_SHA256 = "70322ba699d5068e4fbe7f8b014f85492aeebf15d3c743551fbf1062cd889b2c"
REQUIRED = ("schema", "source_commit", "coins", "sealed_excluded", "lock_sha256", "time_seal", "max_bar_time",
            "data_sha256")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ArtifactError(ValueError):
    """The directory is not an acceptable contract artifact; the message says which field or check failed."""


@dataclass(frozen=True)
class ArtifactManifest:
    """The D1 fields of an artifact. ``data_fingerprint`` is what ``build_run_identity`` takes as data fingerprint."""
    source_commit: str
    coins: tuple[str, ...]
    sealed_excluded: tuple[str, ...]
    time_seal: pd.Timestamp
    max_bar_time: pd.Timestamp
    data_fingerprint: str
    files: dict[str, str] | None


@dataclass(frozen=True)
class Artifact:
    dataset: dict
    manifest: ArtifactManifest
    path: Path


def _norm(symbol: str) -> str:
    """Name as the TF guard compares it: upper case, with or without the trailing USDT."""
    n = str(symbol).strip().upper()
    return n[:-4] if n.endswith("USDT") and len(n) > 4 else n


def _timestamp(value, field: str) -> pd.Timestamp:
    try:
        t = pd.Timestamp(value)
    except (ValueError, TypeError) as e:
        raise ArtifactError(f"export.{field}: not a timestamp: {value!r}") from e
    if pd.isna(t):
        raise ArtifactError(f"export.{field}: not a timestamp: {value!r}")
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _name_list(block: dict, field: str) -> tuple[str, ...]:
    v = block[field]
    if not isinstance(v, list) or not v or not all(isinstance(c, str) and c.strip() for c in v):
        raise ArtifactError(f"export.{field}: must be a non-empty list of coin names, got {v!r:.80}")
    if len({_norm(c) for c in v}) != len(v):
        raise ArtifactError(f"export.{field}: duplicate names")
    return tuple(v)


def read_manifest(path: str | Path) -> ArtifactManifest:
    """Parse and check ``<path>/manifest.json``'s ``export`` block (no array is read)."""
    man_path = Path(path) / "manifest.json"
    if not man_path.is_file():
        raise ArtifactError(f"{path}: no manifest.json (incomplete or not an artifact directory)")
    block = json.loads(man_path.read_text()).get("export")
    if not isinstance(block, dict):
        raise ArtifactError(f"{path}: manifest.json has no 'export' block: not a contract artifact (rule D1); "
                            f"export it with crypto-signal-prediction tools/export_torch_artifact.py")
    missing = [k for k in REQUIRED if k not in block]
    if missing:
        raise ArtifactError(f"{path}: export block lacks required fields {missing} (rule D1)")
    if block["schema"] != SCHEMA:
        raise ArtifactError(f"{path}: export.schema {block['schema']!r} is not {SCHEMA!r}")
    if not isinstance(block["source_commit"], str) or not _SHA1.match(block["source_commit"]):
        raise ArtifactError(f"{path}: export.source_commit must be a 40-hex git commit, got {block['source_commit']!r}")
    if not isinstance(block["data_sha256"], str) or not _SHA256.match(block["data_sha256"]):
        raise ArtifactError(f"{path}: export.data_sha256 must be a 64-hex sha256")
    coins, sealed = _name_list(block, "coins"), _name_list(block, "sealed_excluded")
    if len(sealed) != EXPECTED_SEALED_COUNT:
        raise ArtifactError(f"{path}: export.sealed_excluded has {len(sealed)} coins, this project pins "
                            f"{EXPECTED_SEALED_COUNT} (the TF holdout lock): the artifact was exported against another "
                            f"lock or with a truncated sealed list")
    if block["lock_sha256"] != EXPECTED_LOCK_SHA256:
        raise ArtifactError(f"{path}: export.lock_sha256 {block['lock_sha256']!r} is not the pinned TF lock "
                            f"{EXPECTED_LOCK_SHA256} (exported against another lock)")
    sealed_names = {_norm(s) for s in sealed}
    leaked = [c for c in coins if _norm(c) in sealed_names]
    if leaked:
        raise ArtifactError(f"{path}: coins {leaked[:5]} are on the sealed list (export.sealed_excluded): a sealed "
                            f"coin must never be in a training artifact")
    seal, max_bar = _timestamp(block["time_seal"], "time_seal"), _timestamp(block["max_bar_time"], "max_bar_time")
    if seal != TIME_SEAL:
        raise ArtifactError(f"{path}: export.time_seal {seal.isoformat()} is not {TIME_SEAL.isoformat()}")
    if max_bar >= TIME_SEAL:
        raise ArtifactError(f"{path}: export.max_bar_time {max_bar.isoformat()} is not before the time seal "
                            f"{TIME_SEAL.isoformat()}")
    files = block.get("files")
    if files is not None and not (isinstance(files, dict) and all(_SHA256.match(str(v)) for v in files.values())):
        raise ArtifactError(f"{path}: export.files must map file names to sha256 hex digests")
    return ArtifactManifest(source_commit=block["source_commit"], coins=coins, sealed_excluded=sealed, time_seal=seal,
                            max_bar_time=max_bar, data_fingerprint=block["data_sha256"],
                            files=None if files is None else dict(files))


def _bar_ns(tf: str) -> int:
    d = pd.Timedelta(tf)
    if d <= pd.Timedelta(0):
        raise ArtifactError(f"base_tf {tf!r} is not a positive duration")
    return int(d.value)


def load_artifact(path: str | Path, *, mmap: bool = False) -> Artifact:
    """Read an artifact and refuse it unless every D1 field is present and consistent with the data itself.
    ``mmap=True`` keeps the big arrays on disk (only the pages that are read enter memory)."""
    manifest = read_manifest(path)
    dataset = npy_dir.load(path, mmap=mmap)
    names = [b["name"] for b in dataset["asset_bounds"]]
    if list(manifest.coins) != names:
        raise ArtifactError(f"{path}: export.coins {list(manifest.coins)[:5]}... differ from the dataset's "
                            f"asset_bounds {names[:5]}... (the manifest does not describe this data)")
    ts = np.asarray(dataset["last_candles"])[:, TS_COL].astype("int64")
    latest = pd.Timestamp(int(ts.max()) + int(dataset["forecast_horizon"]) * _bar_ns(dataset["base_tf"]), tz="UTC")
    if latest >= TIME_SEAL:
        raise ArtifactError(f"{path}: a window's label bar ({latest.isoformat()}) is at or after the time seal "
                            f"{TIME_SEAL.isoformat()}")
    if latest > manifest.max_bar_time:
        raise ArtifactError(f"{path}: the data reaches {latest.isoformat()}, later than export.max_bar_time "
                            f"{manifest.max_bar_time.isoformat()}")
    return Artifact(dataset=dataset, manifest=manifest, path=Path(path))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hashes(path: str | Path) -> None:
    """Re-hash every file listed in ``export.files`` and the combined ``data_sha256``; raises ArtifactError on a
    difference. Reads the whole artifact once (slow on a big one), so it is separate from ``load_artifact``."""
    manifest = read_manifest(path)
    if manifest.files is None:
        raise ArtifactError(f"{path}: export.files is missing, cannot verify hashes")
    for name, want in manifest.files.items():
        f = Path(path) / name
        if not f.is_file():
            raise ArtifactError(f"{path}: {name} is listed in export.files but missing")
        if _sha256_file(f) != want:
            raise ArtifactError(f"{path}: {name} does not match its sha256 in export.files")
    combined = hashlib.sha256(json.dumps(manifest.files, sort_keys=True).encode()).hexdigest()
    if combined != manifest.data_fingerprint:
        raise ArtifactError(f"{path}: export.data_sha256 does not match export.files")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Check a TF-exported contract artifact and print what it holds.")
    ap.add_argument("path")
    ap.add_argument("--verify-hashes", action="store_true", help="also re-hash every file (reads the whole artifact)")
    args = ap.parse_args(argv)
    try:
        art = load_artifact(args.path)
        if args.verify_hashes:
            verify_hashes(args.path)
    except (ArtifactError, FileNotFoundError, ValueError) as e:
        print(f"artifact refused: {e}", file=sys.stderr)
        return 1
    m = art.manifest
    print(f"OK {art.path}: {len(art.dataset['last_candles'])} windows, {len(m.coins)} coins, "
          f"source {m.source_commit[:10]}, max_bar_time {m.max_bar_time.isoformat()}, "
          f"sealed excluded {len(m.sealed_excluded)}, "
          f"data_fingerprint {m.data_fingerprint[:12]}{' (hashes verified)' if args.verify_hashes else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
