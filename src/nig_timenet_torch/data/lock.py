"""Holdout lock

PURPOSE: The sealed-coin list of the holdout lock, shipped as a hash-pinned package resource; sealed coins are never
    built.
TAGS: holdout lock, sealed coins, time seal, sha256, package resource
PITFALLS: the packaged holdout.json must hash to the value pinned in data_io; changing it needs an owner decision,
    not an edit.

The sealed-coin list of the holdout lock (a copy of crypto-signal-prediction docs/research/plan/lock/holdout.json
shipped as a package resource). The pipeline never builds a window of a sealed coin: the contract artifact
(data_io.artifact) refuses any that is listed. The file's hash is checked against the value pinned in data_io.
"""

import hashlib
import json
from dataclasses import dataclass
from importlib import resources

import pandas as pd

from nig_timenet_torch.data_io.artifact import EXPECTED_LOCK_SHA256, EXPECTED_SEALED_COUNT, TIME_SEAL

__all__ = ["HoldoutLock", "load_lock", "is_sealed"]


@dataclass(frozen=True)
class HoldoutLock:
    sealed: tuple[str, ...]
    time_seal: pd.Timestamp
    sha256: str


def load_lock() -> HoldoutLock:
    """The packaged lock; raises if it is not the pinned one (a changed lock is an owner decision, not a patch)."""
    raw = resources.files("nig_timenet_torch.data").joinpath("resources/holdout.json").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != EXPECTED_LOCK_SHA256:
        raise ValueError(f"packaged holdout.json has sha256 {sha}, the pinned lock is {EXPECTED_LOCK_SHA256}")
    d = json.loads(raw)
    sealed = tuple(sorted(str(s).strip().upper() for s in d["sealed"]))
    if len(sealed) != EXPECTED_SEALED_COUNT or pd.Timestamp(d["time_seal"]).tz_convert("UTC") != TIME_SEAL:
        raise ValueError("packaged holdout.json does not match the sealed count / time seal pinned in data_io.artifact")
    return HoldoutLock(sealed=sealed, time_seal=TIME_SEAL, sha256=sha)


def is_sealed(name: str, lock: HoldoutLock) -> bool:
    """True when `name` (with or without the USDT suffix, any case) is a sealed coin."""
    n = str(name).strip().upper()
    return n in lock.sealed or (not n.endswith("USDT") and n + "USDT" in lock.sealed)
