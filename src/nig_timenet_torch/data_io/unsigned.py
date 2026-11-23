"""PURPOSE: The manifest of a dataset with no export block (a TF ``.pkl.gz``, a plain ``npy_dir``): DERIVED from the
    data after the checks the data itself allows (no sealed coin, no label bar at or after the time seal).
TAGS: unsigned dataset, derived manifest, pickle, rule d1, sealed coins, time seal, data fingerprint, old project
PITFALLS: provenance is NOT proven (the source commit is unknown, nothing says the file is what the TF pipeline
    wrote): only what the data shows is checked, so the manifest says ``derived=True`` and its fingerprint starts with
    ``unsigned:`` (a run on it never has the identity of a run on a signed artifact); a dataset without the contract
    keys (``asset_bounds``, ``last_candles``, ``base_tf``, ``forecast_horizon``) gets NO manifest (None) and cannot be
    trained on; the sealed list is the packaged, hash-pinned lock, never the file's own claim.

Why (owner, 2026-11-23): a processed dataset of the old project (``crypto_model_latest.pkl.gz``, the same arrays as
its artifact folder) loaded only with ``allow_unchecked=True`` and then failed in ``build`` (``art.manifest`` was None,
the run identity needs ``data_fingerprint``). The flag now loads such a file with the checks that can be made and a
derived manifest, so prepare -> build -> train work, and a file that holds a sealed coin or data after the seal is
still refused.
"""

import numpy as np
import pandas as pd

from nig_timenet_torch.core.label_time import label_bar_open
from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.data_io.artifact import TIME_SEAL, ArtifactError, ArtifactManifest, norm_coin
from nig_timenet_torch.data_io.lock import HoldoutLock, is_sealed, load_lock

__all__ = ["derive_manifest", "UNSIGNED_COMMIT", "UNSIGNED_PREFIX", "CONTRACT_KEYS"]

UNSIGNED_COMMIT = "unsigned: no export block, source commit unknown"
UNSIGNED_PREFIX = "unsigned:"
CONTRACT_KEYS = ("asset_bounds", "last_candles", "base_tf", "forecast_horizon")


def derive_manifest(dataset: dict, fingerprint: str, where="dataset", lock: HoldoutLock | None = None):
    """``ArtifactManifest`` (``derived=True``) of ``dataset``, or None when it lacks ``CONTRACT_KEYS`` (not a dataset).
    ``fingerprint``: a hash of the file (the caller reads it: ``UNSIGNED_PREFIX`` + sha256). Raises ``ArtifactError``
    when a coin of ``asset_bounds`` is sealed, two coins share a name, or a window's label bar is at or after the
    time seal."""
    if any(k not in dataset for k in CONTRACT_KEYS):
        return None
    lock = lock or load_lock()
    coins = tuple(b["name"] for b in dataset["asset_bounds"])
    if not coins:
        raise ArtifactError(f"{where}: asset_bounds is empty")
    if len({norm_coin(c) for c in coins}) != len(coins):
        raise ArtifactError(f"{where}: two coins of asset_bounds share a name")
    leaked = [c for c in coins if is_sealed(c, lock)]
    if leaked:
        raise ArtifactError(f"{where}: coins {leaked[:5]} are on the sealed list of the holdout lock (rule D1): a "
                            "sealed coin must never be in a training dataset; rebuild the file without them")
    bar = pd.Timedelta(dataset["base_tf"])
    ts = np.asarray(dataset["last_candles"])[:, TS_COL].astype("int64")
    latest = pd.Timestamp(int(label_bar_open(ts.max(), dataset["forecast_horizon"], int(bar.value))), tz="UTC")
    if latest >= TIME_SEAL:
        raise ArtifactError(f"{where}: a window's label bar ({latest.isoformat()}) is at or after the time seal "
                            f"{TIME_SEAL.isoformat()} (rule D1)")
    tfs = dataset.get("timeframes")
    return ArtifactManifest(source_commit=UNSIGNED_COMMIT, coins=coins, sealed_excluded=lock.sealed,
                            time_seal=TIME_SEAL, max_bar_time=latest, data_fingerprint=fingerprint, files=None,
                            base_tf=dataset["base_tf"] if tfs else None,
                            timeframes=tuple(str(t) for t in tfs) if tfs else None, derived=True)
