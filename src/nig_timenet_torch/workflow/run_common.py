"""Steps every run shares, flat or panel: load, retarget, split, device, identity.

Each is a small function of the settings so a notebook can call them one by one (and look at the result) or call
``workflow.run.run`` once.
"""

import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from nig_timenet_torch.colab.identity import identity_for_run
from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.data.targets import retarget
from nig_timenet_torch.data_io.loading import LoadedData, load_dataset
from nig_timenet_torch.data_io.split_plan import SplitPlan, purge_embargo_split, sample_times
from nig_timenet_torch.trainer.identity import RunIdentity
from nig_timenet_torch.workflow.settings import RunSettings

__all__ = ["load_data", "apply_targets", "split_bounds", "make_plan", "pick_device", "code_sha", "data_fingerprint",
           "run_identity"]


def load_data(settings: RunSettings) -> LoadedData:
    """The dataset ``settings.data`` names (see data_io/loading.py), as stored; retarget it after the split is known
    (``apply_targets`` with the train rows), so a scale fitted from the data never sees validation or test rows."""
    return load_dataset(settings.data)


def apply_targets(dataset: dict, settings: RunSettings, train_idx: np.ndarray | None = None) -> tuple[dict, dict]:
    """The dataset under ``settings.targets`` (and the report of the change); unchanged when no mode is set. The
    scale of a data-dependent mode is fitted on ``train_idx`` (default: every row; pass the train rows)."""
    t = settings.targets
    if not t.mode:
        return dataset, {}
    return retarget(dataset, t.mode, train_idx=train_idx, clip=t.clip or None, center=t.center, group_freq=t.group_freq)


def split_bounds(dataset: dict, settings: RunSettings) -> tuple:
    """(train_end, val_end): the settings' instants, else the artifact's ``split_dates``, else share quantiles."""
    s = settings.split
    if s.train_end and s.val_end:
        return pd.Timestamp(s.train_end), pd.Timestamp(s.val_end)
    sd = dataset.get("split_dates")
    if sd:
        return sd["train_end"], sd["val_end"]
    ts = np.sort(np.asarray(dataset["last_candles"])[:, TS_COL].astype(np.int64))
    q = [ts[min(len(ts) - 1, int(len(ts) * f))] for f in (s.train_share, s.train_share + s.val_share)]
    return tuple(pd.Timestamp(int(v), tz="UTC") for v in q)


def make_plan(dataset: dict, settings: RunSettings) -> SplitPlan:
    fs, fe, le = sample_times(dataset)
    train_end, val_end = split_bounds(dataset, settings)
    return purge_embargo_split(le, fs, train_end, train_end, val_end, val_end,
                               pd.Timedelta(hours=settings.split.embargo_hours), feature_end=fe)


def pick_device(settings: RunSettings) -> torch.device:
    d = settings.train.device
    return torch.device("cuda" if d == "auto" and torch.cuda.is_available() else "cpu" if d == "auto" else d)


def code_sha(settings: RunSettings) -> str:
    if settings.code_sha:
        return settings.code_sha
    try:
        root = Path(__file__).resolve().parent
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def data_fingerprint(loaded: LoadedData, dataset: dict) -> str:
    """The manifest's fingerprint for an artifact; otherwise a hash of the window times and shapes."""
    if loaded.manifest is not None:
        return loaded.manifest.data_fingerprint
    h = hashlib.sha256(np.ascontiguousarray(np.asarray(dataset["last_candles"])[:, TS_COL]).tobytes())
    for k in sorted(k for k in dataset if k.startswith("X_")):
        h.update(f"{k}{np.asarray(dataset[k]).shape}".encode())
    return h.hexdigest()


def run_identity(trainer, fit_settings, model_settings, loaded: LoadedData, dataset: dict, settings: RunSettings
                 ) -> RunIdentity:
    """The identity of the run the built ``trainer`` is about to fit (colab/identity.py is the one builder)."""
    stub = SimpleNamespace(data_fingerprint=data_fingerprint(loaded, dataset))
    return identity_for_run(trainer, fit_settings, model_settings, stub, code_sha(settings))
