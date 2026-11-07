"""Shared artifact-test helpers (HF-32: helpers live here, not in a test module). The ``art`` fixture is registered in
tests/conftest.py."""

import hashlib
import json
import shutil

import pandas as pd
import pytest

from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.data_io import artifact, npy_dir
from nig_timenet_torch.data_io.split_plan import purge_embargo_split, sample_times
from tests.contract_doc import DATASET_DIR

COMMIT = "a" * 40


def sealed_list(*names) -> list[str]:
    """A sealed list of the pinned size: ``names`` first, then filler names."""
    return [*names, *[f"SEALED{i:03d}USDT" for i in range(artifact.EXPECTED_SEALED_COUNT - len(names))]]


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_block(path) -> dict:
    """What tools/export_torch_artifact.py would write for ``path`` (computed from the data, not copied from it)."""
    d = npy_dir.load(path)
    names = sorted(json.loads((path / "manifest.json").read_text())["arrays"])
    files = {f"{k}.npy": _sha(path / f"{k}.npy") for k in names}
    files["meta.pkl"] = _sha(path / "meta.pkl")
    last = pd.Timestamp(int(d["last_candles"][:, TS_COL].astype("int64").max()) + 3600 * 10**9, tz="UTC")
    return {"schema": artifact.SCHEMA, "source_commit": COMMIT, "coins": [b["name"] for b in d["asset_bounds"]],
            "sealed_excluded": sealed_list(), "sealed_found": [],
            "lock_sha256": artifact.EXPECTED_LOCK_SHA256, "time_seal": "2026-10-01T00:00:00+00:00",
            "max_bar_time": last.isoformat(), "files": files,
            "data_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def write_manifest(path, **changes) -> dict:
    man = json.loads((path / "manifest.json").read_text())
    man["export"] = {**export_block(path), **changes}
    for k in [k for k, v in man["export"].items() if v is ...]:
        del man["export"][k]                      # `field=...` removes the field
    (path / "manifest.json").write_text(json.dumps(man))
    return man["export"]


@pytest.fixture()
def art(tmp_path):
    dest = tmp_path / "artifact.dataset"
    shutil.copytree(DATASET_DIR, dest)
    write_manifest(dest)
    return dest


def fixture_plan():
    d = npy_dir.load(DATASET_DIR)
    fs, fe, le = sample_times(d)
    sd = d["split_dates"]
    bar = pd.Timedelta(d["base_tf"])
    embargo = (d["window_sizes"][d["base_tf"]] + d["forecast_horizon"]) * bar
    return purge_embargo_split(le, fs, sd["train_end"], sd["train_end"], sd["val_end"], sd["val_end"], embargo,
                               feature_end=fe)
