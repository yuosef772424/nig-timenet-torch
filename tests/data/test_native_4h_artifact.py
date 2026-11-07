"""HF-15 / I2: the native pipeline BUILDS the real artifact shape (4h, split_dates None); the baseline job trains on it.

Also the writer's seal rule (D1) end to end: the synthetic coins run past the time seal and one coin is on the sealed
list; the artifact on disk holds no sealed coin and no window whose label bar is at or after the seal.
"""

import json

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.data.lock import load_lock
from nig_timenet_torch.data.pipeline import build_artifact
from nig_timenet_torch.data_io.artifact import load_artifact
from nig_timenet_torch.data_io.loading import LoaderSettings, load_dataset
from tests.colab.test_baseline import run_cfg
from tests.data.helpers import COINS, tiny_settings

SHA = "cd" * 20


@pytest.fixture(scope="module")
def native_4h(tmp_path_factory):
    """4h bars, one 4h frame, synthetic coins that run past the 2026-10-01 seal, plus one sealed coin name."""
    out = tmp_path_factory.mktemp("native") / "n.dataset"
    sealed = load_lock().sealed[0]
    s = tiny_settings(**{
        "source.interval": "4h", "source.synthetic_days": 700, "source.coins": (*COINS, sealed),
        "windows.tf_order": ("4h",), "windows.window_sizes": {"4h": 16}, "windows.higher_tf_mode": "legacy",
        "output.path": str(out), "output.source_commit": SHA})
    ds, manifest = build_artifact(s)
    return out, ds, manifest, sealed


def test_native_artifact_is_4h_with_no_split_dates_and_loads_through_the_loader(native_4h):
    out, ds, manifest, _ = native_4h
    art = load_dataset(LoaderSettings(directory=str(out.parent), name=out.name, local_dir=""))
    assert "X_4h" in art.dataset and "X_1h" not in art.dataset
    assert art.dataset["timeframes"] == ["4h"] and art.dataset["base_tf"] == "4h"
    assert art.dataset.get("split_dates") is None
    assert len(art.dataset["last_candles"]) == manifest["export"]["rows"]


def test_baseline_job_trains_on_the_native_4h_artifact(native_4h, tmp_path, monkeypatch):
    out = run_cfg(tmp_path, "native", native_4h[0], monkeypatch, max_rows=100)
    s = json.loads((out / "baseline_summary.json").read_text())
    assert s["input_key"] == "X_4h" and s["targets"] == ["high", "low", "close"]
    assert 0 < s["n_windows"]["train"] <= 100 and 0 < s["n_windows"]["test"] <= 100
    assert (out / "test_report.json").is_file()


def test_the_writer_drops_sealed_coins_and_windows_after_the_time_seal(native_4h):
    out, ds, manifest, sealed = native_4h
    lock = load_lock()
    art = load_artifact(out)
    names = [b["name"] for b in art.dataset["asset_bounds"]]
    assert sealed not in names and set(COINS) <= set(names)
    label_ts = pd.to_datetime(art.dataset["last_candles"][:, TS_COL].astype("int64"), unit="ns", utc=True)
    assert label_ts.max() < lock.time_seal
    # the built dataset crossed the seal (else this test proves nothing): the writer removed exactly those windows
    built_ts = pd.to_datetime(ds["last_candles"][:, TS_COL].astype("int64"), unit="ns", utc=True)
    late = int((built_ts >= lock.time_seal).sum())
    assert late > 0 and manifest["export"]["rows"] == len(ds["last_candles"]) - late
    assert np.isfinite(art.dataset["X_4h"]).all()
