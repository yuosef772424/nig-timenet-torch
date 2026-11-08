"""E2 (global label permutation) and E3 (training-era slice of the real artifact) test environments (ROADMAP G1)."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.core.schema import LAST_COLUMNS
from nig_timenet_torch.data.artifact_writer import write_artifact
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.data_io.environments import input_keys, label_keys, permute_labels, real_slice
from nig_timenet_torch.data_io.loading import LoaderSettings, load_dataset
from nig_timenet_torch.data_io.split_plan import sample_times
from nig_timenet_torch.data_io.synthetic import ORACLE_PREFIX
from nig_timenet_torch.workflow import run_common
from nig_timenet_torch.workflow.settings import SplitSettings
from tests.contract_doc import DATASET_DIR
from tests.data_io import envs

SHA = "c" * 40
INPUT_COLUMNS = [LAST_COLUMNS.index(c) for c in ("last_high", "last_low", "last_close", "timestamp")]
#: local path of the real artifact (never in CI): set NIG_REAL_ARTIFACT to its folder to run the E3 check on it
REAL = os.environ.get("NIG_REAL_ARTIFACT", "")


@pytest.fixture(scope="module")
def e1():
    return envs.small(n_coins=24, n_bars=2400)          # 400 days of 4h bars


def _fixture():
    return npy_dir.load(DATASET_DIR)


@pytest.mark.parametrize("source", ["e1", "fixture"])
def test_e2_inputs_byte_identical_labels_permuted(source, request):
    ds = request.getfixturevalue("e1") if source == "e1" else _fixture()
    e2 = permute_labels(ds, seed=3)
    for k in input_keys(ds):
        assert np.asarray(e2[k]).tobytes() == np.asarray(ds[k]).tobytes(), k
    lc0, lc2 = np.asarray(ds["last_candles"]), np.asarray(e2["last_candles"])
    assert lc2[:, INPUT_COLUMNS].tobytes() == lc0[:, INPUT_COLUMNS].tobytes()
    n = len(lc0)
    names = envs.assets(ds)
    reg = label_keys(ds)[0]
    perm = np.random.default_rng(3).permutation(n)                # the documented permutation of seed 3
    for k in label_keys(ds):
        a, b = np.asarray(ds[k]), np.asarray(e2[k])
        assert np.array_equal(np.sort(a), np.sort(b)) and not np.array_equal(a, b), k
        np.testing.assert_array_equal(b, a[perm])                  # ONE permutation for every label
    assert np.mean(perm == np.arange(n)) < 0.01
    assert np.mean(names[perm] == names) < 2.0 / len(ds["asset_bounds"])        # across coins
    assert abs(np.corrcoef(np.asarray(ds[reg]), np.asarray(e2[reg]))[0, 1]) < 4 / np.sqrt(n)
    run_common.check_decode(e2, run_common.price_targets(e2))         # future columns follow the labels
    assert not any(k.startswith(ORACLE_PREFIX) for k in e2)
    assert e2["environment"]["name"] == "E2" and e2["environment"]["seed"] == 3
    assert permute_labels(ds, seed=3)[reg].tobytes() == e2[reg].tobytes()


def test_e2_of_e1_round_trips_as_a_contract_artifact(e1, tmp_path):
    write_artifact(permute_labels(e1), tmp_path / "e2.dataset", source_commit=SHA)
    ds = load_dataset(LoaderSettings(directory=str(tmp_path), name="e2.dataset", local_dir="")).dataset
    run_common.check_decode(ds, run_common.price_targets(ds))
    assert ds["environment"]["dropped"] and np.array_equal(ds["X_4h"], e1["X_4h"])


def _check_e3(full: dict, e3: dict, train_end: pd.Timestamp, n_coins: int):
    _, _, le = sample_times(e3)
    assert le.max() <= train_end.value                               # never past the full run's train end
    assert len(e3["asset_bounds"]) == n_coins == len(e3["environment"]["coins"])
    fs = sample_times(e3)[0]
    assert fs.min() >= (train_end - pd.DateOffset(months=e3["environment"]["months"])).value
    bounds = e3["asset_bounds"]
    assert bounds[0]["start"] == 0 and bounds[-1]["end"] == len(e3["last_candles"])
    assert all(a["end"] == b["start"] for a, b in zip(bounds, bounds[1:], strict=False))
    assert set(e3["environment"]["coins"]) <= {b["name"] for b in full["asset_bounds"]}
    (tr, va, te), (t_end, v_end) = run_common.make_plan(e3, SplitSettings())
    assert (t_end, v_end) == (e3["split_dates"]["train_end"], e3["split_dates"]["val_end"])
    assert len(tr) and len(va) and len(te)
    assert 0.6 < len(tr) / len(e3["last_candles"]) < 0.72                # 70 / 15 / 15, purged
    assert pd.Timestamp(v_end) < train_end


def test_e3_slice_of_e1_stays_inside_the_full_training_era(e1, tmp_path):
    train_end, _ = run_common.split_bounds(e1, SplitSettings())         # the full run's own split
    e3 = real_slice(e1, train_end=train_end)
    _check_e3(e1, e3, train_end, 20)
    full_train = set(run_common.make_plan(e1, SplitSettings())[0][0].tolist())
    lc = np.asarray(e1["last_candles"])
    index = {k: i for i, k in enumerate(zip(envs.assets(e1), lc[:, 3], strict=True))}
    rows = set(zip(envs.assets(e3), np.asarray(e3["last_candles"])[:, 3], strict=True))
    assert rows <= set(index) and {index[r] for r in rows} <= full_train      # all are full-run TRAIN rows
    write_artifact(e3, tmp_path / "e3.dataset", source_commit=SHA)
    ds = load_dataset(LoaderSettings(directory=str(tmp_path), name="e3.dataset", local_dir="")).dataset
    assert ds["split_dates"] == e3["split_dates"] and ds["environment"]["name"] == "E3"
    e2 = permute_labels(ds)                                             # the protocol's E2: shuffled E3
    assert np.array_equal(e2["X_4h"], ds["X_4h"])


def test_e3_on_the_fixture_and_its_refusals():
    ds = _fixture()
    train_end = pd.Timestamp(run_common.split_bounds(ds, SplitSettings())[0])
    train_end = train_end.tz_localize("UTC") if train_end.tzinfo is None else train_end
    e3 = real_slice(ds, train_end=train_end, n_coins=3, months=1)
    assert sample_times(e3)[2].max() <= train_end.value and len(e3["asset_bounds"]) == 3
    with pytest.raises(ValueError, match="coins cover"):
        real_slice(ds, train_end=train_end, n_coins=7, months=1)
    with pytest.raises(ValueError):
        real_slice(ds, train_end=train_end, train_share=0.9, val_share=0.2)


@pytest.mark.skipif(not REAL or not (Path(REAL) / "manifest.json").is_file(),
                    reason="the real artifact is not on this machine (set NIG_REAL_ARTIFACT to its folder; CI has "
                           "none)")
def test_e3_on_the_real_artifact():
    path = Path(REAL)
    full = load_dataset(LoaderSettings(directory=str(path.parent), name=path.name, local_dir="")).dataset
    train_end, _ = run_common.split_bounds(full, SplitSettings())
    e3 = real_slice(full, train_end=train_end)
    _check_e3(full, e3, pd.Timestamp(train_end), 20)
    run_common.check_decode(e3, run_common.price_targets(e3))


def test_nig_envs_command_writes_e1_e2_e3(tmp_path, capsys):
    from nig_timenet_torch.data_io.environments import main

    e1 = tmp_path / "e1.dataset"
    assert main(["e1", str(e1), "--settings", '{"n_coins": 22, "n_bars": 2400, "seed": 5}']) == 0
    assert main(["e2", str(e1), str(tmp_path / "e2.dataset"), "--seed", "1"]) == 0
    assert main(["e3", str(e1), str(tmp_path / "e3.dataset")]) == 0
    assert "wrote e3" in capsys.readouterr().out
    ds = {k: load_dataset(LoaderSettings(name=str(tmp_path / f"{k}.dataset"), local_dir="")).dataset
          for k in ("e1", "e2", "e3")}
    assert ds["e2"]["environment"]["name"] == "E2" and np.array_equal(ds["e2"]["X_4h"], ds["e1"]["X_4h"])
    train_end, _ = run_common.split_bounds(ds["e1"], SplitSettings())
    assert sample_times(ds["e3"])[2].max() <= train_end.value and len(ds["e3"]["asset_bounds"]) == 20
    assert ds["e3"]["environment"]["source_fingerprint"] == ds["e2"]["environment"]["source_fingerprint"]
    with pytest.raises(SystemExit):
        main(["e2", str(e1)])
