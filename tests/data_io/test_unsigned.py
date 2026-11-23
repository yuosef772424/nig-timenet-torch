"""data_io/unsigned.py + loading.py: a dataset of the old project WITHOUT an export block (a .pkl.gz, a plain npy_dir)
loads with allow_unchecked=True after the checks the data allows, with a derived manifest, and can be trained on."""

import gzip
import pickle
import shutil

import numpy as np
import pytest

from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.data_io.artifact import TIME_SEAL, ArtifactError
from nig_timenet_torch.data_io.loading import LoaderSettings, load_dataset
from nig_timenet_torch.data_io.lock import load_lock
from nig_timenet_torch.data_io.unsigned import UNSIGNED_PREFIX, derive_manifest
from nig_timenet_torch.workflow import run_common
from tests.contract_doc import DATASET_DIR


@pytest.fixture(scope="module")
def dataset():
    return npy_dir.load(DATASET_DIR)


def _pkl(path, data):
    with gzip.open(path, "wb") as f:
        pickle.dump(data, f)
    return path


def test_a_pickle_of_the_old_project_loads_with_the_flag_and_has_a_derived_manifest(dataset, tmp_path):
    f = _pkl(tmp_path / "crypto_model_latest.pkl.gz", dataset)
    got = load_dataset(LoaderSettings(name=str(f), allow_unchecked=True))
    m = got.manifest
    assert got.format == "pkl.gz" and m.derived and list(m.coins) == [b["name"] for b in dataset["asset_bounds"]]
    assert m.data_fingerprint.startswith(UNSIGNED_PREFIX) and len(m.data_fingerprint) == len(UNSIGNED_PREFIX) + 64
    assert "unsigned" in m.source_commit and m.max_bar_time < TIME_SEAL and m.sealed_excluded == load_lock().sealed
    assert m.base_tf == dataset["base_tf"] and list(m.timeframes) == list(dataset["timeframes"])
    again = load_dataset(LoaderSettings(name=str(f), allow_unchecked=True)).manifest
    assert again.data_fingerprint == m.data_fingerprint                     # the same file, the same fingerprint
    other = _pkl(tmp_path / "other.pkl.gz", {**dataset, "stride": 99})
    assert load_dataset(LoaderSettings(name=str(other), allow_unchecked=True)).manifest.data_fingerprint \
        != m.data_fingerprint                                               # another file, another fingerprint
    with pytest.raises(ArtifactError, match="allow_unchecked=True"):         # the default still refuses, and says how
        load_dataset(LoaderSettings(name=str(f)))


def test_a_pickle_with_a_sealed_coin_or_data_after_the_seal_is_refused_even_with_the_flag(dataset, tmp_path):
    sealed = load_lock().sealed[0]
    leaky = {**dataset, "asset_bounds": [{**dataset["asset_bounds"][0], "name": sealed}, *dataset["asset_bounds"][1:]]}
    with pytest.raises(ArtifactError, match="sealed list.*rule D1"):
        load_dataset(LoaderSettings(name=str(_pkl(tmp_path / "a.pkl.gz", leaky)), allow_unchecked=True))
    late = np.array(dataset["last_candles"], copy=True)
    late[-1, TS_COL] = TIME_SEAL.value
    with pytest.raises(ArtifactError, match="time seal"):
        load_dataset(LoaderSettings(name=str(_pkl(tmp_path / "b.pkl.gz", {**dataset, "last_candles": late})),
                                    allow_unchecked=True))
    twice = {**dataset, "asset_bounds": [dataset["asset_bounds"][0], dataset["asset_bounds"][0]]}
    with pytest.raises(ArtifactError, match="share a name"):
        derive_manifest(twice, "unsigned:x")


def test_a_plain_npy_dir_gets_the_same_treatment_and_something_that_is_not_a_dataset_gets_none(dataset, tmp_path):
    shutil.copytree(DATASET_DIR, tmp_path / "plain")
    got = load_dataset(LoaderSettings(name=str(tmp_path / "plain"), allow_unchecked=True))
    assert got.manifest.derived and got.manifest.data_fingerprint.startswith(UNSIGNED_PREFIX)
    assert derive_manifest({"tag": "x"}, "unsigned:x") is None
    bare = load_dataset(LoaderSettings(name=str(_pkl(tmp_path / "bare.pkl.gz", {"tag": "x"})), allow_unchecked=True))
    assert bare.manifest is None


def test_training_on_something_that_is_not_a_dataset_says_why_not_an_attribute_error(dataset, tmp_path):
    bare = load_dataset(LoaderSettings(name=str(_pkl(tmp_path / "bare.pkl.gz", {"tag": "x"})), allow_unchecked=True))
    with pytest.raises(ValueError, match="no manifest.*derived manifest"):
        run_common.run_identity(None, None, None, bare, None)


def test_writing_a_new_artifact_from_an_unsigned_source_is_refused(dataset, tmp_path):
    from nig_timenet_torch.data_io.artifact_writer import write_artifact
    m = derive_manifest(dataset, "unsigned:x")
    with pytest.raises(ValueError, match="40 hex"):
        write_artifact(dataset, tmp_path / "out", source_commit=m.source_commit)
