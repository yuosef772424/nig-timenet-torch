"""Name resolution (base name -> _latest, full name as written), formats, mmap, local copy, artifact checks.

Ported from feat/ar-stream 2de2846; the branch built its datasets with the native pipeline (I2, not ported yet), so
here they are copies of the committed 6-coin fixture with an ``export`` block (``tests.data_io.test_artifact``)."""

import gzip
import json
import os
import pickle
import shutil

import numpy as np
import pytest

from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.data_io.artifact import ArtifactError
from nig_timenet_torch.data_io.loading import LoaderSettings, load_dataset, resolve_dataset_path
from tests.contract_doc import DATASET_DIR
from tests.data_io.helpers import COMMIT, write_manifest


def _pkl(path, data):
    with gzip.open(path, "wb") as f:
        pickle.dump(data, f)


def _artifact(dest, commit=COMMIT):
    """A copy of the fixture at ``dest`` (replaced if present) with a valid ``export`` block."""
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(DATASET_DIR, dest)
    write_manifest(dest, source_commit=commit)
    return dest


@pytest.fixture(scope="module")
def dataset():
    return npy_dir.load(DATASET_DIR)


def test_a_base_name_gets_latest_added_and_a_full_name_is_read_as_written(tmp_path):
    _pkl(tmp_path / "preprocessing_output_20261027_latest.pkl.gz", {"tag": "latest"})
    _pkl(tmp_path / "preprocessing_outputvjd_284027.pkl.gz", {"tag": "full"})
    for name in ("preprocessing_output_20261027", "preprocessing_output_20261027_latest",
                 "preprocessing_output_20261027_latest.pkl.gz"):
        assert load_dataset(LoaderSettings(str(tmp_path), name, allow_unchecked=True)).dataset == {"tag": "latest"}
    got = load_dataset(LoaderSettings(str(tmp_path), "preprocessing_outputvjd_284027.pkl.gz", allow_unchecked=True))
    assert got.dataset == {"tag": "full"} and got.format == "pkl.gz"
    absolute = resolve_dataset_path(".", str(tmp_path / "preprocessing_outputvjd_284027.pkl.gz"))
    assert absolute.name.endswith("284027.pkl.gz")


def test_a_missing_name_lists_what_was_tried(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"preprocessing_output_1_latest\.dataset.*full file name"):
        resolve_dataset_path(tmp_path, "preprocessing_output_1")
    with pytest.raises(FileNotFoundError, match="read as written"):
        resolve_dataset_path(tmp_path, "nothing.pkl.gz")


def test_folder_and_file_side_by_side_the_newer_wins_and_format_can_force(tmp_path):
    _artifact(tmp_path / "preprocessing_output_latest.dataset")
    _pkl(tmp_path / "preprocessing_output_latest.pkl.gz", {"tag": "pkl"})
    folder = tmp_path / "preprocessing_output_latest.dataset" / "manifest.json"
    os.utime(folder, (1, 1))
    assert resolve_dataset_path(tmp_path, "preprocessing_output").name.endswith(".pkl.gz")
    os.utime(tmp_path / "preprocessing_output_latest.pkl.gz", (1, 1))
    os.utime(folder, (5, 5))
    assert resolve_dataset_path(tmp_path, "preprocessing_output").name.endswith(".dataset")
    assert resolve_dataset_path(tmp_path, "preprocessing_output", "pkl.gz").name.endswith(".pkl.gz")


def test_artifact_folder_is_checked_memory_mapped_and_hash_verified(tmp_path, dataset):
    _artifact(tmp_path / "preprocessing_output_20261027_latest.dataset")
    got = load_dataset(LoaderSettings(str(tmp_path), "preprocessing_output_20261027", mmap=True, verify_hashes=True))
    assert got.format == "npy_dir" and got.manifest is not None and got.manifest.source_commit == COMMIT
    np.testing.assert_array_equal(got.dataset["X_1h"], dataset["X_1h"])
    eager = load_dataset(LoaderSettings(str(tmp_path), "preprocessing_output_20261027", mmap=False))
    assert not isinstance(eager.dataset["X_1h"], np.memmap)


def test_mmap_keeps_big_arrays_on_disk_and_small_ones_in_memory(dataset):
    lazy = npy_dir.load(DATASET_DIR, mmap=True, small_bytes=100_000)
    assert isinstance(lazy["X_1h"], np.memmap) and not isinstance(lazy["y_close_reg"], np.memmap)
    assert not isinstance(npy_dir.load(DATASET_DIR)["X_1h"], np.memmap)
    np.testing.assert_array_equal(lazy["X_1h"], dataset["X_1h"])


def test_local_copy_is_made_reused_and_refreshed(tmp_path):
    _artifact(tmp_path / "src")
    dst = tmp_path / "disk" / "copy"
    got = load_dataset(LoaderSettings(str(tmp_path), "src", local_dir=str(dst)))
    assert got.path == dst and (dst / "manifest.json").is_file()
    stamp = (dst / "X_1h.npy").stat().st_mtime_ns
    load_dataset(LoaderSettings(str(tmp_path), "src", local_dir=str(dst)))
    assert (dst / "X_1h.npy").stat().st_mtime_ns == stamp                      # matching copy reused
    _artifact(tmp_path / "src", commit="12" * 20)
    assert load_dataset(LoaderSettings(str(tmp_path), "src", local_dir=str(dst))).manifest.source_commit == "12" * 20
    assert load_dataset(LoaderSettings(str(tmp_path), "src", local_dir="")).path == tmp_path / "src"
    assert npy_dir.localize(tmp_path / "src", tmp_path / "src") == tmp_path / "src"


def test_unchecked_sources_are_refused_by_default_rule_d1(tmp_path):
    """The branch loaded a plain npy_dir and any pickle without a single D1 check; now that needs an explicit flag."""
    assert LoaderSettings().allow_unchecked is False
    shutil.copytree(DATASET_DIR, tmp_path / "plain")
    with pytest.raises(ArtifactError, match="no 'export' block"):
        load_dataset(LoaderSettings(str(tmp_path), "plain"))
    _pkl(tmp_path / "old_latest.pkl.gz", {"tag": "old"})
    with pytest.raises(ArtifactError, match="rule D1.*allow_unchecked=True"):
        load_dataset(LoaderSettings(str(tmp_path), "old"))
    with pytest.raises(ArtifactError, match="rule D1"):
        load_dataset(LoaderSettings(str(tmp_path), "old_latest.pkl.gz"))
    # explicit opt-out: loads, and says so (no manifest)
    plain = load_dataset(LoaderSettings(str(tmp_path), "plain", allow_unchecked=True))
    assert plain.manifest is None and plain.format == "npy_dir"
    old = load_dataset(LoaderSettings(str(tmp_path), "old", allow_unchecked=True))
    assert old.manifest is None and old.dataset == {"tag": "old"}
    # a checked source carries its manifest, with or without the flag
    _artifact(tmp_path / "good")
    for flag in (False, True):
        got = load_dataset(LoaderSettings(str(tmp_path), "good", allow_unchecked=flag))
        assert got.manifest.source_commit == COMMIT


def test_an_invalid_export_block_is_refused_even_with_allow_unchecked(tmp_path):
    _artifact(tmp_path / "bad")
    write_manifest(tmp_path / "bad", coins=["ZZZ"])
    for flag in (False, True):
        with pytest.raises(ArtifactError, match="differ from the dataset"):
            load_dataset(LoaderSettings(str(tmp_path), "bad", allow_unchecked=flag))


def test_a_plain_npy_dir_without_export_block_and_the_failure_modes(tmp_path):
    shutil.copytree(DATASET_DIR, tmp_path / "plain")
    with pytest.raises(ValueError, match="verify_hashes needs"):
        load_dataset(LoaderSettings(str(tmp_path), "plain", verify_hashes=True, allow_unchecked=True))
    _pkl(tmp_path / "old_latest.pkl.gz", {"tag": "old"})
    with pytest.raises(ValueError, match="verify_hashes needs"):
        load_dataset(LoaderSettings(str(tmp_path), "old", verify_hashes=True, allow_unchecked=True))
    with pytest.raises(ValueError, match="format must be"):
        LoaderSettings(format="zip")
    _pkl(tmp_path / "list_latest.pkl.gz", [1, 2])
    with pytest.raises(ValueError, match="expected a dataset dict"):
        load_dataset(LoaderSettings(str(tmp_path), "list", allow_unchecked=True))
    _artifact(tmp_path / "b")
    np.save(tmp_path / "b" / "y_close_reg.npy", np.zeros(3, "float32"))
    with pytest.raises(ValueError):
        load_dataset(LoaderSettings(str(tmp_path), "b"))
    json.loads((tmp_path / "b" / "manifest.json").read_text())  # still a readable manifest
