"""Name resolution (base name -> _latest, full name as written), formats, mmap, local copy, artifact checks."""

import gzip
import os
import pickle

import numpy as np
import pytest

from nig_timenet_torch.data.artifact_writer import write_artifact
from nig_timenet_torch.data.build import build_dataset
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.data_io.artifact import ArtifactError
from nig_timenet_torch.data_io.loading import LoaderSettings, load_dataset, resolve_dataset_path
from tests.data.helpers import tiny_settings

SHA = "ef" * 20


@pytest.fixture(scope="module")
def dataset():
    return build_dataset(tiny_settings())


def _pkl(path, data):
    with gzip.open(path, "wb") as f:
        pickle.dump(data, f)


def test_a_base_name_gets_latest_added_and_a_full_name_is_read_as_written(tmp_path, dataset):
    _pkl(tmp_path / "preprocessing_output_20261027_latest.pkl.gz", {"tag": "latest"})
    _pkl(tmp_path / "preprocessing_outputvjd_284027.pkl.gz", {"tag": "full"})
    for name in ("preprocessing_output_20261027", "preprocessing_output_20261027_latest",
                 "preprocessing_output_20261027_latest.pkl.gz"):
        assert load_dataset(LoaderSettings(str(tmp_path), name)).dataset == {"tag": "latest"}
    got = load_dataset(LoaderSettings(str(tmp_path), "preprocessing_outputvjd_284027.pkl.gz"))
    assert got.dataset == {"tag": "full"} and got.format == "pkl.gz"
    absolute = resolve_dataset_path(".", str(tmp_path / "preprocessing_outputvjd_284027.pkl.gz"))
    assert absolute.name.endswith("284027.pkl.gz")


def test_a_missing_name_lists_what_was_tried(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"preprocessing_output_1_latest\.dataset.*full file name"):
        resolve_dataset_path(tmp_path, "preprocessing_output_1")
    with pytest.raises(FileNotFoundError, match="read as written"):
        resolve_dataset_path(tmp_path, "nothing.pkl.gz")


def test_folder_and_file_side_by_side_the_newer_wins_and_format_can_force(tmp_path, dataset):
    write_artifact(dataset, tmp_path / "preprocessing_output_latest.dataset", source_commit=SHA)
    _pkl(tmp_path / "preprocessing_output_latest.pkl.gz", {"tag": "pkl"})
    folder = tmp_path / "preprocessing_output_latest.dataset" / "manifest.json"
    os.utime(folder, (1, 1))
    assert resolve_dataset_path(tmp_path, "preprocessing_output").name.endswith(".pkl.gz")
    os.utime(tmp_path / "preprocessing_output_latest.pkl.gz", (1, 1))
    os.utime(folder, (5, 5))
    assert resolve_dataset_path(tmp_path, "preprocessing_output").name.endswith(".dataset")
    assert resolve_dataset_path(tmp_path, "preprocessing_output", "pkl.gz").name.endswith(".pkl.gz")


def test_artifact_folder_is_checked_memory_mapped_and_hash_verified(tmp_path, dataset):
    write_artifact(dataset, tmp_path / "preprocessing_output_20261027_latest.dataset", source_commit=SHA)
    got = load_dataset(LoaderSettings(str(tmp_path), "preprocessing_output_20261027", mmap=True, verify_hashes=True))
    assert got.format == "npy_dir" and got.manifest is not None and got.manifest.source_commit == SHA
    np.testing.assert_array_equal(got.dataset["X_1h"], dataset["X_1h"])
    eager = load_dataset(LoaderSettings(str(tmp_path), "preprocessing_output_20261027", mmap=False))
    assert not isinstance(eager.dataset["X_1h"], np.memmap)


def test_mmap_keeps_big_arrays_on_disk_and_small_ones_in_memory(tmp_path, dataset):
    write_artifact(dataset, tmp_path / "a", source_commit=SHA)
    lazy = npy_dir.load(tmp_path / "a", mmap=True, small_bytes=100_000)
    assert isinstance(lazy["X_1h"], np.memmap) and not isinstance(lazy["y_close_reg"], np.memmap)
    assert not isinstance(npy_dir.load(tmp_path / "a")["X_1h"], np.memmap)
    np.testing.assert_array_equal(lazy["X_1h"], dataset["X_1h"])


def test_local_copy_is_made_reused_and_refreshed(tmp_path, dataset):
    write_artifact(dataset, tmp_path / "src", source_commit=SHA)
    dst = tmp_path / "disk" / "copy"
    got = load_dataset(LoaderSettings(str(tmp_path), "src", local_dir=str(dst)))
    assert got.path == dst and (dst / "manifest.json").is_file()
    stamp = (dst / "X_1h.npy").stat().st_mtime_ns
    load_dataset(LoaderSettings(str(tmp_path), "src", local_dir=str(dst)))
    assert (dst / "X_1h.npy").stat().st_mtime_ns == stamp                      # matching copy reused
    write_artifact(dataset, tmp_path / "src", source_commit="12" * 20, overwrite=True)
    assert load_dataset(LoaderSettings(str(tmp_path), "src", local_dir=str(dst))).manifest.source_commit == "12" * 20
    assert load_dataset(LoaderSettings(str(tmp_path), "src", local_dir="")).path == tmp_path / "src"
    assert npy_dir.localize(tmp_path / "src", tmp_path / "src") == tmp_path / "src"


def test_a_plain_npy_dir_without_export_block_and_the_failure_modes(tmp_path, dataset):
    plain = {k: dataset[k] for k in ("X_1h", "last_candles")}
    npy_dir_path = tmp_path / "plain"
    write_artifact(dataset, npy_dir_path, source_commit=SHA)
    import json
    m = json.loads((npy_dir_path / "manifest.json").read_text())
    m.pop("export")
    (npy_dir_path / "manifest.json").write_text(json.dumps(m))
    assert set(load_dataset(LoaderSettings(str(tmp_path), "plain")).dataset) >= set(plain)
    with pytest.raises(ValueError, match="verify_hashes needs"):
        load_dataset(LoaderSettings(str(tmp_path), "plain", verify_hashes=True))
    with pytest.raises(ValueError, match="format must be"):
        LoaderSettings(format="zip")
    _pkl(tmp_path / "list_latest.pkl.gz", [1, 2])
    with pytest.raises(ValueError, match="expected a dataset dict"):
        load_dataset(LoaderSettings(str(tmp_path), "list"))
    write_artifact(dataset, tmp_path / "b", source_commit=SHA)
    np.save(tmp_path / "b" / "y_close_reg.npy", np.zeros(3, "float32"))
    with pytest.raises(ValueError):
        load_dataset(LoaderSettings(str(tmp_path), "b"))
    assert ArtifactError  # imported for the corrupt-hash path covered in tests/data/test_artifact_writer.py
