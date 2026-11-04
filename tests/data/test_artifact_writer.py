"""write_artifact -> data_io.artifact.load_artifact: the contract the real-data pipeline must satisfy."""

import json
import subprocess

import numpy as np
import pytest

from nig_timenet_torch.data.artifact_writer import git_head, restrict_to_time_seal, write_artifact
from nig_timenet_torch.data.build import build_dataset
from nig_timenet_torch.data.lock import load_lock
from nig_timenet_torch.data.pipeline import build_artifact
from nig_timenet_torch.data_io.artifact import ArtifactError, load_artifact, verify_hashes
from tests.data.helpers import tiny_settings

SHA = "ab" * 20


@pytest.fixture(scope="module")
def dataset():
    return build_dataset(tiny_settings())


def test_written_artifact_loads_and_verifies(dataset, tmp_path):
    manifest = write_artifact(dataset, tmp_path / "a", source_commit=SHA)
    art = load_artifact(tmp_path / "a")
    verify_hashes(tmp_path / "a")
    assert len(art.dataset["last_candles"]) == manifest["export"]["rows"] == len(dataset["last_candles"])
    np.testing.assert_array_equal(art.dataset["X_1h"], dataset["X_1h"])
    assert art.dataset["feature_order"] == dataset["feature_order"]
    assert manifest["export"]["lock_sha256"] == load_lock().sha256 and len(manifest["export"]["sealed_excluded"]) == 183


def test_existing_folder_needs_overwrite_and_a_bad_sha_is_refused(dataset, tmp_path):
    write_artifact(dataset, tmp_path / "a", source_commit=SHA)
    with pytest.raises(FileExistsError, match="overwrite"):
        write_artifact(dataset, tmp_path / "a", source_commit=SHA)
    write_artifact(dataset, tmp_path / "a", source_commit=SHA, overwrite=True)
    with pytest.raises(ValueError, match="40 hex"):
        write_artifact(dataset, tmp_path / "b", source_commit="abc")
    assert not (tmp_path / "b").exists() and not (tmp_path / "b.tmp").exists()


def test_a_corrupted_array_is_caught_by_the_hash_check(dataset, tmp_path):
    write_artifact(dataset, tmp_path / "a", source_commit=SHA)
    arr = np.load(tmp_path / "a" / "y_close_reg.npy")
    arr[0] += 1.0
    np.save(tmp_path / "a" / "y_close_reg.npy", arr)
    with pytest.raises(ArtifactError):
        verify_hashes(tmp_path / "a")


def test_windows_at_or_after_the_time_seal_and_sealed_coins_are_dropped(dataset):
    lock = load_lock()
    late = {**dataset, "last_candles": dataset["last_candles"].copy()}
    seal_ns = lock.time_seal.value
    late["last_candles"][-5:, 3] = seal_ns                    # label bar = seal + 1h: sealed
    out = restrict_to_time_seal(late, lock)
    assert len(out["last_candles"]) == len(dataset["last_candles"]) - 5
    assert out["asset_bounds"][-1]["end"] == len(out["last_candles"])
    sealed_coin = {**dataset, "asset_bounds": [{**dataset["asset_bounds"][0], "name": lock.sealed[0]},
                                              *dataset["asset_bounds"][1:]]}
    dropped = restrict_to_time_seal(sealed_coin, lock)
    assert len(dropped["asset_bounds"]) == len(dataset["asset_bounds"]) - 1 and dropped["asset_bounds"][0]["start"] == 0


def test_git_head_of_a_checkout_and_the_failure_without_one(tmp_path):
    with pytest.raises(RuntimeError, match="source_commit"):
        git_head(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
                    "--allow-empty", "-m", "x"], check=True)
    assert len(git_head(tmp_path)) == 40


def test_build_artifact_writes_when_a_path_is_set(tmp_path):
    s = tiny_settings(**{"output.path": str(tmp_path / "out"), "output.source_commit": SHA,
                         "output.producer": "unit-test"})
    ds, manifest = build_artifact(s)
    assert manifest["export"]["producer"] == "unit-test" and len(ds["last_candles"]) == manifest["export"]["rows"]
    assert json.loads((tmp_path / "out" / "manifest.json").read_text())["export"]["source_commit"] == SHA
    assert build_artifact(tiny_settings())[1] is None


def test_named_output_writes_a_dated_folder_and_latest_that_the_loader_finds(dataset, tmp_path):
    from nig_timenet_torch.data_io.loading import LoaderSettings, load_dataset
    out = {"output.directory": str(tmp_path), "output.source_commit": SHA, "output.stamp": "20261027_153000"}
    _, manifest = build_artifact(tiny_settings(**out))
    assert (tmp_path / "preprocessing_output_20261027_153000.dataset" / "manifest.json").exists()
    assert (tmp_path / "preprocessing_output_latest.dataset" / "manifest.json").exists()
    by_base = load_dataset(LoaderSettings(directory=str(tmp_path), local_dir=""))
    by_full = load_dataset(LoaderSettings(directory=str(tmp_path), name="preprocessing_output_20261027_153000.dataset",
                                          local_dir=""))
    assert by_base.path.name == "preprocessing_output_latest.dataset" and by_full.path.name.endswith("153000.dataset")
    assert len(by_base.dataset["last_candles"]) == manifest["export"]["rows"]
    # a second build with another stamp moves _latest, keeps the first dated copy; latest=False leaves _latest alone
    build_artifact(tiny_settings(**{**out, "output.stamp": "20261028_000000"}))
    build_artifact(tiny_settings(**{**out, "output.stamp": "20261029_000000", "output.latest": False}))
    assert len(list(tmp_path.glob("preprocessing_output_2026*.dataset"))) == 3
    latest = json.loads((tmp_path / "preprocessing_output_latest.dataset" / "manifest.json").read_text())
    second = json.loads((tmp_path / "preprocessing_output_20261028_000000.dataset" / "manifest.json").read_text())
    assert latest["id"] == second["id"]
