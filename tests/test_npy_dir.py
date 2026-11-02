"""P06: data_io.npy_dir.load reads the P01 fixture exactly as docs/DATA_ARTIFACT_CONTRACT.md describes."""

import shutil

import numpy as np
import pytest

from nig_timenet_torch.data_io import npy_dir
from tests.contract_doc import DATASET_DIR, contract, ndim


def test_load_matches_contract():
    d = npy_dir.load(DATASET_DIR)
    c = contract()["dataset"]
    assert set(d) == set(c)
    n = d["base_params"].shape[0]
    for k, spec in c.items():
        v = d[k]
        if isinstance(v, np.ndarray):
            assert str(v.dtype) == spec["dtype"], k
            assert v.ndim == ndim(spec["shape_meaning"]), k
            assert v.shape[0] == n, k  # axis 0 = windows, shared by every array
        else:
            assert type(v).__name__ == spec["dtype"].split("[")[0], k
    assert sum(b["end"] - b["start"] for b in d["asset_bounds"]) == n
    assert d["X_1h"].shape[1:] == (d["window_sizes"]["1h"], len(d["feature_order"]))


def test_missing_manifest_raises_file_not_found(tmp_path):
    dst = tmp_path / "partial.dataset"
    shutil.copytree(DATASET_DIR, dst)
    (dst / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        npy_dir.load(dst)
    with pytest.raises(FileNotFoundError):
        npy_dir.load(tmp_path / "does_not_exist")
