"""P01: the committed fixture and split dump match docs/DATA_ARTIFACT_CONTRACT.md's dict literal."""

import json
import pickle

from tests.contract_doc import DATASET_DIR, FIXTURES, contract, ndim

FIX = FIXTURES


def check_split(c, dump, path):
    assert set(dump) == set(c), f"{path}: keys {sorted(dump)} != contract {sorted(c)}"
    for k, spec in c.items():
        if spec["dtype"] == "dict":
            check_split(spec["items"], dump[k], f"{path}/{k}")
        else:
            assert dump[k]["dtype"] == spec["dtype"], f"{path}/{k}: dtype"
            assert len(dump[k]["shape"]) == ndim(spec["shape_meaning"]), f"{path}/{k}: rank"


def test_fixture_loads():
    from nig_timenet_torch.data_io import npy_dir

    d = npy_dir.load(DATASET_DIR)
    assert set(d) == set(contract()["dataset"])


def test_manifest_matches_doc():
    man = json.loads((DATASET_DIR / "manifest.json").read_text())
    c = contract()["dataset"]
    assert man["format"] == "npy_dir" and man["version"] == 1
    assert set(man["key_order"]) == set(c)
    for k, a in man["arrays"].items():
        assert a["dtype"] == c[k]["dtype"], k
        assert len(a["shape"]) == ndim(c[k]["shape_meaning"]), k
    with open(DATASET_DIR / "meta.pkl", "rb") as f:
        meta = pickle.load(f)  # fixture produced by the upstream pipeline, committed in this repo
    assert set(meta) | set(man["arrays"]) == set(c)


def test_split_schema_matches_doc():
    dump = json.loads((FIX / "tiny_split.json").read_text())
    c = contract()
    check_split(c["train"], dump["train"], "train")
    check_split(c["val"], dump["val"], "val")
    with open(DATASET_DIR / "meta.pkl", "rb") as f:
        assets = [b["name"] for b in pickle.load(f)["asset_bounds"]]
    assert set(dump["test"]) == set(assets)
    for a in assets:
        check_split(c["test"]["<asset>"]["items"], dump["test"][a], f"test/{a}")
