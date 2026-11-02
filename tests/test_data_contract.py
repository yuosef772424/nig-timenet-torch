"""P01: the committed fixture and split dump match docs/DATA_ARTIFACT_CONTRACT.md's dict literal."""

import ast
import json
import pickle
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
DATASET_DIR = FIX / "tiny_dataset.dataset"


def contract():
    text = (ROOT / "docs" / "DATA_ARTIFACT_CONTRACT.md").read_text(encoding="utf-8")
    block = re.search(r"```python\n(CONTRACT = .*?)\n```", text, re.S).group(1)
    return ast.literal_eval(block.split("=", 1)[1])


def ndim(meaning):
    """Rank declared at the start of a shape_meaning, e.g. '(N, T=16, F=18): ...' -> 3, '(): ...' -> 0."""
    inside = re.match(r"\(([^)]*)\)", meaning).group(1).strip().rstrip(",")
    return 0 if not inside else len(inside.split(","))


def check_split(c, dump, path):
    assert set(dump) == set(c), f"{path}: keys {sorted(dump)} != contract {sorted(c)}"
    for k, spec in c.items():
        if spec["dtype"] == "dict":
            check_split(spec["items"], dump[k], f"{path}/{k}")
        else:
            assert dump[k]["dtype"] == spec["dtype"], f"{path}/{k}: dtype"
            assert len(dump[k]["shape"]) == ndim(spec["shape_meaning"]), f"{path}/{k}: rank"


@pytest.mark.xfail(raises=ModuleNotFoundError, strict=True, reason="data_io.npy_dir lands in P06")
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
