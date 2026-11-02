"""P45: execute the Colab notebook's code cells headless on CPU (fixture data, 1 epoch). Cells tagged ``colab``
(clone, pip install, Drive mount) are skipped; the notebook's own pytest cell is disabled (no recursion)."""

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "notebooks" / "colab_nig_timenet_torch.ipynb"


def _cells():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return nb, [c for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_is_clean():
    nb, code = _cells()
    assert nb["nbformat"] == 4 and code
    for c in code:
        assert c["outputs"] == [] and c["execution_count"] is None, "commit the notebook without outputs"
        src = "".join(c["source"])
        assert "ghp_" not in src and "github_pat_" not in src, "no tokens in the notebook"
        if "colab" not in c["metadata"].get("tags", []):
            assert not any(line.lstrip().startswith(("!", "%")) for line in src.splitlines()), "plain Python only"


@pytest.mark.filterwarnings("ignore")
def test_notebook_runs_on_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("NB_EPOCHS", "1")
    monkeypatch.setenv("NB_SKIP_PYTEST", "1")
    monkeypatch.chdir(ROOT)
    _, code = _cells()
    ns: dict = {"__name__": "__notebook__"}
    for i, c in enumerate(code):
        if "colab" in c["metadata"].get("tags", []):
            continue
        exec(compile("".join(c["source"]), f"<notebook cell {i}>", "exec"), ns)
        if "OUT_DIR" in ns and not str(ns["OUT_DIR"]).startswith(str(tmp_path)):
            ns["OUT_DIR"] = str(tmp_path / "out")
    assert (ns["parity"]["budget_used"] <= 1.0).all() and ns["parity"]["model"].nunique() == 7
    assert len(ns["history"]) == 1 and len(ns["test_df"]) == len(ns["plan"].test_idx)
    assert {"predictions_test.parquet", "predictions_test.parquet.meta.json", "model.pt"} <= set(
        os.listdir(tmp_path / "out"))
