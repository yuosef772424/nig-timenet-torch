"""Baseline-job test helper (HF-32)."""

import json
import pathlib

from nig_timenet_torch.jobs import baseline

ROOT = pathlib.Path(__file__).resolve().parents[2]


def run_cfg(tmp_path, name, art_dir, monkeypatch, **over):
    """baseline.main on ``art_dir`` with a job config (1 epoch, CPU); ``over`` overrides / adds config keys."""
    monkeypatch.setenv("NIG_REPO_DIR", str(ROOT))
    cfg = {"device": "cpu", "seed": 0, "epochs": 1, "batch_size": 64, "run_dir": str(tmp_path / f"{name}_run"),
           "artifact": str(art_dir), "health_max_rows": 50, **over}
    cp, out = tmp_path / f"{name}.json", tmp_path / name
    cp.write_text(json.dumps(cfg))
    assert baseline.main(["--config", str(cp), "--out", str(out), "--max-minutes", "5"]) == 0
    return out
