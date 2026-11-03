"""T5r: the baseline entry end to end on the fixture artifact (1 epoch, CPU), resume, and the committed job files."""

import json
import pathlib

import pytest

from nig_timenet_torch.colab import contract
from nig_timenet_torch.colab.job import parse_job
from nig_timenet_torch.jobs import baseline
from tests.data_io.test_artifact import art  # noqa: F401  (fixture)

ROOT = pathlib.Path(__file__).resolve().parents[2]


def run(tmp_path, art, name, epochs, run_dir):  # noqa: F811
    cfg = {"device": "cpu", "seed": 0, "epochs": epochs, "batch_size": 64, "run_dir": str(run_dir),
           "artifact": str(art), "health_max_rows": 50}
    cp, out = tmp_path / f"{name}.json", tmp_path / name
    cp.write_text(json.dumps(cfg))
    assert baseline.main(["--config", str(cp), "--out", str(out), "--max-minutes", "5"]) == 0
    return out, cfg


def test_baseline_runs_and_resumes(tmp_path, art, monkeypatch):  # noqa: F811
    monkeypatch.setenv("NIG_REPO_DIR", str(ROOT))
    run_dir = tmp_path / "drive" / "train"
    out1, cfg = run(tmp_path, art, "o1", 1, run_dir)
    s1 = json.loads((out1 / "baseline_summary.json").read_text())
    assert s1["epochs_done"] == 1 and s1["seed"] == 0 and s1["n_windows"]["train"] > 0
    assert s1["step_seconds_per_batch"] > 0 and len(s1["data_fingerprint"]) > 8 and s1["identity_fingerprint"]
    for name in ("health_report.json", "health_report.md", "test_report.json", "device.json", "manifest.json"):
        assert (out1 / name).is_file()
    test_rep = json.loads((out1 / "test_report.json").read_text())
    assert test_rep["n_rows"] == s1["n_windows"]["test"] > 0 and test_rep["targets"]
    job = {"device": "cpu", "expected_outputs": ["baseline_summary.json", "health_report.json", "health_report.md"]}
    man = json.loads((out1 / "manifest.json").read_text())
    assert contract.validate_outputs(out1, job, man["git_sha"]) == []
    out2, _ = run(tmp_path, art, "o2", 2, run_dir)  # same run_dir: resumes epoch 1, runs epoch 2
    s2 = json.loads((out2 / "baseline_summary.json").read_text())
    assert s2["epochs_done"] == 2 and s2["history"][0]["epoch"] == 1 and len(s2["history"]) == 2


def test_deadline_stops_cleanly_and_data_may_stay_on_cpu(tmp_path, art, monkeypatch):  # noqa: F811
    from nig_timenet_torch.trainer.trainer import FitSettings
    from tests.trainer.test_fit import identity_for, load_fixture, make_trainer

    x, y = load_fixture(120)
    t = make_trainer()
    s = FitSettings(epochs=50, batch_size=32)
    r = t.fit(x[:80], {k: v[:80] for k, v in y.items()}, x[80:], {k: v[80:] for k, v in y.items()}, s,
              run_dir=tmp_path / "r", identity=identity_for(s), deadline=0.0)
    assert 1 <= r.epochs_done <= 2  # one epoch always runs; the second is refused by the deadline


@pytest.mark.parametrize("seed", [0, 1])
def test_job_files_are_valid(seed):
    job = parse_job((ROOT / "docs" / "jobs" / f"baseline-001-s{seed}.json").read_text())
    assert job["entry"] == "baseline" and job["device"] == "cuda" and job["needs_drive"] and job["max_minutes"] == 170
    assert job["config"]["seed"] == seed and job["config"]["run_dir"].endswith(f"baseline-001-s{seed}/train")


def test_input_key_follows_the_artifact_timeframe():
    """The real TF artifact is 4h (owner's first Colab run failed on a hard-coded '1h')."""
    from nig_timenet_torch.jobs.baseline import input_key
    assert input_key({"timeframes": ["4h"]}) == "X_4h"
    assert input_key({"timeframes": ["1h"]}) == "X_1h"
    with pytest.raises(ValueError, match="single-timeframe"):
        input_key({"timeframes": ["1h", "4h"]})


def test_split_bounds_without_split_dates_uses_tf_shares():
    """The real TF artifact has split_dates None (the owner's second Colab run failed on it)."""
    import numpy as np
    import pandas as pd

    from nig_timenet_torch.core.schema import TS_COL
    from nig_timenet_torch.jobs.baseline import split_bounds
    lc = np.zeros((100, 7))
    lc[:, TS_COL] = pd.date_range("2026-01-01", periods=100, freq="4h", tz="UTC").as_unit("ns").asi8
    tr, va = split_bounds({"split_dates": None, "last_candles": lc}, {})
    assert tr == pd.Timestamp(int(lc[80, TS_COL]), tz="UTC") and va == pd.Timestamp(int(lc[90, TS_COL]), tz="UTC")
    given = {"train_end": "2026-01-05", "val_end": "2026-01-10"}
    got = split_bounds({"split_dates": None, "last_candles": lc}, {"split_dates": given})
    assert got == ("2026-01-05", "2026-01-10")


def test_epoch_line_shows_progress():
    from nig_timenet_torch.jobs.baseline import epoch_line
    line = epoch_line({"epoch": 3, "lr": 1e-3, "train_total_loss": 0.5, "val_total_loss": 0.6,
                       "val_selection_metric": 0.52, "trace_epoch_seconds": 41.2}, 60)
    assert line.startswith("epoch 3/60") and "val_loss 0.6000" in line and "41s" in line
    assert "train_loss -" in epoch_line({"epoch": 1, "lr": 1e-3}, 60)
