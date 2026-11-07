"""T5r: the baseline entry end to end on the fixture artifact (1 epoch, CPU), resume, and the committed job files."""

import json
import pathlib

import pytest

from nig_timenet_torch.colab import contract
from nig_timenet_torch.colab.job import parse_job
from nig_timenet_torch.jobs import baseline
from tests.jobs.helpers import run_cfg

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
    for name in ("health_report.json", "health_report.md", "test_report.json", "device.json", "manifest.json",
                 "battery_report.txt", "battery_profit.json"):
        assert (out1 / name).is_file()
    profit = json.loads((out1 / "battery_profit.json").read_text())
    assert profit["cost_per_side"] == 0.0008 and {"profit_close", "verdict"} <= set(profit)
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
    from tests.trainer.helpers import identity_for, load_fixture, make_trainer

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
    assert {"test_report.json", "battery_report.txt", "battery_profit.json"} <= set(job["expected_outputs"])


def test_every_long_job_has_a_smoke_job():
    """Rule E5: a job file with max_minutes > 30 needs a smoke job file of the same entry (max_rows, <= 30 min)."""
    jobs = [parse_job(p.read_text()) for p in sorted((ROOT / "docs" / "jobs").glob("*.json"))]
    smokes = {j["entry"] for j in jobs if j["max_minutes"] <= 30 and j["config"].get("max_rows")}
    long_without = [j["job_id"] for j in jobs if j["max_minutes"] > 30 and j["entry"] not in smokes]
    assert not long_without, f"rule E5: no smoke job for the entry of {long_without}"


def test_smoke_job_file_is_valid():
    job = parse_job((ROOT / "docs" / "jobs" / "baseline-smoke-001.json").read_text())
    assert job["entry"] == "baseline" and job["device"] == "cuda" and job["config"]["max_rows"] == 20000
    assert job["config"]["epochs"] == 1 and job["max_minutes"] == 15
    assert {"test_report.json", "battery_report.txt", "battery_profit.json"} <= set(job["expected_outputs"])


def test_input_key_follows_the_artifact_timeframe():
    """The real TF artifact is 4h (owner's first Colab run failed on a hard-coded '1h')."""
    from nig_timenet_torch.workflow.run_common import input_key
    assert input_key({"timeframes": ["4h"]}) == "X_4h"
    assert input_key({"timeframes": ["1h"]}) == "X_1h"
    with pytest.raises(ValueError, match="single-timeframe"):
        input_key({"timeframes": ["1h", "4h"]})


def test_split_bounds_without_split_dates_uses_tf_shares():
    """The real TF artifact has split_dates None (the owner's second Colab run failed on it)."""
    import numpy as np
    import pandas as pd

    from nig_timenet_torch.core.schema import TS_COL
    from nig_timenet_torch.workflow.run_common import split_bounds
    from nig_timenet_torch.workflow.settings import SplitSettings
    lc = np.zeros((100, 7))
    lc[:, TS_COL] = pd.date_range("2026-01-01", periods=100, freq="4h", tz="UTC").as_unit("ns").asi8
    tr, va = split_bounds({"split_dates": None, "last_candles": lc}, SplitSettings())
    assert tr == pd.Timestamp(int(lc[80, TS_COL]), tz="UTC") and va == pd.Timestamp(int(lc[90, TS_COL]), tz="UTC")
    given = SplitSettings(train_end="2026-01-05", val_end="2026-01-10")
    got = split_bounds({"split_dates": None, "last_candles": lc}, given)
    assert got == ("2026-01-05", "2026-01-10")


def test_epoch_line_shows_progress():
    from nig_timenet_torch.workflow.flat_run import epoch_line
    line = epoch_line({"epoch": 3, "lr": 1e-3, "train_total_loss": 0.5, "val_total_loss": 0.6,
                       "val_selection_metric": 0.52, "trace_epoch_seconds": 41.2}, 60)
    assert line.startswith("epoch 3/60") and "val_loss 0.6000" in line and "41s" in line
    assert "train_loss -" in epoch_line({"epoch": 1, "lr": 1e-3}, 60)




def test_baseline_on_a_tf_shaped_artifact(tmp_path, monkeypatch):
    """HF-15: 4h, split_dates None, real-artifact-like manifest (the shape that broke runs #21 and #22)."""
    from tests.data_io.variants import tf_shaped_artifact
    art_dir = tf_shaped_artifact(tmp_path / "tf.dataset")
    out = run_cfg(tmp_path, "tf", art_dir, monkeypatch, max_rows=100)
    s = json.loads((out / "baseline_summary.json").read_text())
    assert s["input_key"] == "X_4h" and s["max_rows"] == 100 and s["targets"] == ["high", "low", "close"]
    assert s["reg_target_mode"] == "return" and s["reg_target_scale"] == 100.0
    assert 0 < s["n_windows"]["train"] <= 100 and 0 < s["n_windows"]["test"] <= 100
    assert (out / "test_report.json").is_file()


def test_unknown_job_config_key_raises(tmp_path, art, monkeypatch):  # noqa: F811
    with pytest.raises(ValueError, match="unknown key.*epoch"):
        run_cfg(tmp_path, "bad", art, monkeypatch, epoch=3)


def test_unknown_codec_mode_or_missing_scale_or_bad_targets_raise(tmp_path, monkeypatch):
    from nig_timenet_torch.core.registry import UnknownNameError
    from tests.data_io.variants import tf_shaped_artifact
    with pytest.raises(UnknownNameError):
        run_cfg(tmp_path, "m", tf_shaped_artifact(tmp_path / "m.dataset", reg_target_mode="volnorm"), monkeypatch)
    with pytest.raises(ValueError, match="reg_target_scale"):
        run_cfg(tmp_path, "s", tf_shaped_artifact(tmp_path / "s.dataset", reg_target_scale=...), monkeypatch)
    with pytest.raises(ValueError, match="price_targets"):
        run_cfg(tmp_path, "t", tf_shaped_artifact(tmp_path / "t.dataset", price_targets=["high", "mid"]), monkeypatch)


def test_pickle_artifact_is_refused_by_the_job(tmp_path, monkeypatch):
    from nig_timenet_torch.data_io.artifact import ArtifactError
    (tmp_path / "x.pkl.gz").write_bytes(b"")
    with pytest.raises(ArtifactError, match="D1"):
        run_cfg(tmp_path, "p", tmp_path / "x.pkl.gz", monkeypatch)


def test_stride_rows_and_anchors_by_name():
    import numpy as np

    from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX
    from nig_timenet_torch.workflow.flat_run import anchors_by_name
    from nig_timenet_torch.workflow.run_common import stride_rows
    idx = np.arange(1000)
    assert stride_rows(idx, None) is idx and len(stride_rows(idx, 50)) == 50
    assert stride_rows(idx, 50)[0] == 0 and stride_rows(idx, 50)[-1] == 999
    lc = np.arange(70, dtype=float).reshape(10, 7)
    a = anchors_by_name(lc, np.array([2, 3]), ("close", "high"))
    assert (a["close"] == lc[[2, 3], LAST_COLUMN_INDEX["last_close"]]).all()
    assert (a["high"] == lc[[2, 3], LAST_COLUMN_INDEX["last_high"]]).all()


def test_job_validator_checks_the_entry_config():
    from nig_timenet_torch.colab.job import JobError, validate_job
    base = {"job_id": "base-001", "repo_ref": "a" * 40, "entry": "baseline", "args": [],
            "config": {"device": "cpu", "seed": 0, "epochs": 1, "batch_size": 8, "run_dir": "/x"},
            "max_minutes": 5, "device": "cpu", "needs_drive": False, "expected_outputs": ["a.json"], "note": ""}
    validate_job(base)
    with pytest.raises(JobError, match="unknown key"):
        validate_job({**base, "config": {**base["config"], "max_row": 5}})
    with pytest.raises(JobError, match="missing required"):
        validate_job({**base, "config": {"device": "cpu"}})
