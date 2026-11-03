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
    for name in ("health_report.json", "health_report.md", "device.json", "manifest.json"):
        assert (out1 / name).is_file()
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
