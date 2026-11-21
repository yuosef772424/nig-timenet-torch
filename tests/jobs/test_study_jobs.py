"""ST1 / F2: the study entries (jobs/staged.py, jobs/rolling.py) end to end on the fixture artifact (CPU), strict
configs, and the committed job files."""

import json
import pathlib

import pytest

from nig_timenet_torch.colab.job import JobError, parse_job
from nig_timenet_torch.jobs import rolling, staged

# rule E3: 7 s locally; the fast suite was over budget (runs in CI's slow step)
pytestmark = pytest.mark.slow

ROOT = pathlib.Path(__file__).resolve().parents[2]


def run_entry(module, tmp_path, name, art, monkeypatch, **cfg):  # noqa: F811
    monkeypatch.setenv("NIG_REPO_DIR", str(ROOT))
    config = {"device": "cpu", "run_dir": str(tmp_path / f"{name}_run"), "artifact": str(art), "batch_size": 64,
              "lr": 1e-3, **cfg}
    cp, out = tmp_path / f"{name}.json", tmp_path / name
    cp.write_text(json.dumps(config))
    assert module.main(["--config", str(cp), "--out", str(out), "--max-minutes", "5"]) == 0
    return out


@pytest.mark.slow  # rule E3: fast suite under 100 s on CI (owner 2026-11-19); still runs in CI
def test_staged_entry_runs_the_study(tmp_path, art, monkeypatch):  # noqa: F811
    out = run_entry(staged, tmp_path, "st", art, monkeypatch, seeds=[0, 1], max_rows=96,
                    staged={"start_share": 0.5, "eval_every": 2, "step_cap_epochs": 2, "warmup_steps": 2,
                            "bn_refresh_steps": 2, "k": 2, "probe_rows": 16})
    study = json.loads((out / "study.json").read_text())
    assert set(study["arms"]) == {"staged", "b1", "b2", "b3"} and study["data_fingerprint"]
    assert (out / "study.md").is_file() and (out / "manifest.json").is_file() and (out / "device.json").is_file()
    assert (tmp_path / "st_run" / "staged_s0" / "ckpt" / "checkpoint.pt").is_file()      # checkpoints under run_dir


def test_rolling_entry_runs_the_study(tmp_path, art, monkeypatch):  # noqa: F811
    out = run_entry(rolling, tmp_path, "ro", art, monkeypatch, max_rows=200,
                    rolling={"train_span": "20D", "test_span": "7D", "step": "10D", "embargo": "1D", "max_folds": 2},
                    train={"epochs": 1})
    summary = json.loads((out / "rolling_summary.json").read_text())
    assert summary["n_folds"] == 2 and (out / "rolling_folds.csv").is_file()
    assert (tmp_path / "ro_run" / "fold_000_seed_0" / "ckpt" / "checkpoint.pt").is_file()


@pytest.mark.slow  # rule E3: fast suite under 100 s on CI (owner 2026-11-19); still runs in CI
def test_rolling_entry_pools_the_day_metrics_of_its_folds(tmp_path, art, monkeypatch):  # noqa: F811
    out = run_entry(rolling, tmp_path, "rd", art, monkeypatch, day_metrics=True, split={"cross_section": True},
                    rolling={"train_span": "20D", "test_span": "7D", "step": "10D", "embargo": "1D", "max_folds": 2},
                    train={"epochs": 1})
    assert (out / "fold_001_seed_0" / "panel_report.md").is_file()
    pooled = json.loads((out / "pooled_days_seed_0" / "panel_report.json").read_text())
    folds = [json.loads((out / f"fold_00{i}_seed_0" / "panel_report.json").read_text()) for i in (0, 1)]
    assert pooled["test"]["settings"]["rows"] == sum(f["test"]["settings"]["rows"] for f in folds)
    with pytest.raises(ValueError, match="step >= test_span"):
        rolling.CONFIG(device="cpu", run_dir="x", day_metrics=True,
                       rolling={"train_span": "20D", "test_span": "7D", "step": "5D", "embargo": "1D"})


@pytest.mark.slow  # rule E3: trains 2 folds on CPU; runs in CI's slow step
def test_rolling_entry_parts_share_one_study(tmp_path, art, monkeypatch):  # noqa: F811
    """Owner 2026-11-20: two processes split the folds (``rolling.fold_ids``) into ONE out folder; only the part that
    completes the set writes the summary, the pooled days and the manifest."""
    roll = {"train_span": "20D", "test_span": "7D", "step": "10D", "embargo": "1D", "max_folds": 2}
    common = dict(day_metrics=True, train={"epochs": 1, "selection_metric": "val_raw_loss"},
                  run_dir=str(tmp_path / "parts_run"))
    out = run_entry(rolling, tmp_path, "parts", art, monkeypatch, rolling={**roll, "fold_ids": [1]}, **common)
    assert (out / "fold_001_seed_0" / "fold_row.json").is_file()
    assert not any((out / n).exists() for n in ("rolling_summary.json", "manifest.json", "pooled_days_seed_0"))
    run_entry(rolling, tmp_path, "parts", art, monkeypatch, rolling={**roll, "fold_ids": [0]}, **common)
    summary = json.loads((out / "rolling_summary.json").read_text())
    assert summary["n_folds"] == 2 and (out / "manifest.json").is_file()
    pooled = json.loads((out / "pooled_days_seed_0" / "panel_report.json").read_text())
    folds = [json.loads((out / f"fold_00{i}_seed_0" / "panel_report.json").read_text()) for i in (0, 1)]
    assert pooled["test"]["settings"]["rows"] == sum(f["test"]["settings"]["rows"] for f in folds) > 0


def test_configs_are_strict():
    ok = {"device": "cpu", "run_dir": "x"}
    with pytest.raises(ValueError, match="unknown key"):
        staged.CONFIG(**ok, staged={"gama": 0.5})
    with pytest.raises(ValueError, match="subset"):
        staged.CONFIG(**ok, baselines=("b9",))
    with pytest.raises(ValueError, match="own keys"):
        staged.CONFIG(**ok, train={"lr": 1e-3})
    with pytest.raises(ValueError, match="unknown key"):
        rolling.CONFIG(**ok, rolling={"train_span": "5D", "test_span": "1D", "step": "1D", "embargo": "0D",
                                      "val_shr": 0.1})
    with pytest.raises(TypeError):
        rolling.CONFIG(**ok)                      # the rolling spans have no defaults


@pytest.mark.slow  # rule E3: fast suite under 100 s on CI (owner 2026-11-19); still runs in CI
def test_rolling_entry_pools_the_day_metrics_of_a_single_coin_run(tmp_path, art, monkeypatch):  # noqa: F811
    """rolling-single-001: the day metrics do not need whole-timestamp batches (no ``split.cross_section``)."""
    out = run_entry(rolling, tmp_path, "rs", art, monkeypatch, day_metrics=True,
                    rolling={"train_span": "20D", "test_span": "7D", "step": "10D", "embargo": "1D", "max_folds": 2},
                    train={"epochs": 1, "selection_metric": "val_raw_loss"})
    pooled = json.loads((out / "pooled_days_seed_0" / "panel_report.json").read_text())
    folds = [json.loads((out / f"fold_00{i}_seed_0" / "panel_report.json").read_text()) for i in (0, 1)]
    assert pooled["test"]["settings"]["rows"] == sum(f["test"]["settings"]["rows"] for f in folds) > 0


@pytest.mark.parametrize("name", ["panel-rolling-001", "panel-rolling-base-001", "rolling-single-001",
                                  "rolling-single-smoke-001", "rolling-indr-001", "rolling-indr-smoke-001"])
def test_walk_forward_job_files_build_their_config(name):
    job = parse_job((ROOT / "docs" / "jobs" / f"{name}.json").read_text())
    cfg = rolling.CONFIG(**job["config"])
    assert job["entry"] == "rolling" and cfg.day_metrics and (job["max_minutes"] <= 30) == ("smoke" in name)
    assert cfg.split.get("cross_section", False) == name.startswith("panel")


@pytest.mark.parametrize("name", ["staged-smoke-001", "staged-001", "rolling-smoke-001", "rolling-001",
                                  "panel-smoke-001", "panel-001"])
def test_job_files_are_valid(name):
    job = parse_job((ROOT / "docs" / "jobs" / f"{name}.json").read_text())
    assert job["entry"] == name.split("-")[0] and job["device"] == "cuda" and job["needs_drive"]
    assert (job["max_minutes"] <= 30) == ("smoke" in name) and bool(job["config"].get("max_rows")) == ("smoke" in name)
    with pytest.raises(JobError):
        parse_job(json.dumps({**json.loads((ROOT / "docs" / "jobs" / f"{name}.json").read_text()),
                              "config": {"device": "cuda", "run_dir": "/x/train", "bogus": 1}}))
