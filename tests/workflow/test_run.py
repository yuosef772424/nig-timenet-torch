"""The run steps (workflow/run.py) on the fixture artifact: settings are strict, steps run in order and write the
outputs the jobs and notebooks rely on."""

import json

import pytest

from nig_timenet_torch.core.config import from_dict, updated
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow.settings import RunSettings
from tests.data_io.test_artifact import art  # noqa: F401  (fixture)


def settings(art_dir, tmp_path, **over):  # noqa: F811
    base = RunSettings(run_dir=str(tmp_path / "run"), data=LoaderSettings(name=str(art_dir)))
    return updated(base, {"train.epochs": 1, "train.batch_size": 64, "train.device": "cpu", **over})


def test_settings_are_strict(tmp_path):
    with pytest.raises(ValueError, match="unknown key"):
        from_dict(RunSettings, {"run_dir": "x", "epochs": 3})
    with pytest.raises(ValueError, match="run_dir"):
        RunSettings(run_dir="")
    with pytest.raises(ValueError, match="together"):
        updated(RunSettings(run_dir="x"), {"split.train_end": "2026-01-01"})
    with pytest.raises(ValueError, match="max_rows"):
        updated(RunSettings(run_dir="x"), {"split.max_rows": 0})


def test_steps_need_their_predecessors(art, tmp_path):  # noqa: F811
    r = steps.prepare(settings(art, tmp_path), tmp_path / "out")
    with pytest.raises(RuntimeError, match="build"):
        steps.train(r)
    steps.build(r)
    with pytest.raises(RuntimeError, match="train"):
        steps.test_step(r)


def test_run_all_writes_the_reports_and_smoke_mode_limits_rows(art, tmp_path):  # noqa: F811
    out = tmp_path / "out"
    r = steps.run_all(settings(art, tmp_path, **{"split.max_rows": 60, "evaluation.health_max_rows": 40}), out)
    assert all(len(r.idx[k]) <= 60 for k in ("train", "val", "test"))
    for name in ("test_report.json", "health_report.json", "health_report.md", "battery_report.txt",
                 "battery_profit.json"):
        assert (out / name).is_file()
    summary = steps.write_summary(r)
    assert summary["max_rows"] == 60 and summary["epochs_done"] == 1 and (out / steps.SUMMARY).is_file()
    assert json.loads((out / "test_report.json").read_text())["n_rows"] == len(r.idx["test"])


def test_battery_can_be_switched_off(art, tmp_path):  # noqa: F811
    out = tmp_path / "out"
    r = steps.run_all(settings(art, tmp_path, **{"evaluation.battery.enabled": False}), out)
    assert not (out / "battery_report.txt").exists() and steps.battery_step(r) == {}


def test_battery_reruns_from_the_finished_folder_alone(art, tmp_path):  # noqa: F811
    from nig_timenet_torch.workflow.battery_run import battery_from_folder, render
    out = tmp_path / "out"
    r = steps.run_all(settings(art, tmp_path), out)
    again = battery_from_folder(out, cost=0.0008)
    assert render(again) == (out / "battery_report.txt").read_text()
    other = battery_from_folder(out, cost=0.002, n_variants=3, out_dir=tmp_path / "again")
    assert "0.2000%" in other["trades"]["note"] and (tmp_path / "again" / "battery_profit.json").is_file()
    assert r.settings.evaluation.battery.cost_per_side == 0.0008
    with pytest.raises(FileNotFoundError, match="battery tables"):
        battery_from_folder(tmp_path)
