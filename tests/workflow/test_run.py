"""The run steps (workflow/run.py) on the fixture artifact: settings are strict, steps run in order and write the
outputs the jobs and notebooks rely on."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nig_timenet_torch.core.config import from_dict, updated
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.trainer import checkpoint
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow.settings import RunSettings


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


def test_restore_rereads_a_finished_run_folder_without_training(art, tmp_path):  # noqa: F811
    """T5r: restore() puts the best weights of a finished run folder in a freshly built model; its test report equals
    the trained run's; a changed seed (another identity) is refused."""
    s = settings(art, tmp_path)
    trained = steps.run_all(s, tmp_path / "out1")
    rep1 = json.loads((tmp_path / "out1" / "test_report.json").read_text())
    r = steps.restore(steps.build(steps.prepare(s, tmp_path / "out2")))
    assert r.result.best_epoch == trained.result.best_epoch and r.result.restored_best
    rep2 = json.loads(json.dumps(steps.test_step(r), default=float))
    assert rep2 == rep1  # the same weights on the same rows (CPU, deterministic)
    other = steps.build(steps.prepare(updated(s, {"train.seed": 7})))
    with pytest.raises(ValueError, match="identity changed"):
        steps.restore(other)
    empty = steps.build(steps.prepare(updated(s, {"run_dir": str(tmp_path / "none")})))
    with pytest.raises(FileNotFoundError, match="nothing to restore"):
        steps.restore(empty)


def test_restore_last_loads_the_last_epoch_weights_not_the_best(art, tmp_path):  # noqa: F811
    """weights="last" puts the weights after the LAST finished epoch (checkpoint.pt) in the model, weights="best"
    (the default) best.pt; an unknown name raises."""
    s = settings(art, tmp_path)
    steps.run_all(s, tmp_path / "out1")
    ckpt = checkpoint.load_checkpoint(Path(s.run_dir))

    def weights_after(**kw):
        r = steps.restore(steps.build(steps.prepare(s, tmp_path / "out2")), **kw)
        return r, r.trainer.model.state_dict()

    r_last, w_last = weights_after(weights="last")
    assert not r_last.result.restored_best
    assert all(torch.equal(w_last[k], v) for k, v in ckpt["model"].items())
    best = checkpoint.load_best(Path(s.run_dir))
    _, w_default = weights_after()
    assert all(torch.equal(w_default[k], v) for k, v in best.items())
    with pytest.raises(ValueError, match="weights must be one of"):
        steps.restore(steps.build(steps.prepare(s, tmp_path / "out3")), weights="newest")


def test_fit_without_restore_best_weights_leaves_the_last_epoch_in_the_model(art, tmp_path):  # noqa: F811
    """train.early_stop.restore_best_weights=False: after fit the model holds the weights of the last finished epoch
    (the ones checkpoint.pt keeps), so every report step reads them."""
    s = updated(settings(art, tmp_path), {"train.early_stop.restore_best_weights": False})
    r = steps.run_all(s, tmp_path / "out")
    ckpt = checkpoint.load_checkpoint(Path(s.run_dir))
    w = r.trainer.model.state_dict()
    assert not r.result.restored_best
    assert all(torch.equal(w[k], v) for k, v in ckpt["model"].items())


def test_exclude_coins_removes_their_rows_from_every_split_and_keeps_the_cut_dates(art, tmp_path):  # noqa: F811
    from nig_timenet_torch.workflow.flat_run import row_coins
    base = steps.prepare(settings(art, tmp_path))
    names = [b["name"] for b in base.dataset["asset_bounds"]]
    gone = [names[0], names[1].replace("USDT", "").lower()]  # one exact, one without USDT and in lower case
    r = steps.prepare(settings(art, tmp_path, **{"split.exclude_coins": gone}))
    keep = {n for n in names[:2]}
    for stage in ("train", "val", "test"):
        assert not keep & set(row_coins(r.dataset, r.idx[stage]).astype(str))
        want = base.idx[stage][~np.isin(row_coins(base.dataset, base.idx[stage]).astype(str), list(keep))]
        assert np.array_equal(r.idx[stage], want)  # nothing else moved
    assert r.bounds == base.bounds
    assert len(r.x["train"]) < len(base.x["train"])


def test_exclude_coins_refuses_unknown_names_and_removing_everything(art, tmp_path):  # noqa: F811
    with pytest.raises(ValueError, match="not coins of this artifact"):
        steps.prepare(settings(art, tmp_path, **{"split.exclude_coins": ["NOPEUSDT"]}))
    names = [b["name"] for b in steps.prepare(settings(art, tmp_path)).dataset["asset_bounds"]]
    with pytest.raises(ValueError, match="every coin"):
        steps.prepare(settings(art, tmp_path, **{"split.exclude_coins": names}))
    with pytest.raises(ValueError, match="coin names"):
        updated(RunSettings(run_dir="x"), {"split.exclude_coins": [""]})
