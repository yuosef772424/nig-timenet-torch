"""The panel study (workflow/panel_study.py, workflow/panel_report.py, jobs/panel.py) on a small E4 artifact (CPU):
arms x seeds, the day-metric outputs, resume from the kept arm files, the deadline, the oracle frame."""

import json
import time

import numpy as np
import pandas as pd
import pytest
import torch

from nig_timenet_torch.core.config import updated
from nig_timenet_torch.data.artifact_writer import write_artifact
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.data_io.synthetic_panel import PanelSyntheticSettings, build_panel_synthetic
from nig_timenet_torch.eval.day_metrics import FRAME_COLUMNS
from nig_timenet_torch.jobs import panel
from nig_timenet_torch.workflow import panel_report, panel_study
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow.settings import RunSettings

ARMS = {"per_coin": {}, "rank": {"model": {"cross_input": "relative"},
                                 "train": {"cross_section_loss": "rank_ic", "selection_metric": "val_ls_sharpe_close"}}}


@pytest.fixture(scope="module")
def e4_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("e4") / "e4.dataset"
    write_artifact(build_panel_synthetic(PanelSyntheticSettings(n_coins=8, n_bars=720, specific_ic=0.3)), path,
                   source_commit="0" * 40)
    return path


def base(e4_dir, tmp_path) -> RunSettings:
    s = RunSettings(run_dir=str(tmp_path / "run"), data=LoaderSettings(name=str(e4_dir), local_dir=""))
    return updated(s, {"train.epochs": 2, "train.batch_size": 64, "train.device": "cpu", "split.cross_section": True,
                       "evaluation.battery.enabled": False, "train.selection_metric": "val_rank_ic_reg"})


@pytest.mark.slow   # rule E3: 16.7 s on CI, the slowest fast test (fast suite 312.6 s); runs in CI's slow step
def test_study_runs_arms_and_seeds_writes_day_metrics_and_reads_finished_arms_back(e4_dir, tmp_path):
    out, keep = tmp_path / "out", tmp_path / "keep"
    study = panel_study.run_panel_study(base(e4_dir, tmp_path), ARMS, (0, 1), out, ckpt_root=keep)
    assert study["finished"] and set(study["arms"]) == set(ARMS) and study["data_fingerprint"]
    for name in ARMS:
        for seed in (0, 1):
            folder = out / f"{name}_s{seed}"
            assert all((folder / f).is_file() for f in ("panel_report.json", "panel_report.md", "test_report.json",
                                                        "panel_frame_val.parquet", "panel_frame_test.parquet"))
            assert (keep / f"{name}_s{seed}" / "arm.json").is_file() and (keep / f"{name}_s{seed}" / "ckpt").is_dir()
    row = study["arms"]["rank"][0]
    assert row["selection_metric"] == "val_ls_sharpe_close" and row["finished"] and row["absolute"]["days"] > 5
    assert {"win_share", "win_lo", "win_hi", "mean_net", "beats_every_baseline"} <= set(row["neutral"])
    assert set(study["accept"]["rank"]) == {"absolute", "neutral"} and study["accept"]["rank"]["neutral"]["seeds"] == 2
    assert [p["seed"] for p in study["vs_baseline_arm"]["rank"]["neutral"]] == [0, 1]
    text = (out / "study.md").read_text()
    assert "Accept rule" in text and "Against the baseline arm" in text and "| rank | 1 |" in text
    frame = pd.read_parquet(out / "rank_s0" / "panel_frame_test.parquet")
    assert set(FRAME_COLUMNS) <= set(frame.columns) and frame["ret_1d"].notna().mean() > 0.9
    again = panel_report.panel_from_folder(out / "rank_s0", cost=0.0)
    model = again["test"]["neutral"]["model"]
    assert model["gross"]["mean"] == pytest.approx(model["net"]["mean"])
    # a second call trains nothing: every arm is read back from its kept files (new output folder)
    t0 = time.monotonic()
    second = panel_study.run_panel_study(base(e4_dir, tmp_path), ARMS, (0, 1), tmp_path / "out2", ckpt_root=keep)
    assert time.monotonic() - t0 < 5 and second["arms"]["rank"][0]["neutral"] == row["neutral"]
    assert (tmp_path / "out2" / "rank_s1" / "panel_report.md").is_file()


def test_deadline_cuts_between_epochs_and_the_next_call_continues(e4_dir, tmp_path):
    s, out = base(e4_dir, tmp_path), tmp_path / "out"
    cut = panel_study.run_panel_study(s, {"per_coin": {}}, (0, 1), out, deadline=time.monotonic() + 0.01)
    assert not cut["finished"] and cut["arms"]["per_coin"][0]["epochs_done"] == 1 and len(cut["arms"]["per_coin"]) == 1
    assert "NOT finished" in (out / "study.md").read_text() and not cut["accept"]
    done = panel_study.run_panel_study(s, {"per_coin": {}}, (0, 1), out)
    assert done["finished"] and [r["epochs_done"] for r in done["arms"]["per_coin"]] == [2, 2]


def test_oracle_frame_is_the_known_answer_and_beats_a_two_epoch_model(e4_dir, tmp_path):
    r = steps.train(steps.build(steps.prepare(base(e4_dir, tmp_path), tmp_path / "out")))
    oracle = panel_report.panel_report(panel_report.oracle_frame(r, "val"), panel_report.oracle_frame(r, "test"))
    assert oracle["test"]["relative_rank_ic"]["model"] > 0.15                       # planted specific IC 0.3
    assert oracle["test"]["neutral"]["model"]["gross"]["mean"] > 0                  # few days here: the sign only
    frame = panel_report.stage_frame(r, "test")
    ts = np.asarray(r.dataset["last_candles"])[r.idx["test"], 3].astype(np.int64)
    assert np.array_equal(frame["time"].to_numpy().astype(np.int64), ts + int(pd.Timedelta("4h").value))
    assert np.isnan(panel_report.trailing_return(r.dataset, np.array([0]), 6)[0])   # no earlier bar in the artifact


def test_study_and_arms_are_strict(e4_dir, tmp_path):
    s = base(e4_dir, tmp_path)
    with pytest.raises(ValueError, match="cross_section"):
        panel_study.run_panel_study(updated(s, {"split.cross_section": False}), ARMS, (0,), tmp_path / "o")
    with pytest.raises(ValueError, match="'model' and 'train'"):
        panel_study.arm_settings(s, {"split": {}}, 0, "x")
    with pytest.raises(ValueError, match="unknown key"):
        panel_study.arm_settings(s, {"train": {"cross_section_los": "ic"}}, 0, "x")
    ok = {"device": "cpu", "run_dir": "x"}
    assert list(panel.arms_of(panel.CONFIG(**ok))) == list(panel_study.PANEL_001_ARMS)
    assert panel.run_settings(panel.CONFIG(**ok)).split.cross_section
    for bad in ({"arms": ["nope"]}, {"split": {"cross_section": False}}, {"train": {"epochs": 3}},
                {"extra_arms": {"per_coin": {}}}, {"extra_arms": {"x": {"train": {"bogus": 1}}}}, {"seeds": []}):
        with pytest.raises(ValueError):
            panel.CONFIG(**ok, **bad)


def test_panel_entry_runs_and_resumes(e4_dir, tmp_path, monkeypatch):
    import pathlib

    monkeypatch.setenv("NIG_REPO_DIR", str(pathlib.Path(__file__).resolve().parents[2]))
    cfg = {"device": "cpu", "run_dir": str(tmp_path / "drive"), "artifact": str(e4_dir), "batch_size": 64,
           "epochs": 1, "top_coins": 6, "seeds": [0], "arms": ["per_coin", "relative_rank_loss_center"]}
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    for out in ("o1", "o2"):                                    # the second run reads the kept arms back
        assert panel.main(["--config", str(tmp_path / "c.json"), "--out", str(tmp_path / out), "--max-minutes",
                           "5"]) == 0
        study = json.loads((tmp_path / out / "study.json").read_text())
        assert study["finished"] and set(study["arms"]) == {"per_coin", "relative_rank_loss_center"}
        assert (tmp_path / out / "manifest.json").is_file() and (tmp_path / out / "device.json").is_file()
    frame = pd.read_parquet(tmp_path / "o2" / "per_coin_s0" / "panel_frame_test.parquet")
    assert frame["coin"].nunique() == 6                          # top_coins
    for out in ("o1", "o2"):                                    # the battery ran per arm and its report is kept
        assert (tmp_path / out / "per_coin_s0" / "battery_report.txt").is_file()
    assert "battery_tradeable" in study["arms"]["per_coin"][0]


def test_battery_step_writes_the_day_metrics_when_asked(e4_dir, tmp_path):
    s = updated(base(e4_dir, tmp_path), {"evaluation.battery.day_metrics": True})
    r = steps.train(steps.build(steps.prepare(s, tmp_path / "out")))
    steps.test_step(r)
    assert steps.battery_step(r) == {}                           # the battery itself is off in ``base``
    assert (tmp_path / "out" / "panel_report.md").is_file() and "absolute" in r.extra["panel"]
    off = steps.train(steps.build(steps.prepare(base(e4_dir, tmp_path / "b"), tmp_path / "out_b")))
    steps.test_step(off)
    steps.battery_step(off)
    assert not (tmp_path / "out_b" / "panel_report.md").exists()  # off by default


@pytest.mark.slow
def test_logret_std_arm_trains_with_the_rank_loss_and_decodes_to_label_units(e4_dir, tmp_path):
    """The owner arm of 2026-11-17: train-standardised inputs + per-head standardised labels (model.target_norm) with
    the relative inputs and the cross-section rank term; the fitted state is the TRAIN label mean / std and the
    outputs (the frames the day metrics read) are back in label units."""
    arm = panel_study.ARMS["relative_rank_loss_logret_std"]
    assert arm["model"]["target_norm"] == "logret_std" and arm["train"]["cross_section_loss"] == "rank_ic"
    assert "relative_rank_loss_logret_std" not in panel.PanelConfig(device="cpu", run_dir="x").arms
    r = steps.build(steps.prepare(panel_study.arm_settings(base(e4_dir, tmp_path), arm, 0, tmp_path / "ck"),
                                  tmp_path / "o"))
    state = r.extra["target_norm"]
    for t in r.targets:
        y = torch.as_tensor(r.y["train"][f"y_{t}_reg"]).double().numpy()
        assert state[t]["mean"] == pytest.approx(float(y.mean()), rel=1e-6, abs=1e-9)
        assert state[t]["std"] == pytest.approx(float(y.std()), rel=1e-6)
    row = panel_study.run_arm(base(e4_dir, tmp_path), "relative_rank_loss_logret_std", arm, 0, tmp_path / "out")
    assert row["finished"] and np.isfinite(row["test_rank_ic"]["close"])
    frame = pd.read_parquet(panel_report.frame_path(tmp_path / "out" / "relative_rank_loss_logret_std_s0", "test"))
    y_sd = float(torch.as_tensor(r.y["train"]["y_close_reg"]).double().std())
    assert np.isfinite(frame["mu"]).all() and float(np.std(frame["mu"])) < 5 * y_sd   # label units, not z units
