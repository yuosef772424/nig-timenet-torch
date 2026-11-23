"""The verification step (workflow/verification_run.py) and its settings: a Run in, the verdict table files out."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from nig_timenet_torch.core.config import from_dict, updated
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow.settings import EvalSettings, RunSettings, VerificationSettings
from nig_timenet_torch.workflow.verification_run import verification_step
from tests.data_io.helpers import art  # noqa: F401  (fixture)
from tests.eval.verification_world import TARGETS, make_world


def fake_run(tmp_path, changes=None, trained=True):
    """A Run-shaped object over the synthetic world: what ``verification_step`` reads, nothing it does not."""
    ds, idx, tables = make_world(kind="skilled")
    settings = updated(RunSettings(run_dir=str(tmp_path / "run")), {"evaluation.verification.n_boot": 50,
                                                                    "evaluation.verification.run_candle": False,
                                                                    **(changes or {})})
    cached = {s: t for s, t in tables.items()}
    calls = []

    def table(stage):
        calls.append(stage)
        return cached[stage]
    run = SimpleNamespace(settings=settings, dataset=ds, idx=idx, targets=TARGETS, out=tmp_path / "out", table=table,
                          result=object() if trained else None, extra={})
    return run, calls


def test_the_step_writes_the_summary_files_with_the_settings_and_the_battery_cost(tmp_path):
    run, calls = fake_run(tmp_path, {"evaluation.verification.target_acc": 0.7, "evaluation.verification.rr": 3.0,
                                     "evaluation.battery.cost_per_side": 0.0}, )
    out = verification_step(run)
    assert set(out) == {"summary", "details"} and calls == ["val", "test"]
    assert (tmp_path / "out" / "verification_summary.csv").is_file()
    latest = pd.read_csv(tmp_path / "out" / "verification_latest.csv")      # the per-target latest table, next to it
    assert set(latest["target"]) == set(TARGETS) and len(latest) == 3 * 3 and "err_px" in latest.columns
    records = json.loads((tmp_path / "out" / "verification_summary.json").read_text(encoding="utf-8"))
    assert len(records) == len(out["summary"])
    s = out["summary"]
    assert s["check"].str.contains("slice >= 70%").any() and s["check"].str.contains("trades 3:1").any()
    assert s["verdict"].eq("✅").sum() > 10                      # a skilled model: most checks pass
    assert set(run.extra["tables"]) == {"val", "test"}                # kept for the other steps, predicted once
    verification_step(run)
    assert calls == ["val", "test"]                                   # the second call reads the cached tables


def test_the_step_reuses_the_tables_the_battery_kept_and_reads_the_target_mode(tmp_path):
    run, calls = fake_run(tmp_path)
    run.extra["tables"] = {"val": run.table("val"), "test": run.table("test")}
    calls.clear()
    run.dataset = {**run.dataset, "target_mode": "scaled"}
    s = verification_step(run)["summary"]
    assert calls == [] and (s["verdict"] == "ℹ️").sum() >= 3        # label check x2 and block c skipped


def test_match_old_gives_the_old_text_cost_and_no_latest_table(tmp_path):
    run, _ = fake_run(tmp_path, {"evaluation.verification.match_old": True, "evaluation.verification.n_boot": 30})
    out = verification_step(run, verbose=True)
    s = out["summary"]
    assert "أ) البيانات" in set(s["section"]) and "latest" not in out["details"]
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["verification_summary.csv",
                                                                    "verification_summary.json"]
    run2, _ = fake_run(tmp_path / "b", {"evaluation.verification.lang": "ar", "evaluation.verification.n_boot": 30})
    ar = verification_step(run2)["summary"]
    assert "أ) البيانات" in set(ar["section"]) and (tmp_path / "b" / "out" / "verification_latest.csv").is_file()


def test_the_step_off_returns_nothing_and_writes_nothing(tmp_path):
    run, calls = fake_run(tmp_path, {"evaluation.verification.enabled": False})
    assert verification_step(run) == {} and calls == [] and not (tmp_path / "out").exists()


def test_the_step_needs_a_trained_run_and_test_rows(tmp_path):
    run, _ = fake_run(tmp_path, trained=False)
    with pytest.raises(RuntimeError, match="train"):
        verification_step(run)
    run, _ = fake_run(tmp_path)
    run.idx = {**run.idx, "test": run.idx["test"][:0]}
    with pytest.raises(ValueError, match="no test rows"):
        verification_step(run)


@pytest.mark.slow
def test_candle_runs_when_asked(tmp_path):
    run, _ = fake_run(tmp_path, {"evaluation.verification.run_candle": True})
    s = verification_step(run)["summary"]
    assert (s["section"] == "d) candle shape").sum() == 2


def test_verification_settings_are_strict_and_default_to_the_old_values():
    v = VerificationSettings()
    assert (v.enabled, v.target_acc, v.rr, v.stop_mode, v.n_boot, v.run_candle) == (True, 0.65, 2.0, "nig", 500, True)
    assert RunSettings(run_dir="x").evaluation.verification == v and EvalSettings().verification == v
    assert updated(RunSettings(run_dir="x"), {"evaluation.verification.stop_mode": "model_levels"}) \
        .evaluation.verification.stop_mode == "model_levels"
    assert (v.latest_n, v.latest_coins) == (1, 3)
    assert (v.match_old, v.lang, v.stop_width, v.bootstrap_unit) == (False, "en", "predictive_std", "day")
    for bad in ({"stop_mode": "atr"}, {"target_acc": 1.0}, {"target_acc": 0.0}, {"rr": 0.0}, {"n_boot": 0},
                {"latest_n": -1}, {"latest_coins": 0}, {"lang": "fr"}, {"stop_width": "sd"},
                {"bootstrap_unit": "week"}):
        with pytest.raises(ValueError):
            VerificationSettings(**bad)
    with pytest.raises(ValueError, match="unknown key"):
        from_dict(RunSettings, {"run_dir": "x", "evaluation": {"verification": {"min_n": 5}}})


@pytest.mark.slow
def test_verification_step_on_a_real_run_of_the_fixture(art, tmp_path):  # noqa: F811
    s = updated(RunSettings(run_dir=str(tmp_path / "run"), data=LoaderSettings(name=str(art))),
                {"train.epochs": 1, "train.batch_size": 64, "train.device": "cpu",
                 "evaluation.verification.n_boot": 30})
    run = steps.run_all(s, tmp_path / "out")
    out = verification_step(run)
    summary = pd.read_csv(tmp_path / "out" / "verification_summary.csv", encoding="utf-8-sig")
    assert len(summary) == len(out["summary"])
    assert {"a) data", "b) baselines", "d) candle shape"} <= set(summary["section"])
    labels = summary[summary["check"].str.startswith("labels match")]
    assert (labels["verdict"] == "✅").all()                 # the fixture's labels ARE the raw prices
