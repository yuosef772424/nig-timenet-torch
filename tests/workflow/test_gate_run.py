"""The gate run (workflow/gate_run.py) and the job-side check: E1 -> E2 through the normal run steps writes a gate
report keyed by the configuration hash; a full training of a non-baseline configuration is refused without a passing
report of its key; the baseline arm is exempt and recorded (ROADMAP G2, rule C6)."""

import json
from dataclasses import replace

import pytest

from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.data_io.synthetic import SyntheticSettings, write_synthetic
from nig_timenet_torch.jobs.gates import GatesConfig, gate_settings, run_settings
from nig_timenet_torch.model.architectures import ModelSettings
from nig_timenet_torch.workflow import gate_run, run
from nig_timenet_torch.workflow.settings import RunSettings, SplitSettings, TrainSettings

SHA = "d" * 40
TINY_E1 = {"n_coins": 16, "n_bars": 600}


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    d = tmp_path_factory.mktemp("gates")
    cfg = GatesConfig(device="cpu", run_dir=str(d / "work"), architecture="tiny_mlp", seeds=(0,), e1=TINY_E1,
                      e1_epochs=1, e2_epochs=1, train={"batch_size": 128})
    return gate_run.run_gates(run_settings(cfg, SHA), gate_settings(cfg), d / "out"), d


def test_gate_run_writes_a_keyed_report(report):
    rep, d = report
    on_disk = json.loads((d / "out" / gate_run.GATE_REPORT).read_text())
    assert on_disk["key"] == rep["key"] and len(rep["key"]) == 64 and rep["schema"] == gate_run.REPORT_SCHEMA
    assert (d / "out" / gate_run.GATE_REPORT_MD).read_text().startswith("# Gate report")
    assert set(rep["envs"]) == {"E1", "E2"} and rep["envs"]["E2"]["permutes"].startswith("E1")
    assert {(r["env"], r["gate"]) for r in rep["rows"]} >= {("E1", g) for g in "ABCSD"} | {
        ("E2", g) for g in ("A", "B", "C", "canary")}
    assert {r["head"] for r in rep["rows"] if r["gate"] == "A"} == {"start", "end"}
    s = rep["summary"]
    assert not s["passed"] and set(s["envs_missing_seeds"]) == {"E1", "E2", "E3"}, "1 seed and no E3: never passes"
    assert rep["baseline_arm"] is False and rep["configuration"]["model"]["architecture"] == "tiny_mlp"
    assert "seed" not in rep["configuration"]["train"]


def _built(art, tmp_path, model=None, epochs=2, max_rows=None, batch_size=128):
    s = RunSettings(run_dir=str(tmp_path / "run"), code_sha=SHA, data=LoaderSettings(name=str(art), local_dir=""),
                    split=SplitSettings(max_rows=max_rows),
                    train=TrainSettings(epochs=epochs, batch_size=batch_size, device="cpu"))
    if model is not None:
        s = replace(s, model=model)
    return run.build(run.prepare(s))


@pytest.fixture(scope="module")
def e1_art(tmp_path_factory):
    d = tmp_path_factory.mktemp("e1") / "E1"
    write_synthetic(d, SyntheticSettings(**TINY_E1), source_commit=SHA)
    return d


def test_gate_check_refuses_an_ungated_full_training(e1_art, tmp_path, report):
    tiny = ModelSettings("tiny_mlp", {})
    r = _built(e1_art, tmp_path, model=tiny)
    assert not gate_run.baseline_arm(r.settings)
    with pytest.raises(gate_run.GateRefused, match="without a gate report"):
        gate_run.gate_check(r, None)
    rep, d = report
    assert gate_run.configuration_key(r) == rep["key"], "same configuration, other data and seed: same key"
    with pytest.raises(gate_run.GateRefused, match="did not pass"):
        gate_run.gate_check(r, str(d / "out" / gate_run.GATE_REPORT))
    passing = tmp_path / "pass.json"
    passing.write_text(json.dumps({"key": rep["key"], "summary": {"passed": True, "failed": []}}))
    assert gate_run.gate_check(r, str(passing))["result"] == "gate report passed"
    assert r.extra["gate_check"]["report_passed"]
    other = _built(e1_art, tmp_path, model=tiny, batch_size=64)
    with pytest.raises(gate_run.GateRefused, match="is for configuration"):
        gate_run.gate_check(other, str(passing))


def test_baseline_arm_is_exempt_and_short_runs_are_not_checked(e1_art, tmp_path):
    r = _built(e1_art, tmp_path)
    assert gate_run.baseline_arm(r.settings)
    rec = gate_run.gate_check(r, None)
    assert rec["result"].startswith("baseline arm: exempt") and rec["full_training"]
    smoke = _built(e1_art, tmp_path, model=ModelSettings("tiny_mlp", {}), epochs=1)
    assert gate_run.gate_check(smoke, None)["result"] == "short run: not checked"
    strided = _built(e1_art, tmp_path, model=ModelSettings("tiny_mlp", {}), max_rows=500)
    assert gate_run.gate_check(strided, None)["result"] == "short run: not checked"


def test_gates_config_is_strict():
    with pytest.raises(ValueError, match="may not set"):
        GatesConfig(device="cpu", run_dir="x", train={"seed": 3})
    with pytest.raises(ValueError, match="unknown key"):
        GatesConfig(device="cpu", run_dir="x", e1={"no_such_field": 1})
    base = run_settings(GatesConfig(device="cpu", run_dir="x"))
    assert gate_run.baseline_arm(base), "the default gates job tests the baseline configuration"
