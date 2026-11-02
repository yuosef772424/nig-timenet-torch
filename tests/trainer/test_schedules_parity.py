"""T1: unit parity of the lambda / LR schedules against values computed by EXECUTING the TF source
(reference/export_schedules.py -> reference/data_schedules/schedules.json; trainer/schedules.py is identical at the
port base aeafc564 and at cea9cd58). float64 pure arithmetic: compared at rtol 1e-12."""

import json
from pathlib import Path

import pytest

from nig_timenet_torch.trainer.schedules import (
    LrSettings,
    ParamSchedule,
    apply_schedules,
    cosine_warm_restarts,
    lr_at,
    schedule_value,
)

REF = json.loads((Path(__file__).resolve().parents[2] / "reference" / "data_schedules" / "schedules.json")
                 .read_text(encoding="utf-8"))
RTOL = 1e-12


def test_reference_file_records_its_source():
    assert REF["source_repo"] == "yuosef772424/crypto-signal-prediction"
    assert REF["source_file"] == "trainer/schedules.py"
    assert len(REF["source_sha256"]) == 64 and "aeafc564f25c78f252936affbd374f68aa49b8d9" in REF["source_commits"]


def _case_id(c):
    return f"{c['schedule']}-{c['start']}-{c['end']}-{c['warmup_epochs']}"


@pytest.mark.parametrize("case", REF["schedule_value"], ids=_case_id)
def test_schedule_value_matches_tf(case):
    p = ParamSchedule(case["start"], case["end"], case["warmup_epochs"], case["schedule"])
    got = [schedule_value(p, ep) for ep in range(len(case["values"]))]
    assert got == pytest.approx(case["values"], rel=RTOL, abs=0)


@pytest.mark.parametrize("case", REF["cosine_warm_restarts"], ids=lambda c: f"{c['cycle_length']}x{c['cycle_mult']}")
def test_cosine_warm_restarts_matches_tf(case):
    got = [cosine_warm_restarts(ep, case["lr_initial"], case["lr_min"], case["cycle_length"], case["cycle_mult"])
           for ep in range(len(case["values"]))]
    assert got == pytest.approx(case["values"], rel=RTOL, abs=0)


def _lr_settings(opt_cfg: dict) -> LrSettings:
    """TF opt_cfg -> LrSettings; absent TF keys are left out so the dataclass defaults (= TF .get defaults) apply."""
    sched = opt_cfg.get("lr_schedule", {})
    kw = {"lr_initial": opt_cfg["lr_initial"], "lr_min": opt_cfg["lr_min"]}
    if "lr_warmup_epochs" in opt_cfg:
        kw["warmup_epochs"] = opt_cfg["lr_warmup_epochs"]
    for tf_key, key in (("type", "kind"), ("cycle_length", "cycle_length"), ("cycle_mult", "cycle_mult"),
                        ("total_epochs", "total_epochs")):
        if tf_key in sched:
            kw[key] = sched[tf_key]
    return LrSettings(**kw)


@pytest.mark.parametrize("name", sorted(REF["lr_schedule_fn"]))
def test_lr_schedule_matches_tf(name):
    case = REF["lr_schedule_fn"][name]
    s = _lr_settings(case["opt_cfg"])
    assert [lr_at(s, ep) for ep in range(len(case["values"]))] == pytest.approx(case["values"], rel=RTOL, abs=0)


def test_apply_schedules_sets_every_variable_in_place():
    scheduled = {"lambda_reg": 0.0, "lambda_calib": 0.0, "other": 5.0}
    schedules = {"lambda_reg": ParamSchedule(0.0, 0.05, 5, "linear"),
                 "lambda_calib": ParamSchedule(0.0, 0.1, 5, "cosine")}
    apply_schedules(scheduled, schedules, 0)
    assert scheduled == {"lambda_reg": 0.0, "lambda_calib": 0.0, "other": 5.0}
    apply_schedules(scheduled, schedules, 2)
    assert scheduled["lambda_reg"] == pytest.approx(0.02) and scheduled["other"] == 5.0
    apply_schedules(scheduled, schedules, 5)
    assert (scheduled["lambda_reg"], scheduled["lambda_calib"]) == (0.05, 0.1)


def test_unknown_names_and_bad_values_raise():
    """Deliberate difference from TF (rule B5): no silent fallback to linear / constant."""
    with pytest.raises(ValueError, match="unknown schedule 'cosin'"):
        ParamSchedule(0.0, 1.0, 5, "cosin")
    with pytest.raises(ValueError, match="unknown lr kind"):
        LrSettings(1e-3, 1e-5, kind="cosine_restart")
    with pytest.raises(ValueError, match="warmup_epochs"):
        ParamSchedule(0.0, 1.0, -1)
    with pytest.raises(ValueError, match="cycle_mult"):
        LrSettings(1e-3, 1e-5, kind="cosine_restarts", cycle_mult=0.5)
