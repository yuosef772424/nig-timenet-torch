"""doctor/readiness.py (card E): the diagnosis reading order and the command line entry."""

import json

import pytest

from nig_timenet_torch.doctor import readiness
from nig_timenet_torch.doctor.common import Check


def test_diagnosis_orders_data_before_trainer_before_model():
    checks = [
        Check("A", "model/trunk", "y_close: probe IC at the trunk", 0.0, 0.2, "fail"),
        Check("B", "trainer/optimisation", "overfit one batch (no regularisation): worst head corr", 0.1, 0.9, "fail"),
        Check("C", "data/alignment", "high label == the next bars", 1.0, 2e-5, "fail"),
        Check("D", "profit/val", "GOAL: any use of the outputs profitable after cost", "no", "", "fail"),
    ]
    lines = readiness.diagnosis(checks)
    assert lines[0].startswith("1. DATA") and lines[1].startswith("2. TRAINER") and lines[2].startswith("3. MODEL")
    assert "DOES NOT LEARN" in lines[1] and lines[3].startswith("4. GOAL") and "NOT met" in lines[3]


def test_diagnosis_with_nothing_failing():
    assert readiness.diagnosis([Check("C", "data/alignment", "x", 0, "", "pass")]) == ["No failing check."]


@pytest.mark.slow
def test_main_writes_the_report_files(tmp_path):
    assert readiness.main(["e1", "--out", str(tmp_path), "--cards", "C", "--epochs", "1", "--batch-size", "128"]) == 0
    md = (tmp_path / "readiness.md").read_text(encoding="utf-8")
    assert "# Readiness report" in md and "## Diagnosis" in md
    rows = json.loads((tmp_path / "readiness.json").read_text(encoding="utf-8"))
    assert rows and {r["card"] for r in rows} == {"C"}
