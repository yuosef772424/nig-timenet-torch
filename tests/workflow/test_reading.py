"""workflow/reading.py helpers: the evidence rule, coins from asset_bounds, the extended battery table."""

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.workflow import reading


def row(use, control, split, avg, lo):
    return {"use": use, "control": control, "split": split, "n": 100, "avg": avg, "ci_lo": lo, "ci_hi": avg + 1}


def test_evidence_needs_both_splits_every_control_and_the_same_sign():
    rows = [row("a", "model", "VAL", 0.2, 0.1), row("a", "shuffled", "VAL", 0.1, -0.1),
            row("a", "model", "TEST", 0.3, 0.05), row("a", "shuffled", "TEST", 0.0, -0.2),
            row("b", "model", "VAL", -0.1, -0.2), row("b", "model", "TEST", 0.4, 0.1),
            row("c", "model", "VAL", 0.2, 0.1), row("c", "random", "VAL", 0.3, 0.2),
            row("c", "model", "TEST", 0.2, 0.1)]
    ev = {e["use"]: e for e in reading.evidence(rows)}
    assert ev["a"]["evidence"] and not ev["b"]["evidence"] and not ev["b"]["same_sign"]
    assert not ev["c"]["evidence"] and not ev["c"]["beats_controls"]


def test_row_coins_and_extended_table():
    ds = {"asset_bounds": [{"name": "B", "start": 3, "end": 5}, {"name": "A", "start": 0, "end": 3}]}
    assert list(reading.row_coins(ds, np.array([0, 2, 3, 4]))) == ["A", "A", "B", "B"]
    bt = pd.DataFrame({"pred_high": [0.02], "pred_low": [-0.01]})
    tab = pd.DataFrame({"y_high": [0.02], "y_high_aleatoric": [0.01], "y_low": [-0.01], "y_low_aleatoric": [0.02]})
    out = reading.extended_table(bt, tab, {"atr14": np.array([0.01])}, np.array([0.0]))
    assert out["ale_high"].iloc[0] == pytest.approx(0.01) and out["vol"].iloc[0] == pytest.approx(np.expm1(0.01))
    assert out["last_ret"].iloc[0] == 0.0
