"""T4 NIG floor-hit shares: known answer on a synthetic table, floors read from the model's NIGHead modules."""

import numpy as np
import pandas as pd
import pytest
import torch

from nig_timenet_torch.eval.nig_floor import nig_floor_hit_report, nig_floors
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.model.nig_layers import NIGHead
from tests.trainer.fixture_setup import build_model


def test_floors_are_read_from_the_model():
    floors = nig_floors(build_model(16, 18))
    assert set(floors) == {"high", "low", "close"}
    assert floors["close"] == {"nu": 0.1, "alpha": 2.0, "beta": 0.01}
    with pytest.raises(ValueError):
        nig_floors(torch.nn.Linear(2, 2))

    class Bad(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = NIGHead(4)
    with pytest.raises(ValueError):
        nig_floors(Bad())


def test_share_at_floor_known_answer():
    keys = head_spec("nig_regression", "close").output_keys
    n = 1000
    nu = np.where(np.arange(n) < 300, 0.1 + 1e-6, 0.5)        # 30 % on the floor
    alpha = np.full(n, 2.0 + 5.0)                             # none
    beta = np.where(np.arange(n) < 900, 0.01, 0.2)            # 90 %
    table = pd.DataFrame({keys["nu"]: nu, keys["alpha"]: alpha, keys["beta"]: beta})
    r = nig_floor_hit_report(table, {"close": {"nu": 0.1, "alpha": 2.0, "beta": 0.01}})["close"]
    assert r["nu"]["share_at_floor"] == pytest.approx(0.3) and r["alpha"]["share_at_floor"] == 0.0
    assert r["beta"]["share_at_floor"] == pytest.approx(0.9) and r["beta"]["min"] == 0.01
    with pytest.raises(KeyError):
        nig_floor_hit_report(table.drop(columns=[keys["nu"]]), {"close": {"nu": 0.1, "alpha": 2.0, "beta": 0.01}})
    with pytest.raises(ValueError):
        nig_floor_hit_report(table, {"close": {"nu": 0.0, "alpha": 2.0, "beta": 0.01}})
