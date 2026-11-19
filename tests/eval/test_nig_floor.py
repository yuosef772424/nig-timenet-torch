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


def _floor_outputs(model, seq_len=16, n_features=18):
    """The model's own nu / alpha / beta outputs with every head's softplus driven to 0 (raw -> -1e4): the floors in
    the units of the outputs, whatever scale path the model takes."""
    hooks = []
    for head in model.head_modules().values():
        def force(_m, _i, out):
            out = out.clone()
            out[:, 1:4] = -1e4
            return out
        hooks.append(head.out.register_forward_hook(force))
    model.eval()
    with torch.no_grad():
        o = model(torch.randn(4, seq_len, n_features))
    for h in hooks:
        h.remove()
    return o


@pytest.mark.parametrize("over", [{}, {"nig_scale": 0.01}, {"nig_scale_fit": "train_std"},
                                  {"nig_scale_fit": "train_std", "target_norm": "logret_std"},
                                  {"target_norm": "logret_std"}])
def test_floors_are_in_the_units_of_the_outputs(over):
    """Guard (rule B3, the class): for every scale path of the heads, nig_floors equals what the model outputs when
    softplus is 0. Before 2026-11-18 the beta floor was beta_min in head units, so a scaled model showed 100 % of
    rows "on the floor" (health report of notebook-main2)."""
    from nig_timenet_torch.model.builder import build_nig_timenet_v2
    from nig_timenet_torch.model.label_scale import fit_label_scale
    from nig_timenet_torch.model.target_norm import fit_target_norm
    from tests.trainer.fixture_setup import HEAD_TYPES
    model = build_nig_timenet_v2(16, 18, d_model=16, num_layers=1, num_heads=4, num_kv_heads=2, head_hidden=16,
                                 class_head_hidden=8, dropout=0.0, head_types=HEAD_TYPES, **over)
    rng = np.random.default_rng(0)
    y = {spec.label_key: torch.as_tensor(rng.normal(0.01 * (i + 1), 0.03, 500), dtype=torch.float32)
         for i, spec in enumerate(model.head_specs.values())}
    fit_target_norm(model, y)
    fit_label_scale(model, y)
    floors, out = nig_floors(model), _floor_outputs(model)
    for t, fl in floors.items():
        keys = head_spec("nig_regression", t).output_keys
        for p in PARAMS_CHECKED:
            got = out[keys[p]].numpy().ravel()
            np.testing.assert_allclose(got, fl[p], rtol=1e-5, err_msg=f"{over} {t}.{p}")


PARAMS_CHECKED = ("nu", "alpha", "beta")
