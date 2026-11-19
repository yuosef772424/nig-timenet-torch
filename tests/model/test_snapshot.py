"""The snapshot panel model (owner 2026-11-18, "100 x 3"): the moment as one input, per-coin weights."""

import numpy as np
import pytest
import torch

from nig_timenet_torch.core.config import updated
from nig_timenet_torch.data_io.synthetic_panel import PanelSyntheticSettings, write_panel_synthetic
from nig_timenet_torch.eval.nig_floor import nig_floors
from nig_timenet_torch.model.architectures import ModelSettings, build_model
from nig_timenet_torch.model.scalers import fit_scalers
from nig_timenet_torch.model.snapshot import CoinSlots, SnapshotPanelModel
from nig_timenet_torch.workflow.settings import RunSettings, SplitSettings

T, NF = 16, 3


def _x(n_moments=3, n_coins=5, seed=0, coins=None):
    """Rows of ``n_moments`` whole timestamps: [features (NF), coin id, timestamp id], time-major."""
    g = torch.Generator().manual_seed(seed)
    coins = list(range(n_coins)) if coins is None else coins
    rows = []
    for m in range(n_moments):
        for c in coins:
            f = torch.randn(T, NF, generator=g) * 0.02
            rows.append(torch.cat([f, torch.full((T, 1), float(c)), torch.full((T, 1), float(m))], dim=-1))
    return torch.stack(rows)


def _model(**over):
    torch.manual_seed(0)
    m = SnapshotPanelModel(T, NF + 1, n_slots=8, max_assets=64, d_joint=32, d_coin=16, d_emb=8, d_read=16,
                           head_hidden=16, class_head_hidden=8, **over)
    x = _x()
    y = {spec.label_key: torch.randn(len(x)) * 0.03 for spec in m.head_specs.values()}
    fit_scalers(m, x[..., :-1], y)
    with torch.no_grad():   # give the per-coin output weights a value (they start at 0)
        m.slot_out.normal_(0, 0.1) if m.joint_path else None
    return m.eval()


def test_outputs_match_the_default_architecture_keys_and_floors_are_read():
    m = _model()
    out = m(_x())
    ref = build_model(ModelSettings("nig_timenet_v2", {"d_model": 16}), T, NF)(torch.zeros(2, T, NF))
    assert set(out) == set(ref)
    assert all(v.shape[0] == 15 and torch.isfinite(v).all() for v in out.values())
    assert set(nig_floors(m)) == {"high", "low", "close"}


def test_rows_order_does_not_matter_and_moments_do_not_leak():
    m, x = _model(), _x(n_moments=3)
    out = m(x)
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(1))
    out_p = m(x[perm])
    for k in out:
        torch.testing.assert_close(out_p[k], out[k][perm], rtol=1e-5, atol=1e-6)
    alone = m(x[:5])                       # moment 0 alone gives the same outputs as inside the batch of 3 moments
    for k in out:
        torch.testing.assert_close(alone[k], out[k][:5], rtol=1e-5, atol=1e-6)


def test_a_coins_output_depends_on_the_other_coins_of_its_moment_through_the_joint_path():
    x = _x(n_moments=1)
    y = x.clone()
    y[1, :, :NF] += 1.0                    # change coin 1 only
    joint = _model(shared_path=False)
    assert not torch.allclose(joint(x)["y_close"][0], joint(y)["y_close"][0])
    shared = _model(joint_path=False)      # the shared path sees the others only through the relative inputs
    assert torch.isfinite(shared(y)["y_close"]).all()


def test_coin_slots_fit_on_train_counts_and_unknown_coins_get_the_shared_slot():
    s = CoinSlots(2, max_assets=10)
    x = _x(n_moments=1, coins=[3, 3, 3, 7, 7, 1])
    s.fit_from_inputs(x[..., :-1])
    assert s(torch.tensor([3, 7, 1, 9])).tolist() == [0, 1, 2, 2]   # 1 and 9: no slot -> unknown (2)
    with pytest.raises(RuntimeError):
        CoinSlots(2)(torch.tensor([0]))
    m = _model()
    out = m(_x(n_moments=1, coins=[0, 1, 50]))  # coin 50 was not in train: it still gets outputs
    assert torch.isfinite(out["y_close"]).all()


def test_paths_and_settings_are_checked():
    with pytest.raises(ValueError):
        SnapshotPanelModel(T, NF + 1, joint_path=False, shared_path=False)
    with pytest.raises(ValueError):
        SnapshotPanelModel(T, 1)
    with pytest.raises(ValueError):
        SplitSettings(coin_channel=True)
    assert float(SnapshotPanelModel(T, NF + 1, n_slots=4, max_assets=8).slot_out.abs().sum()) == 0.0


def test_the_run_steps_train_and_test_the_snapshot_model_on_a_tiny_panel(tmp_path):
    from nig_timenet_torch.workflow import run as steps
    art = tmp_path / "e4"
    write_panel_synthetic(art, PanelSyntheticSettings(n_coins=6, n_bars=400, burn_in_bars=64), source_commit="0" * 40)
    s = updated(RunSettings(run_dir=str(tmp_path / "rd")), {
        "data.name": str(art), "train.device": "cpu", "train.epochs": 1, "train.batch_size": 60,
        "split.cross_section": True, "split.coin_channel": True, "train.cross_section_loss": "rank_ic",
        "evaluation.battery.enabled": False})
    s = updated(s, {"model": ModelSettings("snapshot_panel", {"n_slots": 8, "d_joint": 16, "d_coin": 8, "d_emb": 4,
                                                              "d_read": 8, "head_hidden": 8, "class_head_hidden": 4})})
    run = steps.prepare(s, tmp_path / "out")
    steps.build(run)
    assert run.extra["fitted_scalers"]["coin_slots"]["coins_with_slot"] == 6
    steps.train(run)
    rep = steps.test_step(run)
    assert rep["n_rows"] > 0
    assert np.isfinite(rep["targets"]["close"]["skill"]["rank_ic_by_timestamp"])
