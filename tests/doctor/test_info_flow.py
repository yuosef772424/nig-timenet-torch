"""doctor/info_flow.py (card A): the stage table follows the forward order and a PLANTED loss is the stage it names."""

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.doctor.common import probe_scores, row_times, time_folds
from nig_timenet_torch.doctor.info_flow import collect_streams, destroy_rows, info_flow, stage_table, tracers


@pytest.fixture(scope="module")
def flow(ctx):
    return info_flow(ctx)


@pytest.mark.slow
def test_info_flow_stages_in_forward_order_and_self_check_passes(flow):
    checks, table = flow
    stages = list(table["stage"])
    assert stages[:2] == ["input", "scaled"] or stages[0] == "input"
    order = ["input", "inst_norm", "enc_in", "enc_out", "pooled", "readout", "linear_path", "branch_out", "trunk"]
    assert [stages.index(s) for s in order] == sorted(stages.index(s) for s in order)
    assert any(s.startswith("head_") for s in stages)
    assert {"vol_level", "last_move", "drift"} <= set(table.columns)
    selfcheck = [c for c in checks if c.stage == "instrument"]
    assert len(selfcheck) == 1 and selfcheck[0].verdict == "pass", selfcheck


def test_a_planted_loss_is_found_at_the_planted_stage(ctx):
    ds, model = ctx.dataset, ctx.model
    idx, x, _ = ctx.xy("val", 1500)
    folds = time_folds(row_times(ds, idx), 4, embargo=int(pd.Timedelta(ds["base_tf"]).value) * 40)
    sig = {"vol_level": tracers(x.numpy(), list(ds["feature_order"]))["vol_level"]}
    metric = {"vol_level": "r2"}
    base_acts, _ = collect_streams(model, x)
    acts, _ = collect_streams(model, x, edit={"branch_out": destroy_rows})
    clean = stage_table(base_acts, sig, folds, metric).set_index("stage")
    hurt = stage_table(acts, sig, folds, metric).set_index("stage")
    assert clean.loc["branch_out", "vol_level"] > 0.4                         # the tracer is readable before the loss
    assert hurt.loc["branch_out", "d_vol_level"] < -0.5 * clean.loc["branch_out", "vol_level"]
    assert hurt.loc["trunk", "vol_level"] < 0.1                               # ... and gone at the trunk
    assert clean.loc["trunk", "vol_level"] > 0.4
    for stage in ("input", "inst_norm", "enc_in", "enc_out", "pooled", "readout", "linear_path"):
        assert hurt.loc[stage, "vol_level"] == pytest.approx(clean.loc[stage, "vol_level"], abs=1e-9)


def test_tracers_are_known_functions_of_the_window():
    x = np.random.default_rng(0).normal(size=(20, 8, 3)) * 0.01
    x[:, :, 0], x[:, :, 1] = np.abs(x[:, :, 0]) + 0.01, -np.abs(x[:, :, 1]) - 0.01      # high > low on every bar
    tr = tracers(x, ["high", "low", "close"])
    assert set(tr) == {"vol_level", "last_move", "drift"}
    np.testing.assert_allclose(tr["last_move"], x[:, -1, 2])
    np.testing.assert_allclose(tr["drift"], x[:, :, 2].mean(axis=1))
    np.testing.assert_allclose(tr["vol_level"], np.log((x[:, :, 0] - x[:, :, 1]).mean(axis=1)), equal_nan=True)
    assert probe_scores(x.reshape(20, -1), {"d": tr["drift"]}, time_folds(np.arange(20), 2))["d"] is not None
