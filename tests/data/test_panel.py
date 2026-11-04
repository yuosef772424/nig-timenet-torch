"""Panel grid: windows grouped by time into days x slots, two-dimensional targets, lazy inputs."""

import numpy as np
import pytest

from nig_timenet_torch.data.build import build_dataset
from nig_timenet_torch.data.panel import PanelSettings, build_panel, panel_inputs
from tests.data.helpers import tiny_settings


@pytest.fixture(scope="module")
def ds():
    return build_dataset(tiny_settings())


@pytest.fixture(scope="module")
def ds2():
    return build_dataset(tiny_settings(two_tf=True))


def test_grid_covers_every_row_once_and_packs_slots_left(ds):
    idx = np.arange(len(ds["last_candles"]))
    g = build_panel(ds, idx, PanelSettings(min_group=1))
    assert g.rows.shape == g.mask.shape == (g.n_days, g.n_slots) and g.y_cls.shape == (g.n_days, g.n_slots, 3)
    assert sorted(g.rows[g.mask].tolist()) == idx.tolist()
    assert (g.rows[~g.mask] == -1).all()
    assert all(m[: m.sum()].all() and not m[m.sum():].any() for m in g.mask)          # real coins first
    ts = np.asarray(ds["last_candles"])[:, 3].astype(np.int64)
    for d in range(g.n_days):                                                          # one bucket per day
        assert (ts[g.rows[d][g.mask[d]]] // g.group_ns == g.group[d]).all()
    assert (np.diff(g.group) > 0).all()


def test_relative_targets_are_centred_per_day_and_ranked(ds):
    g = build_panel(ds, np.arange(len(ds["last_candles"])), PanelSettings(min_group=3))
    reg = np.stack([np.asarray(ds[f"y_{t}_reg"]) for t in g.targets], -1)
    for d in range(g.n_days):
        m = g.mask[d]
        centred = reg[g.rows[d][m]] - np.median(reg[g.rows[d][m]], axis=0)
        np.testing.assert_allclose(g.y_reg[d][m] * g.reg_scale, centred, atol=1e-4)
        np.testing.assert_array_equal(g.y_cls[d][m], (centred > 0).astype(np.float32))
        assert (np.abs(g.y_rank[d][m]) < 0.5).all() and abs(g.y_rank[d][m].mean()) < 1e-6
        j = 2
        assert (np.argsort(np.argsort(centred[:, j])) == np.argsort(np.argsort(g.y_rank[d][m][:, j]))).all()
    assert (g.y_reg[~g.mask] == 0).all() and (g.y_cls[~g.mask] == 0).all()
    assert (g.reg_scale > 0).all()


def test_relative_none_keeps_the_datasets_own_targets(ds):
    idx = np.arange(len(ds["last_candles"]))
    g = build_panel(ds, idx, PanelSettings(min_group=1, relative="raw", reg_scale=2.0))
    r = g.rows[g.mask]
    np.testing.assert_allclose(g.y_reg[..., 0][g.mask] * 2.0, np.asarray(ds["y_high_reg"])[r], atol=1e-5)
    np.testing.assert_array_equal(g.y_cls[..., 1][g.mask], np.asarray(ds["y_low_class"])[r])


def test_min_group_max_slots_and_scale_reuse(ds):
    idx = np.arange(len(ds["last_candles"]))
    full = build_panel(ds, idx, PanelSettings(min_group=1))
    big = build_panel(ds, idx, PanelSettings(min_group=3))
    assert big.n_days < full.n_days and big.mask.sum(axis=1).min() >= 3
    capped = build_panel(ds, idx, PanelSettings(min_group=1, max_slots=2))
    assert capped.n_slots == 2 and capped.mask.sum() < full.mask.sum()
    part = build_panel(ds, idx[: len(idx) // 2], PanelSettings(min_group=3), reg_scale=big.reg_scale)
    np.testing.assert_array_equal(part.reg_scale, big.reg_scale)
    mean = build_panel(ds, idx, PanelSettings(min_group=3, relative="mean"))
    assert not np.allclose(mean.y_reg, big.y_reg)


def test_settings_validation_and_missing_stride(ds):
    for bad in ({"relative": "mode"}, {"min_group": 0}, {"max_slots": -1}, {"targets": ()}):
        with pytest.raises(ValueError):
            PanelSettings(**bad)
    no_stride = {k: v for k, v in ds.items() if k != "stride"}
    with pytest.raises(ValueError, match="group_bars"):
        build_panel(no_stride, np.arange(10))
    assert build_panel(no_stride, np.arange(len(ds["last_candles"])), PanelSettings(group_bars=8, min_group=1)).n_days
    with pytest.raises(ValueError, match="reg_scale must be > 0"):
        build_panel(ds, np.arange(2), PanelSettings(min_group=1, relative="median"))


def test_lazy_inputs_read_only_the_asked_days(ds):
    g = build_panel(ds, np.arange(len(ds["last_candles"])), PanelSettings(min_group=3))
    inp = panel_inputs(ds, g)
    assert len(inp) == g.n_days
    batch = inp[np.array([2, 0])]
    assert batch["x"].shape == (2, g.n_slots, *ds["X_1h"].shape[1:]) and batch["mask"].shape == (2, g.n_slots)
    for b, d in enumerate((2, 0)):
        m = g.mask[d]
        np.testing.assert_array_equal(batch["x"][b][m], ds["X_1h"][g.rows[d][m]])
        assert (batch["x"][b][~m] == 0).all()
    assert inp[slice(1, 3)]["x"].shape[0] == 2 and inp[5]["x"].shape[0] == 1


def test_two_timeframes_give_a_dict_per_timeframe(ds2):
    g = build_panel(ds2, np.arange(len(ds2["last_candles"])), PanelSettings(min_group=3))
    x = panel_inputs(ds2, g)[np.array([0, 1])]["x"]
    assert set(x) == {"1h", "4h"} and x["4h"].shape[:2] == x["1h"].shape[:2]
