"""D-034 batch_by: one batch = all training rows of one timestamp / one UTC day, the batches shuffled (not the rows);
"rows" (the default) is unchanged, and the run identity is unchanged at the default."""

import dataclasses

import numpy as np
import pytest
import torch
from torch import nn

from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.trainer.batch_influence import BatchInfluence, InfluenceSettings
from nig_timenet_torch.trainer.identity import build_run_identity
from nig_timenet_torch.trainer.tracker import EarlyStopSettings
from nig_timenet_torch.trainer.trainer import FitSettings, Trainer
from nig_timenet_torch.workflow import flat_run
from nig_timenet_torch.workflow.settings import RunSettings, SplitSettings, TrainSettings
from tests.trainer.helpers import LIN_CFG, Linear, linear_data

KEEP_LAST = EarlyStopSettings(restore_best_weights=False)
N_TIMES, COINS = 24, 10


def _data():
    x, y, _ = linear_data(N_TIMES * COINS, 0)
    times = np.repeat(np.arange(N_TIMES), COINS)[np.random.default_rng(0).permutation(N_TIMES * COINS)]
    return x, y, torch.as_tensor(times)


def _trainer():
    torch.manual_seed(0)
    return Trainer(Linear(x_dim=6), LIN_CFG, optimizer=lambda ps: torch.optim.SGD(ps, lr=0.05))


def test_every_batch_is_one_whole_group_and_the_group_order_is_shuffled():
    x, y, times = _data()
    xv, yv, _ = linear_data(64, 1)
    orders = []
    for epochs in (2,):
        t = _trainer()
        m = BatchInfluence({"val": (xv, yv)}, InfluenceSettings(probe_rows=32))
        t.fit(x, y, xv, yv, FitSettings(epochs=epochs, batch_size=7, batch_by="timestamp", early_stop=KEEP_LAST),
              batch_monitor=m, batch_groups=times)
        assert len(m.batch_rows) == 2 * N_TIMES                    # one step per timestamp per epoch
        for rows in m.batch_rows:
            assert len(np.unique(times.numpy()[rows])) == 1 and len(rows) == COINS     # all rows of ONE time
        per_epoch = [m.batch_rows[:N_TIMES], m.batch_rows[N_TIMES:]]
        for ep in per_epoch:
            assert sorted(np.concatenate(ep)) == list(range(len(x)))                   # every row once per epoch
        orders = [[int(times[r[0]]) for r in ep] for ep in per_epoch]
    assert orders[0] != orders[1] and orders[0] != sorted(orders[0])                   # the groups are shuffled


def test_rows_default_is_unchanged_and_resume_identity_is_kept():
    x, y, _ = _data()
    xv, yv, _ = linear_data(64, 1)
    a, b = _trainer(), _trainer()
    a.fit(x, y, xv, yv, FitSettings(epochs=2, batch_size=16, early_stop=KEEP_LAST))
    b.fit(x, y, xv, yv, FitSettings(epochs=2, batch_size=16, batch_by="rows", early_stop=KEEP_LAST))
    assert all(torch.equal(p, q) for p, q in zip(a.model.state_dict().values(), b.model.state_dict().values(),
                                                    strict=True))
    fs = FitSettings(epochs=2, batch_size=16)
    ident = lambda s: build_run_identity(s, {"m": 1}, LIN_CFG, a.optimizer, "fp", code_sha="0")  # noqa: E731
    assert ident(dataclasses.replace(fs, batch_by="day")).settings_hash != ident(fs).settings_hash


def test_the_checks():
    x, y, times = _data()
    xv, yv, _ = linear_data(64, 1)
    with pytest.raises(ValueError, match="batch_by must be one of"):
        FitSettings(epochs=1, batch_by="week")
    with pytest.raises(ValueError, match="needs batch_groups"):
        _trainer().fit(x, y, xv, yv, FitSettings(epochs=1, batch_by="day"))
    with pytest.raises(ValueError, match="batch_by is 'rows'"):
        _trainer().fit(x, y, xv, yv, FitSettings(epochs=1), batch_groups=times)
    with pytest.raises(ValueError, match="ids for"):
        _trainer().fit(x, y, xv, yv, FitSettings(epochs=1, batch_by="day"), batch_groups=times[:5])
    t = _trainer()
    t.group_channel = True
    with pytest.raises(ValueError, match="cross-section"):
        t.fit(x, y, xv, yv, FitSettings(epochs=1, batch_by="day"), batch_groups=times)
    with pytest.raises(ValueError, match="batch_by must be one of"):
        TrainSettings(batch_by="hour")
    with pytest.raises(ValueError, match="cross_section"):
        RunSettings(run_dir="r", train=TrainSettings(batch_by="day"), split=SplitSettings(cross_section=True))


def test_batch_keys_of_the_dataset():
    hour = 3_600_000_000_000
    ts = np.array([0, 4, 8, 20, 24, 28]) * hour + 1_700_006_400_000_000_000       # 2023-11-15 00:00 UTC onwards
    lc = np.zeros((6, TS_COL + 1))
    lc[:, TS_COL] = ts
    ds = {"last_candles": lc}
    assert flat_run.batch_groups(ds, np.arange(6), "rows") is None
    assert len(torch.unique(flat_run.batch_groups(ds, np.arange(6), "timestamp"))) == 6
    day = flat_run.batch_groups(ds, np.arange(6), "day")
    assert day.tolist()[:4] == [day[0]] * 4 and day[4] == day[5] != day[0]           # 00 / 04 / 08 / 20 h, next day


def test_grouping_reaches_the_monitor_unchanged(tmp_path):
    """A model that is not linear: batch_by works through any model (the model never sees the group id)."""
    x, y, times = _data()
    xv, yv, _ = linear_data(64, 1)
    torch.manual_seed(0)
    t = Trainer(nn.Sequential(nn.Linear(6, 4), nn.Tanh(), Linear(x_dim=4)), LIN_CFG,
                optimizer=lambda ps: torch.optim.Adam(ps, lr=1e-2))
    res = t.fit(x, y, xv, yv, FitSettings(epochs=3, batch_by="day", early_stop=KEEP_LAST), batch_groups=times)
    assert res.epochs_done == 3 and res.history[-1]["val_raw_loss"] < res.history[0]["val_raw_loss"]
