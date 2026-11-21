"""D-039 day weights (owner 2026-11-20): every UTC day the same total weight, the days with a high loss weighted up.
Checks: the per-row loss read from the weight gradient equals the loss of the row alone; the weights balance the days
and follow the day loss; a planted hard day gets weight; resume equals an uninterrupted run; default unchanged; the
refusals."""

import dataclasses

import numpy as np
import pytest
import torch

from nig_timenet_torch.trainer.day_weights import (
    DayWeighter,
    DayWeightSettings,
    day_weights_from_losses,
    make_day_weighter,
    row_losses,
)
from nig_timenet_torch.trainer.identity import build_run_identity
from nig_timenet_torch.trainer.tracker import EarlyStopSettings
from nig_timenet_torch.trainer.trainer import FitSettings, Trainer
from nig_timenet_torch.workflow.settings import RunSettings, SplitSettings, TrainSettings
from tests.trainer.helpers import LIN_CFG, Linear, identity_for, linear_data, make_trainer, state_equal

KEEP_LAST = EarlyStopSettings(restore_best_weights=False)


def _lin_trainer():
    torch.manual_seed(0)
    return Trainer(Linear(x_dim=6), LIN_CFG, optimizer=lambda ps: torch.optim.SGD(ps, lr=0.05))


def test_row_losses_equal_the_loss_of_each_row_alone(data):
    """NIG + class heads with Kendall weights: n * d(objective)/d(w_i) + objective = the objective of row i alone."""
    x, y, _, _ = data
    t = make_trainer()
    t.model.train()                                   # row_losses switches to eval and back
    got = row_losses(t, x[:12], {k: v[:12] for k, v in y.items()}, chunk=5)    # chunks of 5, 5, 2
    assert t.model.training
    t.model.eval()
    with torch.no_grad():
        alone = torch.stack([t.compute_losses(x[i:i + 1], {k: v[i:i + 1] for k, v in y.items()}, plain=True)[0]
                             for i in range(12)]).double()
    assert torch.allclose(got, alone, rtol=1e-5, atol=1e-5)


def test_weights_balance_the_days_and_follow_the_day_loss():
    counts = torch.tensor([10, 40, 50])
    inverse = torch.repeat_interleave(torch.arange(3), counts)
    cfg = DayWeightSettings(strength=0.0, max_ratio=100.0)
    w = day_weights_from_losses(None, counts, inverse, cfg).double()
    totals = torch.zeros(3, dtype=torch.float64).index_add_(0, inverse, w)
    assert torch.allclose(totals, totals.mean().expand(3)) and abs(float(w.mean()) - 1) < 1e-6
    no_balance = day_weights_from_losses(None, counts, inverse, dataclasses.replace(cfg, balance=False))
    assert torch.allclose(no_balance, torch.ones(100))
    hard = day_weights_from_losses(torch.tensor([1.0, 1.0, 3.0], dtype=torch.float64), counts, inverse,
                                   dataclasses.replace(cfg, strength=0.5, balance=False))
    assert hard[-1] > hard[0] == hard[10]             # the high-loss day weighs more, equal days weigh the same
    capped = day_weights_from_losses(torch.tensor([1.0, 1.0, 50.0], dtype=torch.float64), counts, inverse,
                                     DayWeightSettings(strength=5.0, max_ratio=2.0, balance=False))
    assert float(capped.max()) <= 2.0 + 1e-6 and float(capped.min()) >= 0.5 - 1e-6
    assert abs(float(capped.mean()) - 1.0) < 1e-2


def test_update_shrinks_small_days_and_keeps_an_ema():
    days = np.array([0] * 2 + [1] * 50 + [2] * 50)
    wr = DayWeighter(DayWeightSettings(shrink_rows=20, momentum=0.5), days)
    losses = torch.ones(102, dtype=torch.float64)
    losses[:2] = 11.0                                  # a 2-row day with a huge loss: shrunk toward the mean
    wr.update(losses)
    mean = float(losses.mean())
    assert abs(float(wr.last[0]) - (22 + 20 * mean) / 22) < 1e-9
    first = wr.ema.clone()
    wr.update(torch.ones(102, dtype=torch.float64))
    assert torch.allclose(wr.ema, 0.5 * first + 0.5 * wr.last)
    rows = wr.table()
    assert [r["rows"] for r in rows] == [2, 50, 50] and set(rows[0]) == {"day", "rows", "loss_last", "loss_ema",
                                                                         "weight_rel"}


def test_a_planted_hard_day_gets_the_weight():
    """Rows of day 3 carry the opposite label: after the first measurement that day weighs more than the others."""
    x, y, group = linear_data(16 * 30, 0, n_groups=16, bad_groups=(3,))
    xv, yv, _ = linear_data(64, 1)
    t = _lin_trainer()
    res = t.fit(x, y, xv, yv, FitSettings(epochs=3, batch_size=32, early_stop=KEEP_LAST,
                                          day_weights=DayWeightSettings(strength=1.0)), day_groups=group)
    totals = t.day_weighter.day_totals()
    assert int(totals.argmax()) == 3 and float(totals[3] / totals.mean()) > 1.5
    assert res.history[0]["train_day_w_max"] == pytest.approx(1.0)          # epoch 0: balance only (equal days)
    assert res.history[-1]["train_day_w_top10_share"] > 0.1 and 0 < res.history[-1]["train_day_w_ess_share"] < 1


def test_resume_equals_an_uninterrupted_run(tmp_path):
    x, y, group = linear_data(16 * 20, 0, n_groups=16, bad_groups=(5,))
    xv, yv, _ = linear_data(64, 1)
    fs = FitSettings(epochs=4, batch_size=32, early_stop=KEEP_LAST, day_weights=DayWeightSettings(momentum=0.3))
    full = _lin_trainer()
    full.fit(x, y, xv, yv, fs, run_dir=tmp_path / "a", identity=identity_for(fs), day_groups=group)
    cut = _lin_trainer()
    cut.fit(x, y, xv, yv, dataclasses.replace(fs, epochs=2), run_dir=tmp_path / "b", identity=identity_for(fs),
            day_groups=group)
    again = _lin_trainer()
    again.fit(x, y, xv, yv, fs, run_dir=tmp_path / "b", identity=identity_for(fs), day_groups=group)
    assert state_equal(full.model, again.model)
    assert torch.equal(full.day_weighter.ema, again.day_weighter.ema)


def test_default_is_unchanged_and_the_identity_changes_only_when_on():
    x, y, group = linear_data(16 * 10, 0)
    xv, yv, _ = linear_data(64, 1)
    a, b = _lin_trainer(), _lin_trainer()
    a.fit(x, y, xv, yv, FitSettings(epochs=2, batch_size=16, early_stop=KEEP_LAST))
    b.fit(x, y, xv, yv, FitSettings(epochs=2, batch_size=16, early_stop=KEEP_LAST, day_weights=None))
    assert state_equal(a.model, b.model) and a.day_weighter is None
    fs = FitSettings(epochs=2, batch_size=16)
    ident = lambda s: build_run_identity(s, {"m": 1}, LIN_CFG, a.optimizer, "fp", code_sha="0")  # noqa: E731
    assert ident(fs).settings_hash == ident(dataclasses.replace(fs, day_weights=None)).settings_hash
    assert ident(dataclasses.replace(fs, day_weights=DayWeightSettings())).settings_hash != ident(fs).settings_hash


def test_the_checks():
    x, y, group = linear_data(16 * 10, 0)
    xv, yv, _ = linear_data(64, 1)
    on = FitSettings(epochs=1, day_weights=DayWeightSettings())
    with pytest.raises(ValueError, match="needs day_groups"):
        _lin_trainer().fit(x, y, xv, yv, on)
    with pytest.raises(ValueError, match="day_weights is None"):
        _lin_trainer().fit(x, y, xv, yv, FitSettings(epochs=1), day_groups=group)
    with pytest.raises(ValueError, match="ids for"):
        _lin_trainer().fit(x, y, xv, yv, on, day_groups=group[:5])
    with pytest.raises(ValueError, match="shuffled rows"):
        _lin_trainer().fit(x, y, xv, yv, dataclasses.replace(on, batch_by="day"), batch_groups=group,
                           day_groups=group)
    with pytest.raises(ValueError, match="shuffled rows"):
        make_day_weighter(DayWeightSettings(), group, len(group), grouped_batches=True)
    for bad in ({"strength": -1.0}, {"momentum": 1.0}, {"max_ratio": 0.5}, {"every": 0}, {"shrink_rows": -1}):
        with pytest.raises(ValueError, match="day_weights"):
            DayWeightSettings(**bad)
    with pytest.raises(ValueError, match="day_weights needs the flat epoch loop"):
        RunSettings(run_dir="r", train=TrainSettings(day_weights=DayWeightSettings()),
                    split=SplitSettings(cross_section=True))
    with pytest.raises(ValueError, match="day_weights needs the flat epoch loop"):
        RunSettings(run_dir="r", train=TrainSettings(day_weights=DayWeightSettings(), batch_by="day"))
    RunSettings(run_dir="r", train=TrainSettings(day_weights=DayWeightSettings()))
