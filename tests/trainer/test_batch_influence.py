"""In-run batch influence (trainer/batch_influence.py) and its report (eval/influence_report.py): the monitor leaves the
training bitwise unchanged, the step deltas telescope to the total probe change, the first-order estimate follows the
exact delta, and planted harmful data are found (known answer) while clean groups are not flagged."""

import numpy as np
import pandas as pd
import pytest
import torch

from nig_timenet_torch.eval.influence_report import (
    group_attribution,
    influence_summary,
    split_half_reliability,
    top_steps,
    write_influence_report,
)
from nig_timenet_torch.model.architectures import ModelSettings
from nig_timenet_torch.trainer.batch_influence import BatchInfluence, InfluenceSettings, probe_rows
from nig_timenet_torch.trainer.tracker import EarlyStopSettings
from nig_timenet_torch.trainer.trainer import FitSettings, Trainer
from nig_timenet_torch.workflow import flat_run
from nig_timenet_torch.workflow.settings import RunSettings
from tests.trainer.helpers import LIN_CFG, Linear, linear_data

TARGETS = ("high", "low", "close")
T, F = 8, 3
KEEP_LAST = EarlyStopSettings(restore_best_weights=False)


def _nig_data(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, T, F, generator=g)
    vol = 0.01 * (1 + x[:, -1, 0].abs())
    y = {"y_high_reg": vol + 0.002 * torch.rand(n, generator=g), "y_low_reg": -vol - 0.002 * torch.rand(n, generator=g),
         "y_close_reg": 0.01 * x[:, -1, 2] + 0.003 * torch.randn(n, generator=g)}
    y.update({f"y_{t}_class": (y[f"y_{t}_reg"] > 0).float() for t in TARGETS})
    return x, y


def _nig_trainer():
    torch.manual_seed(0)
    ms = ModelSettings("tiny_mlp", {})
    return flat_run.make_trainer(RunSettings(run_dir="x", model=ms), ms, TARGETS, T, F, torch.device("cpu"))


def _lin_data(n, seed, n_groups=16, bad_groups=()):
    return linear_data(n, seed, n_groups, bad_groups)


def _lin_trainer():
    torch.manual_seed(0)
    return Trainer(Linear(), LIN_CFG, optimizer=lambda ps: torch.optim.SGD(ps, lr=0.05))


@pytest.fixture(scope="module")
def planted():
    x, y, group = _lin_data(2048, 0, bad_groups=(0,))
    xv, yv, _ = _lin_data(512, 1)
    t = _lin_trainer()
    m = BatchInfluence({"val": (xv, yv), "train": (x, y)}, InfluenceSettings(probe_rows=512, first_order_every=1))
    t.fit(x, y, xv, yv, FitSettings(epochs=8, batch_size=32, early_stop=KEEP_LAST), batch_monitor=m)
    return m, group


def test_the_monitor_leaves_the_training_unchanged():
    x, y = _nig_data(512, 0)
    xv, yv = _nig_data(256, 1)
    fs = FitSettings(epochs=2, batch_size=64, grad_clip_norm=1.0, early_stop=KEEP_LAST)
    plain, watched = _nig_trainer(), _nig_trainer()
    plain_res = plain.fit(x, y, xv, yv, fs)
    m = BatchInfluence({"val": (xv, yv), "train": (x, y)}, InfluenceSettings(probe_rows=128, first_order_every=2))
    watched_res = watched.fit(x, y, xv, yv, fs, batch_monitor=m)
    sa, sb = plain.model.state_dict(), watched.model.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert plain_res.history == watched_res.history
    f = m.frame()
    assert len(f) == 2 * 8 and f["fo_delta"].notna().sum() == 8          # first-order on every 2nd step
    assert {"delta_close_reg", "train_delta", "grad_alignment", "fo_nig"} <= set(f.columns)


def test_the_step_deltas_telescope_to_the_total_change(planted):
    m, _ = planted
    f = m.frame()
    assert f["delta"].sum() == pytest.approx(m._last["val"]["loss"] - m.start_losses["val"], abs=1e-5)
    assert f["delta"].sum() < 0                                           # training on mostly clean data helps


def test_first_order_follows_the_exact_delta_and_the_halves_agree(planted):
    s = influence_summary(planted[0].frame())
    assert s["first_order"]["r_vs_exact"] > 0.95
    assert s["split_half"]["detrended_r"] > 0.8 and s["split_half"]["reliability"] > 0.9
    assert s["modules"] and "fc" in s["modules"]


def test_planted_harmful_group_is_the_only_flagged_one(planted):
    m, group = planted
    steps, rows = m.rows_of_steps()
    g = group_attribution(m.frame(), steps, rows, group, n_perm=300)
    flagged = g[g["flag"] != ""]
    assert list(flagged["group"]) == [0] and flagged["flag"].iloc[0] == "harmful"
    assert g.loc[g["group"] == 0, "p_perm"].iloc[0] <= 2 / 301


def test_known_answer_a_bad_batch_hurts_and_a_clean_one_helps():
    x, y, _ = _lin_data(4096, 0)
    xv, yv, _ = _lin_data(512, 1)
    perm = torch.randperm(4096, generator=torch.Generator().manual_seed(3))
    t = _lin_trainer()
    for i in range(10):                      # part of the way: clean steps still have something to teach
        ix = torch.arange(i * 32, (i + 1) * 32)
        t.train_step(x[ix], {"y": y["y"][ix]})
    m = BatchInfluence({"val": (xv, yv)}, InfluenceSettings(probe_rows=512, first_order_every=1))
    m.start(t)
    bad = np.arange(40) % 2 == 1
    for i, is_bad in enumerate(bad):
        ix = torch.arange(1000 + i * 32, 1000 + (i + 1) * 32)
        label = y["y"][perm[ix]] if is_bad else y["y"][ix]      # a bad batch has its labels shuffled
        m.before_step(t, 0, ix)
        m.after_step(t, t.train_step(x[ix], {"y": label}))
    f = m.frame()
    assert (f["delta"][bad] > 0).mean() >= 0.8 and (f["delta"][~bad] < 0).mean() >= 0.8
    assert f["grad_alignment"][bad].mean() < f["grad_alignment"][~bad].mean()


def test_no_group_is_flagged_without_a_planted_effect():
    x, y, group = _lin_data(2048, 0)
    xv, yv, _ = _lin_data(512, 1)
    t = _lin_trainer()
    m = BatchInfluence({"val": (xv, yv)}, InfluenceSettings(probe_rows=256))
    t.fit(x, y, xv, yv, FitSettings(epochs=4, batch_size=32, early_stop=KEEP_LAST), batch_monitor=m)
    steps, rows = m.rows_of_steps()
    assert (group_attribution(m.frame(), steps, rows, group, n_perm=200)["flag"] == "").all()


def test_report_files_and_top_steps(planted, tmp_path):
    m, group = planted
    steps, rows = m.rows_of_steps()
    res = write_influence_report(tmp_path, m.frame(), steps, rows, group, n_perm=100)
    md = (tmp_path / "batch_influence_report.md").read_text()
    assert "split-half r" in md and "## Groups" in md and "## Probe `train`" in md
    assert (tmp_path / "batch_influence_report.json").exists() and res["n_steps"] == len(m.frame())
    top = top_steps(m.frame(), steps, rows, group, k=3)
    assert list(top["kind"]) == ["harmful"] * 3 + ["helpful"] * 3 and (top["n_groups"] > 0).all()
    m.save(tmp_path / "rec")
    back = np.load(tmp_path / "rec" / "batch_rows.npz")
    assert len(back["row"]) == 2048 * 8 and len(pd.read_csv(tmp_path / "rec" / "batch_influence.csv")) == len(m.frame())


def test_probe_rows_halves_are_disjoint_and_grouped_draws_keep_whole_groups():
    a, b = probe_rows(100, 40, seed=0)
    assert len(a) + len(b) == 40 and not set(a) & set(b)
    groups = np.repeat(np.arange(20), 5)
    a, b = probe_rows(100, 30, seed=0, groups=groups)
    assert not set(groups[a]) & set(groups[b])
    for half in (a, b):
        assert all((groups[half] == g).sum() == 5 for g in set(groups[half]))


def test_guards():
    assert split_half_reliability(0.5) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        InfluenceSettings(probe_rows=2)
    x, y = _nig_data(64, 0)
    m = BatchInfluence({"val": (x, y)}, InfluenceSettings(probe_rows=32))
    with pytest.raises(RuntimeError, match="start"):
        m.before_step(_nig_trainer(), 0, torch.arange(4))
    with pytest.raises(ValueError):
        influence_summary(pd.DataFrame())
