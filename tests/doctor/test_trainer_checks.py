"""doctor/trainer_checks.py (card B): the selection regret, the verdict mapping and the instruments on a small E1."""

import numpy as np
import pytest

from nig_timenet_torch.doctor.common import Check
from nig_timenet_torch.doctor.trainer_checks import (
    lr_range_test,
    mode_gap,
    overfit_one_batch,
    resume_check,
    selection_check,
    verdict,
)

TARGETS = ("high", "low", "close")


def _history(maes, sel):
    return [{**{f"val_{t}_reg_mae": m for t in TARGETS},
             "val_selection_metric": s} for m, s in zip(maes, sel, strict=True)]


def test_selection_regret_of_a_worse_pick_and_of_the_best_epoch():
    hist = _history([0.05, 0.04, 0.03, 0.035], [1.0, 0.9, 0.95, 0.8])        # MAE best at epoch 3, metric at epoch 4
    worse = selection_check(hist, 4, TARGETS)
    assert worse["best_by_mae"] == 3 and worse["best_by_raw_selection"] == 4
    assert worse["regret"] == pytest.approx(0.035 / 0.03 - 1.0)
    assert worse["regret"] > 0
    assert selection_check(hist, 3, TARGETS)["regret"] == pytest.approx(0.0)


def test_selection_check_without_history_is_nan():
    out = selection_check([], None, TARGETS)
    assert np.isnan(out["regret"]) and out["epochs"] == 0


def test_verdict_mapping():
    def fail(name):
        return Check("B", "trainer/x", name, 0.1, "", "fail")

    ok = Check("B", "trainer/x", "overfit one batch (no regularisation): worst head corr", 0.99, "", "pass")
    assert verdict([ok]) == ("learns", [])
    name = "overfit one batch (no regularisation): worst head corr(mu, label)"
    assert verdict([ok, fail(name)]) == ("does not learn", [name])
    wrong = "train vs eval mode, close: sd ratio eval / train (corr)"
    assert verdict([ok, fail(wrong)]) == ("learns the wrong thing", [wrong])
    assert verdict([fail(wrong), fail(name)])[0] == "does not learn"        # not learning wins
    assert verdict([Check("A", "m", "x", 1, "", "fail")]) == ("learns", [])  # other cards do not decide


def test_resume_is_exact(ctx):
    assert resume_check(ctx)["max_abs_diff"] == 0


def test_mode_gap_has_an_entry_per_target(ctx):
    gap = mode_gap(ctx, rows=256, passes=2)
    assert set(gap["heads"]) == set(ctx.targets)
    for g in gap["heads"].values():
        assert {"corr", "sd_ratio", "mean_shift_sd"} <= set(g)
    assert isinstance(gap["batch_norm_var_ratio"], dict)


def test_overfit_one_batch_reaches_high_corr_on_every_head(ctx):
    res = overfit_one_batch(ctx, steps=150)
    assert all(res[t] >= 0.9 for t in ctx.targets), res
    assert res["loss_last"] < res["loss_first"]


def test_lr_range_test_is_finite_and_ordered(ctx):
    res = lr_range_test(ctx, steps=20)
    assert np.isfinite(res["lr_best"]) and np.isfinite(res["lr_diverge"])
    assert res["lr_best"] <= res["lr_diverge"] and len(res["loss"]) == 20
