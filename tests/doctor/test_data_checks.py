"""doctor/data_checks.py (card C): alignment, overlap, coverage, shift and batch independence on a small E1."""

import numpy as np

from nig_timenet_torch.doctor.data_checks import (
    ALIGN_TOL,
    alignment,
    batch_independence,
    coverage,
    data_checks,
    label_clip,
    overlap,
    shift,
)


def test_alignment_of_the_fixture_is_exact_enough(ctx):
    al = alignment(ctx.dataset)
    assert al["pairs"] > 0 and al["window_err"] <= ALIGN_TOL and ALIGN_TOL == 2e-5
    assert set(al["label_err"]) == set(ctx.targets) and max(al["label_err"].values()) <= 2e-5


def test_rolling_a_label_by_one_row_is_caught(ctx):
    bad = dict(ctx.dataset)
    bad["y_close_reg"] = np.roll(np.asarray(ctx.dataset["y_close_reg"]), 1)
    al = alignment(bad)
    assert al["label_err"]["close"] > 1e-3
    assert al["label_err"]["high"] <= 2e-5 and al["window_err"] <= 2e-5         # only the rolled label is wrong


def test_clipped_labels_of_an_artifact_without_a_recorded_clip_are_not_misalignment(ctx):
    """The Drive artifact records no ``reg_target_clip`` but clips ``return`` labels at 1.0 (B-054): a saturated label
    cannot be rebuilt from the bars and must not read as a label/row misalignment."""
    ds = {k: v for k, v in ctx.dataset.items() if k != "reg_target_clip"}
    ds["reg_target_mode"] = "return"
    y = np.array(ctx.dataset["y_low_reg"], dtype=np.float64)
    scale = float(ds["reg_target_scale"])
    y[100:140] = -1.0 * scale                                    # saturated rows, not the bars' value
    ds["y_low_reg"] = y
    assert alignment(ds)["label_err"]["low"] <= 2e-5
    assert label_clip(ds) == 1.0
    assert label_clip({**ds, "reg_target_mode": "window_scale"}) == 10.0
    assert label_clip({**ds, "reg_target_mode": "nope"}) is None
    assert label_clip({**ds, "reg_target_clip": 10.0}) == 10.0


def test_alignment_is_not_applicable_to_fitted_inputs(ctx):
    assert alignment({**ctx.dataset, "price_norm_state": "fitted"}) is None


def test_overlap_keys(ctx):
    res = overlap(ctx.dataset, ctx.run.idx["train"])
    assert {"row_step_bars", "window_overlap", "label_overlap", "train_rows", "coins_per_timestamp",
            "cross_coin_corr", "effective_coins", "duplicate_window_share"} <= set(res)
    assert res["train_rows"] == len(ctx.run.idx["train"])
    assert res["row_step_bars"] == 2.0                                          # rows are one horizon apart
    assert res["label_overlap"] == 0.0 and 0.9 < res["window_overlap"] < 1.0
    assert 0.0 <= res["duplicate_window_share"] < 0.01


def test_coverage_keys_and_clean_fixture(ctx):
    res = coverage(ctx.dataset, ctx.run.idx["val"], ctx.targets)
    assert res["nonfinite_inputs"] == 0 and 0.0 <= res["constant_window_share"] <= 1.0
    assert {"flat_bar_share", "zero_close_move_share"} <= set(res)
    for t in ctx.targets:
        assert res[f"{t}_nonfinite_labels"] == 0
        assert {f"{t}_zero_label_share", f"{t}_clipped_label_share"} <= set(res)


def test_shift_keys(ctx):
    res = shift(ctx.dataset, ctx.run.idx, ctx.targets)
    assert set(res) == {"train", "val", "test"}
    for s in ("val", "test"):
        assert {"x_sd_ratio", "y_sd_ratio", "y_mean_shift_sd", "up_share"} <= set(res[s])
        assert len(res[s]["x_sd_ratio"]) == 3 and set(res[s]["y_sd_ratio"]) == set(ctx.targets)
    assert {"y_sd", "y_mean", "x_sd", "up_share"} <= set(res["train"])


def test_batch_independence_is_zero(ctx):
    _, x, _ = ctx.xy("val", 256)
    assert batch_independence(ctx.model, x) == 0.0


def test_data_checks_pass_alignment(ctx):
    checks = data_checks(ctx)
    align = [c for c in checks if c.stage == "data/alignment"]
    assert align and not [c for c in align if c.verdict == "fail"]
