"""doctor/common.py: Check rows, the purged time folds, the ridge probe, the simple baseline and the report table."""

import numpy as np
import pytest

from nig_timenet_torch.doctor.common import (
    Check,
    probe_scores,
    render_table,
    ridge_oof,
    simple_features,
    time_folds,
    worst,
)


def test_time_folds_embargo_and_coverage():
    rng = np.random.default_rng(0)
    times = rng.permutation(np.arange(400))            # rows are not in time order (coin-major in an artifact)
    folds = time_folds(times, n_folds=4, embargo=10)
    assert len(folds) == 4
    assert np.array_equal(np.sort(np.concatenate([te for _, te in folds])), np.arange(400))   # tests cover every row
    for tr, te in folds:
        lo, hi = times[te].min(), times[te].max()
        assert not set(tr) & set(te)
        assert np.all((times[tr] < lo - 10) | (times[tr] > hi + 10))                          # embargo on both sides
        near = np.flatnonzero((times >= lo - 10) & (times <= hi + 10))
        assert not set(near) & set(tr)
    assert len(time_folds(times, 4, embargo=0)[1][0]) > len(folds[1][0])    # no embargo keeps more train rows


def _signal_data(n=1200, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, 5))
    y = a @ np.array([1.0, -2.0, 0.5, 0.0, 0.0]) + rng.normal(scale=noise, size=n)
    return a, y, time_folds(np.arange(n), 4, embargo=5)


def test_ridge_oof_recovers_a_planted_linear_signal():
    a, y, folds = _signal_data()
    s = probe_scores(a, {"y": y}, folds, squares=False)["y"]
    assert s["r2"] > 0.8 and s["ic"] > 0.8
    pred = ridge_oof(a, y, folds)
    assert np.isfinite(pred).all()


def test_ridge_oof_scores_about_zero_on_noise():
    a, _, folds = _signal_data()
    noise = np.random.default_rng(5).normal(size=len(a))
    s = probe_scores(a, {"n": noise}, folds)["n"]
    assert s["r2"] < 0.05 and abs(s["ic"]) < 0.15


def test_multi_target_equals_single_target():
    a, y, folds = _signal_data(n=600)
    y2 = np.stack([y, np.random.default_rng(1).normal(size=len(y)), a[:, 0] ** 2], axis=1)
    multi = ridge_oof(a, y2, folds)
    for j in range(3):
        np.testing.assert_allclose(multi[:, j], ridge_oof(a, y2[:, j], folds), rtol=1e-8, atol=1e-10)
    one = probe_scores(a, {"p": y, "q": y2[:, 1]}, folds)
    assert one["p"]["r2"] == pytest.approx(probe_scores(a, {"p": y}, folds)["p"]["r2"])


def test_simple_features_shape_and_names():
    x = np.random.default_rng(0).normal(size=(30, 12, 3)) * 0.01
    f, names = simple_features(x, ["high", "low", "close"])
    assert f.shape == (30, len(names)) and len(set(names)) == len(names)
    assert "close_sd" in names and "high_mean1" in names and "log_range6" in names
    np.testing.assert_allclose(f[:, names.index("close_mean1")], x[:, -1, 2])
    np.testing.assert_allclose(f[:, names.index("range12")], (x[:, :, 0] - x[:, :, 1]).mean(axis=1))
    f2, n2 = simple_features(x[:, :, :1], ["close"])        # no high / low: no range terms
    assert not any(n.startswith("range") for n in n2) and f2.shape[1] == len(n2)


def test_check_rejects_unknown_verdict_and_worst():
    with pytest.raises(ValueError):
        Check("A", "s", "c", 1, verdict="great")
    mk = lambda v: Check("A", "s", "c", 1, verdict=v)  # noqa: E731
    assert worst([]) == "pass" and worst([mk("info"), mk("pass")]) == "pass"
    assert worst([mk("pass"), mk("warn")]) == "warn" and worst([mk("warn"), mk("fail"), mk("pass")]) == "fail"


def test_render_table_has_one_line_per_check():
    checks = [Check("A", "s1", "first | with pipe", 0.123456, 1.0, "pass"),
              Check("B", "s2", "second", "text", "", "fail", "do this")]
    lines = render_table(checks).splitlines()
    assert len(lines) == 2 + len(checks) and "first / with pipe" in lines[2]
    assert "do this" in lines[3]
    assert len(render_table(checks, only={"fail"}).splitlines()) == 3


def test_run_build_fitted_the_scalers_on_the_train_rows(ctx):
    import torch

    from nig_timenet_torch.model.scalers import FittedScale, InputScaler, label_scale

    fitted = ctx.run.extra["fitted_scalers"]
    assert {"input_scaler", "nig_label_scale"} <= set(fitted)
    assert fitted["input_scaler"]["rows"] == len(ctx.run.idx["train"])
    scaler = next(m for m in ctx.model.modules() if isinstance(m, InputScaler))
    flat = ctx.run.x["train"].reshape(-1, 3).double()
    torch.testing.assert_close(scaler.mean.double(), flat.mean(0), rtol=1e-4, atol=1e-8)
    keys = [f"y_{t}_reg" for t in ctx.targets]
    expect = label_scale(ctx.run.y["train"], keys)
    assert fitted["nig_label_scale"] == expect
    assert all(float(m.value) == torch.tensor(expect).item() for m in ctx.model.modules() if isinstance(m, FittedScale))
