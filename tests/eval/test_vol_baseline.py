"""eval/vol_baseline.py: ranges from the window, skill net of the baseline, HAR table, matched-calibration intervals."""

import numpy as np
import pytest

from nig_timenet_torch.eval import vol_baseline as vb


def window(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    vol = rng.lognormal(np.log(0.01), 0.5, n)
    c = rng.normal(0, 1, (n, 32)) * vol[:, None]
    h = np.maximum(c, 0) + np.abs(rng.normal(0, 1, (n, 32))) * vol[:, None]
    lo = np.minimum(c, 0) - np.abs(rng.normal(0, 1, (n, 32))) * vol[:, None]
    return np.stack([h, lo, c], axis=2), vol


def test_window_ranges_and_refusal():
    x, _ = window(10)
    r = vb.window_ranges(x, ["high", "low", "close"])
    assert np.allclose(r["bar1"], x[:, -1, 0] - x[:, -1, 1])
    assert np.allclose(r["mean32"], (x[:, :, 0] - x[:, :, 1]).mean(1))
    assert np.allclose(r["atr14"], (np.maximum(x[:, -14:, 0], 0) - np.minimum(x[:, -14:, 1], 0)).mean(1))
    with pytest.raises(ValueError, match="high < low"):
        vb.window_ranges(x[:, :, [1, 0, 2]], ["high", "low", "close"])


def test_a_model_that_only_knows_the_volatility_has_no_net_skill():
    x, vol = window()
    rng = np.random.default_rng(1)
    fut = vol * np.abs(rng.normal(0, 1, len(vol))) * 2
    groups, coins = np.repeat(np.arange(100), 30), np.tile(np.arange(30), 100)
    base = vb.window_ranges(x, ["high", "low", "close"])
    vol_model = vb.head_vs_baselines(vol * 2, fut, base, groups, coins)
    assert vol_model["model"]["rank_ic"] > 0.2 and abs(vol_model["mean32"]["model_net_of_it"]) < 0.1
    skilled = vb.head_vs_baselines(fut + 0.001 * rng.normal(size=len(fut)), fut, base, groups, coins)
    assert skilled["mean32"]["model_net_of_it"] > 0.5


def test_har_table_and_intervals():
    x, vol = window(4000)
    rng = np.random.default_rng(2)
    base = vb.window_ranges(x, ["high", "low", "close"])
    y = 0.5 * base["mean32"] + vol * rng.normal(0, 1, len(vol))
    half = len(y) // 2
    fv, ft = ({k: v[:half] for k, v in base.items()}, {k: v[half:] for k, v in base.items()})
    t = vb.har_table(fv, y[:half], ft, y[half:], np.arange(half) % 20, {"mean32": ["mean32"]})
    assert t["mean32"]["r2"] > 0.0
    iv = vb.interval_vs_baseline({"y": y[:half], "pred": 0.5 * base["mean32"][:half], "std": vol[:half], "base": fv},
                                 {"y": y[half:], "pred": 0.5 * base["mean32"][half:], "std": vol[half:], "base": ft})
    assert iv["model"]["coverage_val"] == pytest.approx(0.8, abs=0.01)
    assert iv["model"]["coverage_test"] == pytest.approx(0.8, abs=0.03)
    assert iv["volatility_baseline"]["coverage_test"] == pytest.approx(0.8, abs=0.05)
