"""The battery notebook's contract: the same static guarantees as the training notebooks (thin, clean,
real signatures, defined names), and a from-scratch headless run against a synthetic run folder - the
guarantee the owner gets on Colab from the first attempt."""

import numpy as np
import pandas as pd

from tests.notebooks.test_ar_stream_notebook import check_static, run_notebook

BATTERY_SETTINGS = {"TORCH_REPO", "TORCH_REF", "EXPECT_DEVICE", "TOKEN_PATH", "WORKDIR", "TORCH_DIR",
                    "DRIVE_RUNS", "EXTRA_MODULES", "RUN_NAME", "COST", "PCT", "TOP_PCTS",
                    "CONFIDENCE_BUCKETS", "UNCERTAINTY_QUANTILES", "SHOCK_GROUPS", "MIN_WEEK_TRADES",
                    "MIN_COIN_TRADES", "SELECTIVE_QUANTILES", "NEUTRAL_QUANTILES"}


def _run_folder(root, n: int = 600, seed: int = 0) -> None:
    """A tiny but honest run folder: the predictions a GOOD model would leave on Drive."""
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 0.01, n)
    t = rng.uniform(0.05, 4.0, n)
    p_right = 0.55 + 0.40 * (t / 4.0)
    side = np.where(rng.random(n) < p_right, np.sign(y), -np.sign(y))
    frame = pd.DataFrame({
        "step": np.tile(np.arange(1, 9), n // 8)[:n], "mu": 0.8 * y + rng.normal(0, 0.002, n),
        "sigma": np.full(n, 0.004), "p_dir": 1.0 / (1.0 + np.exp(-side * t)),
        "y_ret": y, "dir_true": np.where(y > 0, 1, -1),
        "last_ret": rng.normal(0, 0.01, n), "coin": rng.choice(["AAA", "BBB"], n),
        "end_time": pd.date_range("2026-01-05", periods=n, freq="1h").values})
    from nig_timenet_torch.eval.ar_stream import dump_run
    dump_run(str(root), "trial-000", {"val": frame, "test": frame},
             {"best_val_dir_acc": 0.6}, {"direction_accuracy": 0.6}, {"epochs": 2})


def test_clean_thin_and_the_settings_cell_is_complete():
    check_static("ar_stream_battery", settings_vars=BATTERY_SETTINGS, min_checked=2)


def test_the_battery_notebook_runs_from_the_first_attempt(tmp_path):
    _run_folder(tmp_path)
    ns = run_notebook("ar_stream_battery", {"EXPECT_DEVICE": "cpu", "DRIVE_RUNS": str(tmp_path)})
    assert len(ns["V"]) == len(ns["T"]) == 600
    assert ns["RUN"].name == "trial-000"
    for key in ("integrity", "sanity", "confidence", "cowardice", "trades", "limit", "uncertainty",
                "shocks", "weekly", "per_coin", "flip", "baseline", "selective", "neutral",
                "verdict"):
        assert key in ns["out"] and ns["out"][key]["rows"]
    assert ns["out"]["verdict"]["tradeable"]  # the honest table must be called tradeable
