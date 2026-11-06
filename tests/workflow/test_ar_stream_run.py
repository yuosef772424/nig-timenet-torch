"""workflow.ar_stream: the one-call compositions (single-coin and panel) — raw frames in, one report
out, with a FREE evaluation horizon that may differ from the training horizon."""

import pytest

from nig_timenet_torch.data.ar_windows import raw_candles
from nig_timenet_torch.workflow.ar_stream import run_ar_stream, run_panel_ar_stream

TINY = {"epochs": 2, "d_model": 32, "num_layers": 1, "batch_size": 128, "patience": 2, "verbose": False}
AAA = raw_candles("synthetic", ("AAA",), 40, seed=0)
TWO = raw_candles("synthetic", ("AAA", "BBB"), 30, seed=0)


def test_run_ar_stream_trains_and_reports_at_a_free_eval_horizon():
    run = run_ar_stream(AAA, "1h", 32, 8, 4, val_share=0.15, test_share=0.15, changes=TINY)
    assert run["data"]["train_X"].shape[1] == 32 and run["data"]["train_Y"].shape[1] == 8  # longer future
    assert run["history"]["best_val_dir_acc"] > 0.0 and len(run["history"]["train_loss"]) == 2
    for key in ("direction_accuracy", "persistence_accuracy", "always_up_accuracy", "trivial_gap"):
        assert key in run["report"]
    assert len(run["report"]["direction_accuracy_per_step"]) == 4
    assert run["control"]["shuffled_direction_accuracy"] == pytest.approx(run["control"]["expected"], abs=0.1)


def test_run_panel_ar_stream_pools_the_coins_and_checks_the_control():
    run = run_panel_ar_stream(TWO, "1h", 32, 4, 2, val_share=0.15, test_share=0.15, changes=TINY)
    assert run["data"]["train_X"].shape[1] == 2  # coins stay in lockstep on shared end timestamps
    assert run["data"]["train_X"].shape[2] == 32 and run["data"]["train_Y"].shape[2] == 4
    assert run["history"]["best_val_dir_acc"] > 0.0 and len(run["history"]["train_loss"]) == 2
    assert len(run["report"]["direction_accuracy_per_step"]) == 2
    assert run["control"]["shuffled_direction_accuracy"] == pytest.approx(run["control"]["expected"], abs=0.1)


def test_run_ar_stream_rejects_a_bad_source_name_with_every_option_listed():
    with pytest.raises(ValueError, match="unknown source"):
        raw_candles("nonsense", ("AAA",), 5)
