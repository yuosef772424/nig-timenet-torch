"""The PANEL autoregressive-stream notebook's contract: the same static guarantees as the single-coin
notebook, and a from-scratch headless run on a small synthetic panel (the guarantee the owner gets on
Colab from the first attempt)."""

import pytest

from tests.notebooks.test_ar_stream_notebook import TINY, check_static, run_notebook


def test_clean_thin_and_the_settings_cell_is_complete():
    check_static("panel_ar_stream")


def test_every_call_matches_a_real_signature_and_every_name_is_defined_earlier():
    check_static("panel_ar_stream")


def test_the_panel_notebook_runs_from_the_first_attempt():
    ns = run_notebook("panel_ar_stream", {"SOURCE": "synthetic", "DAYS": 30, "EXPECT_DEVICE": "cpu",
                                          "COINS": ("AAA", "BBB"), "HORIZON_LEN": 4, "EVAL_HORIZON": 2,
                                          "CHANGES": TINY})
    assert ns["data"]["train_X"].ndim == 4  # (windows, coins, candles, features)
    assert ns["data"]["train_X"].shape[1] == 2 and ns["data"]["train_X"].shape[2] == ns["CONTEXT_LEN"]
    assert ns["data"]["train_Y"].shape[2] == ns["FUTURE_LEN"] == 4
    assert ns["history"]["best_val_dir_acc"] > 0.0 and len(ns["history"]["train_loss"]) == 2
    for key in ("direction_accuracy", "persistence_accuracy", "always_up_accuracy", "trivial_gap"):
        assert key in ns["report"]
    assert len(ns["report"]["direction_accuracy_per_step"]) == ns["EVAL_HORIZON"] == 2
    assert ns["control"]["shuffled_direction_accuracy"] == pytest.approx(ns["control"]["expected"], abs=0.1)
