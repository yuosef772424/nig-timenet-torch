"""The rollout evaluation contract: direction accuracy next to its trivial baselines (rule C1) and the
shuffled-labels control that must sit at the always-up share."""

import numpy as np
import pytest
import torch

import nig_timenet_torch.eval.ar_stream as ev
from nig_timenet_torch.eval.ar_stream import (
    always_up_accuracy,
    direction_control,
    panel_direction_control,
    panel_rollout_metrics,
    persistence_accuracy,
    rollout_metrics,
)
from nig_timenet_torch.model.ar_candle_transformer import ArCandleTransformer
from nig_timenet_torch.model.panel_ar_candle import PanelArCandleTransformer


def _model() -> ArCandleTransformer:
    torch.manual_seed(0)
    return ArCandleTransformer(n_features=5, context_len=8, horizon_len=4, d_model=16,
                               num_layers=1, num_heads=4, num_kv_heads=2)


def _data(n: int = 64, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8, 5)).astype("float32")
    Y = rng.normal(size=(n, 4, 5)).astype("float32")
    dirs = np.sign(Y[:, :, 0]).astype("float32")
    dirs[dirs == 0] = 1.0  # no flat moves: every step scores
    return X, Y, dirs, X[:, -1, 0]


def test_persistence_and_always_up_baselines():
    _, Y, dirs, last_ret = _data()
    up = always_up_accuracy(dirs)
    assert 0.0 <= up <= 1.0
    per = persistence_accuracy(last_ret, dirs)
    expected = float(((np.repeat(np.sign(last_ret)[:, None], 4, axis=1) > 0) == (dirs > 0)).mean())
    assert per == pytest.approx(expected)


def test_rollout_metrics_report_and_baselines():
    X, Y, dirs, last_ret = _data()
    rep = rollout_metrics(_model(), X, Y, dirs, last_ret)
    for key in ("direction_accuracy", "persistence_accuracy", "always_up_accuracy", "trivial_gap", "mae_ret"):
        assert key in rep and np.isfinite(rep[key])
    assert len(rep["direction_accuracy_per_step"]) == dirs.shape[1]
    assert rep["n_scored"] == dirs.size  # no flat moves in this fixture


def test_perfect_model_scores_one(monkeypatch):
    """A rollout that returns the true candles scores direction accuracy 1.0 - the metric measures what
    it claims (a model echoing the truth cannot score below it)."""
    X, Y, dirs, last_ret = _data()
    monkeypatch.setattr(ev, "rollout",
                        lambda m, ctx, horizon_len=None: {"mu": torch.as_tensor(Y, dtype=torch.float32)})
    rep = rollout_metrics(_model(), X, Y, dirs, last_ret)
    assert rep["direction_accuracy"] == pytest.approx(1.0)
    assert rep["mae_ret"] == pytest.approx(0.0, abs=1e-6)


def test_direction_control_lands_near_expected():
    X, _, dirs, _ = _data()
    control = direction_control(_model(), X, dirs, n_shuffles=5)
    assert abs(control["shuffled_direction_accuracy"] - control["expected"]) < 0.15
    assert len(control["shuffled_scores"]) == 5


def test_horizon_len_is_free_and_independent_of_the_data():
    """Scoring 2 of the data's 4 horizon steps works with any model, and asks for 0 or 5 steps is a
    loud error, not a silent wrong answer."""
    X, Y, dirs, last_ret = _data()
    rep = rollout_metrics(_model(), X, Y, dirs, last_ret, horizon_len=2)
    assert len(rep["direction_accuracy_per_step"]) == 2 and rep["n_scored"] == 2 * len(X)
    for bad in (0, 5):
        with pytest.raises(ValueError, match="outside the data's horizon"):
            rollout_metrics(_model(), X, Y, dirs, last_ret, horizon_len=bad)
        with pytest.raises(ValueError, match="outside the data's horizon"):
            direction_control(_model(), X, dirs, horizon_len=bad)


def _panel_model() -> PanelArCandleTransformer:
    torch.manual_seed(0)
    return PanelArCandleTransformer(n_features=5, n_coins=3, context_len=8, horizon_len=4, d_model=16,
                                    num_layers=1, num_heads=4, num_kv_heads=2, cross_layers=1)


def _panel_data(n: int = 32, seed: int = 2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3, 8, 5)).astype("float32")
    Y = rng.normal(size=(n, 3, 4, 5)).astype("float32")
    dirs = np.sign(Y[:, :, :, 0]).astype("float32")
    dirs[dirs == 0] = 1.0
    return X, Y, dirs, X[:, :, -1, 0]


def test_panel_rollout_metrics_and_free_horizon():
    X, Y, dirs, last_ret = _panel_data()
    rep = panel_rollout_metrics(_panel_model(), X, Y, dirs, last_ret)
    for key in ("direction_accuracy", "persistence_accuracy", "always_up_accuracy", "trivial_gap", "mae_ret"):
        assert key in rep and np.isfinite(rep[key])
    assert len(rep["direction_accuracy_per_step"]) == 4 and rep["n_scored"] == dirs.size
    short = panel_rollout_metrics(_panel_model(), X, Y, dirs, last_ret, horizon_len=2)
    assert len(short["direction_accuracy_per_step"]) == 2 and short["n_scored"] == 2 * dirs.shape[0] * 3


def test_panel_direction_control_lands_near_expected():
    X, _, dirs, _ = _panel_data()
    control = panel_direction_control(_panel_model(), X, dirs, n_shuffles=5)
    assert abs(control["shuffled_direction_accuracy"] - control["expected"]) < 0.15
