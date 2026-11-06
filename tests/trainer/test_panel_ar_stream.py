"""The panel trainer contract: the same aware recipe on a (windows, coins) grid - it learns a genuinely
predictable stream across coins, handles a data horizon longer than the training horizon, and trains
even when every appended candle is the model's own prediction."""

import numpy as np

from nig_timenet_torch.data.ar_windows import build_windows
from nig_timenet_torch.eval.ar_stream import panel_rollout_metrics
from nig_timenet_torch.trainer.panel_ar_stream import (
    PanelArStreamSettings,
    build_panel_ar_stream_model,
    fit_panel_ar_stream,
)

C, H, F, N = 16, 8, 5, 3


def _settings(**over):
    base = dict(context_len=C, horizon_len=H, d_model=32, num_layers=1, num_heads=4, num_kv_heads=2,
                cross_layers=1, epochs=4, batch_size=64, lr=2e-3, patience=4, verbose=False)
    base.update(over)
    return PanelArStreamSettings(**base)


def _panel(n: int = 900, seed: int = 0):
    """A panel of N coins whose next-return SIGN is learnable from each coin's own context."""
    X = np.zeros((n - C - H + 1, N, C, F), "float32")
    Y = np.zeros((n - C - H + 1, N, H, F), "float32")
    dirs = np.zeros((n - C - H + 1, N, H), "float32")
    for c in range(N):
        ret = np.zeros(n)
        sign, run = 1.0, 0
        r = np.random.default_rng(seed + 100 + c)
        for i in range(1, n):
            if run <= 0:
                sign, run = -sign, r.integers(3, 8)
            ret[i] = sign * r.uniform(0.004, 0.012)
            run -= 1
        feats = np.zeros((n, F), "float32")
        feats[:, 0] = ret
        feats[:, 3] = np.sign(ret) * 0.5
        w = build_windows(feats, [slice(0, n)], C, H)
        X[:, c], Y[:, c], dirs[:, c] = w.X, w.Y, w.dirs
    cut = int(X.shape[0] * 0.8)
    return X[:cut], Y[:cut], dirs[:cut], X[cut:], Y[cut:], dirs[cut:]


def test_panel_learns_direction():
    X, Y, dirs, vX, vY, vdirs = _panel()
    model = build_panel_ar_stream_model(_settings(), n_features=F, n_coins=N)
    hist = fit_panel_ar_stream(model, X, Y, dirs, vX, vY, vdirs, _settings(ss_start=0.0, ss_end=0.5))
    rep = panel_rollout_metrics(model, vX, vY, vdirs, vX[:, :, -1, 0])
    assert rep["direction_accuracy"] > rep["persistence_accuracy"] + 0.05
    assert hist["best_val_dir_acc"] > 0.55


def test_data_horizon_longer_than_training_is_sliced():
    X, Y, dirs, vX, vY, vdirs = _panel(400)
    model = build_panel_ar_stream_model(_settings(horizon_len=4), n_features=F, n_coins=N)
    hist = fit_panel_ar_stream(model, X, Y, dirs, vX, vY, vdirs, _settings(horizon_len=4))
    assert hist["best_val_dir_acc"] > 0.5 and len(hist["train_loss"]) == 4


def test_full_own_predictions_still_trains():
    X, Y, dirs, vX, vY, vdirs = _panel(500)
    model = build_panel_ar_stream_model(_settings(), n_features=F, n_coins=N)
    hist = fit_panel_ar_stream(model, X, Y, dirs, vX, vY, vdirs, _settings(ss_start=1.0, ss_end=1.0))
    assert hist["train_loss"][-1] < hist["train_loss"][0]
    assert hist["p_own"] == [1.0] * len(hist["p_own"])


def test_shape_mismatch_is_refused():
    X, Y, dirs, vX, vY, vdirs = _panel(300)
    model = build_panel_ar_stream_model(_settings(context_len=8), n_features=F, n_coins=N)
    import pytest
    with pytest.raises(ValueError, match="do not fit"):
        fit_panel_ar_stream(model, X, Y, dirs, vX, vY, vdirs, _settings(context_len=8))
