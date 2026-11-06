"""The aware autoregressive trainer contract: the robust directional loss (a sudden move cannot explode
the gradient), scheduled sampling on detached own predictions, and learning a genuinely predictable
stream - direction accuracy above the persistence baseline, not just a falling loss."""

import numpy as np
import pytest
import torch

from nig_timenet_torch.trainer.ar_stream import (
    ArStreamSettings,
    build_ar_stream_model,
    fit_ar_stream,
    robust_candle_loss,
    step_weights,
)

B, H, F = 4, 8, 5


def _settings(**over):
    base = dict(context_len=16, horizon_len=H, d_model=32, num_layers=1, num_heads=4, num_kv_heads=2,
                epochs=4, batch_size=64, lr=2e-3, patience=4, verbose=False)
    base.update(over)
    return ArStreamSettings(**base)


def test_step_weights_shrink_on_outliers():
    w = step_weights(torch.tensor([[0.001, 3.0]]), ret_scale=0.01, outlier_softness=8.0)
    assert w[0, 0] == pytest.approx(1.0, abs=0.01)  # a typical move keeps (nearly) full weight
    assert w[0, 1] < 0.01                            # a sudden move is strongly down-weighted, not deleted
    assert w[0, 1] > 0.0


def test_huber_caps_the_gradient_on_a_shock():
    s = _settings(huber_delta=1.0)
    mu = torch.zeros(1, 1, F, requires_grad=True)
    ls = torch.zeros(1, 1, F)
    y_normal = torch.full((1, 1, F), 0.5)
    y_shock = torch.full((1, 1, F), 50.0)
    grads = []
    for y in (y_normal, y_shock):
        loss, _ = robust_candle_loss(mu, ls, y, torch.zeros(1, 1), torch.ones(1, 1),
                                     torch.ones(1, 1), s)
        loss.backward()
        grads.append(mu.grad.clone().abs().sum().item())
        mu.grad.zero_()
    # the shock's gradient is bounded by delta (linear Huber arm), not 100x the normal one's
    assert grads[1] <= 1.0 + 1e-6 and grads[1] > 0


def test_direction_term_survives_the_outlier_weight():
    s = _settings(w_dir=1.0)
    mu = torch.zeros(1, 1, F)
    ls = torch.zeros(1, 1, F)
    for weight in (torch.ones(1, 1), torch.zeros(1, 1)):  # even a zeroed magnitude weight
        loss, stats = robust_candle_loss(mu, ls, torch.ones(1, 1, F), torch.zeros(1, 1),
                                         torch.ones(1, 1), weight, s)
        assert stats["direction"] > 0  # the direction lesson keeps its full weight


def _predictable_stream(n: int = 900, seed: int = 0):
    """A stream whose next-return SIGN is learnable from context: returns alternate in runs with momentum."""
    rng = np.random.default_rng(seed)
    ret = np.zeros(n)
    sign, run = 1.0, 0
    for i in range(1, n):
        if run <= 0:
            sign, run = -sign, rng.integers(3, 8)
        ret[i] = sign * rng.uniform(0.004, 0.012)
        run -= 1
    feats = np.zeros((n, F), dtype="float32")
    feats[:, 0] = ret
    feats[:, 3] = np.sign(ret) * 0.5
    from nig_timenet_torch.data.ar_windows import build_windows
    w = build_windows(feats, [slice(0, n)], 16, H)
    cut = int(w.n * 0.8)
    return w.X[:cut], w.Y[:cut], w.dirs[:cut], w.X[cut:], w.Y[cut:], w.dirs[cut:]


def test_fit_learns_direction_not_noise():
    X, Y, dirs, vX, vY, vdirs = _predictable_stream()
    model = build_ar_stream_model(_settings(), n_features=F)
    hist = fit_ar_stream(model, X, Y, dirs, vX, vY, vdirs, _settings(ss_start=0.0, ss_end=0.5))
    from nig_timenet_torch.eval.ar_stream import rollout_metrics
    rep = rollout_metrics(model, vX, vY, vdirs, vX[:, -1, 0])
    assert rep["direction_accuracy"] > rep["persistence_accuracy"] + 0.05
    assert hist["best_val_dir_acc"] > 0.55


def test_full_own_predictions_still_trains():
    X, Y, dirs, vX, vY, vdirs = _predictable_stream(600)
    model = build_ar_stream_model(_settings(), n_features=F)
    hist = fit_ar_stream(model, X, Y, dirs, vX, vY, vdirs, _settings(ss_start=1.0, ss_end=1.0))
    assert hist["train_loss"][-1] < hist["train_loss"][0]  # learns from its own rollout too
    assert hist["p_own"] == [1.0] * len(hist["p_own"])


def test_shape_mismatch_is_refused():
    X, Y, dirs, vX, vY, vdirs = _predictable_stream(300)  # data horizon 8
    model = build_ar_stream_model(_settings(horizon_len=16), n_features=F)
    with pytest.raises(ValueError, match="do not fit"):
        fit_ar_stream(model, X, Y, dirs, vX, vY, vdirs, _settings(horizon_len=16))  # 16 > the data's 8
    with pytest.raises(ValueError, match="do not fit"):
        fit_ar_stream(model, X, Y, dirs, vX, vY, vdirs, _settings(context_len=8, horizon_len=8))
