"""The aware autoregressive trainer contract: the robust directional loss (a sudden move cannot explode
the gradient), scheduled sampling on detached own predictions, and learning a genuinely predictable
stream - direction accuracy above the persistence baseline, not just a falling loss."""

import numpy as np
import pytest
import torch

from nig_timenet_torch.trainer.ar_stream import (
    ArStreamSettings,
    build_ar_stream_model,
    czar_loss,
    czar_softening,
    fit_ar_stream,
    local_vol,
    robust_candle_loss,
    step_weights,
)

B, H, F = 4, 8, 5


def _settings(**over):
    base = dict(context_len=16, horizon_len=H, d_model=32, num_layers=1, num_heads=4, num_kv_heads=2,
                epochs=4, batch_size=64, lr=2e-3, patience=4, verbose=False)
    base.update(over)
    return ArStreamSettings(**base)


def test_czar_makes_the_zero_prediction_expensive():
    # a real move (z = 2 local stds): predicting 0 must cost MORE than a bold right-sign overshoot -
    # the anti-cowardice property (arXiv:2609.36061: the zero prediction is an attractor for symmetric
    # losses; CZAR's Region A keeps a linear penalty that pushes away from it)
    z = torch.tensor([[2.0]])
    zero = czar_loss(z, torch.zeros(1, 1))
    bold = czar_loss(z, torch.full((1, 1), 2.5))
    assert zero > bold
    zp = torch.zeros(1, 1, requires_grad=True)
    czar_loss(z, zp).backward()
    assert abs(zp.grad.item()) > 0.1  # the gradient keeps pushing the prediction away from zero


def test_czar_bounds_the_shock_gradient():
    # a 50-sigma shock (a headline, a thin book): the loss stays bounded and the gradient keeps a
    # bounded influence (Cauchy kernel) instead of the quadratic explosion
    z = torch.tensor([[50.0]])
    assert czar_loss(z, torch.zeros(1, 1)).item() < 10.0
    zp = torch.zeros(1, 1, requires_grad=True)
    czar_loss(z, zp).backward()
    assert abs(zp.grad.item()) < 2.0


def test_czar_softening_counts_near_zero_and_decays_on_big_moves():
    near = czar_softening(torch.tensor([[0.0]]), 1.0, 1.0, 0.05)
    mid = czar_softening(torch.tensor([[10.0]]), 1.0, 1.0, 0.05)
    big = czar_softening(torch.tensor([[100.0]]), 1.0, 1.0, 0.05)
    assert near.item() > 0.0          # fitting flat noise cannot pay: the floor is counted...
    assert mid.item() < near.item() / 4  # ...but decays strongly once the truth leaves the epsilon band
    assert big.item() < mid.item()       # ...and keeps decaying as the move grows


def test_local_vol_scores_each_window_in_its_own_regime():
    X = torch.tensor([[[0.01, 0.0], [0.02, 0.0], [0.03, 0.0]],     # a calm stretch
                      [[0.10, 0.0], [-0.12, 0.0], [0.11, 0.0]]])  # a wild one
    vol = local_vol(X, ret_scale=0.01, floor_frac=0.25)
    assert vol[1] > vol[0] * 5           # each window is scored in its own market regime
    assert float(vol.min()) >= 0.25 * 0.01  # and the floor protects the quietest regime


def test_direction_ignores_flat_truths_and_ramps_with_move():
    s = _settings(w_dir=1.0, w_move=0.5)
    mu = torch.zeros(1, 2, F)
    ls = torch.zeros(1, 2, F)
    y = torch.zeros(1, 2, F)
    vol = torch.full((1,), 0.01)
    # a confidently-"up" logit on a FLAT truth changes nothing: the flat sign is noise, excluded
    dl_flat = torch.tensor([[0.0, 12.0]])
    db = torch.tensor([[1, 0]])
    _, a = robust_candle_loss(mu, ls, y, dl_flat, db, vol, torch.ones(1, 2), s)
    _, b = robust_candle_loss(mu, ls, y, torch.zeros(1, 2), db, vol, torch.ones(1, 2), s)
    assert a["direction"] == pytest.approx(b["direction"])
    # the same wrong-sign step costs more on a bigger move: a shock's sign is the lesson
    dl = torch.tensor([[-3.0]])
    db = torch.tensor([[1]])
    step = torch.zeros(1, 1, F)
    step[0, 0, 0] = 0.01  # z = 1 local std
    _, small = robust_candle_loss(mu, ls, step, dl, db, vol, torch.ones(1, 1), s)
    step[0, 0, 0] = 0.05  # z = 5 local stds
    _, big = robust_candle_loss(mu, ls, step, dl, db, vol, torch.ones(1, 1), s)
    assert big["direction"] > small["direction"]


def test_robust_loss_bounds_a_shock_and_skips_the_flat_step():
    s = _settings()
    mu = torch.zeros(1, 2, F, requires_grad=True)
    ls = torch.zeros(1, 2, F)
    y = torch.zeros(1, 2, F)
    y[0, 0, 0] = 50.0  # a 50-sigma shock on the return channel
    loss, _ = robust_candle_loss(mu, ls, y, torch.zeros(1, 2), torch.tensor([[1, 0]]),
                                 torch.ones(1), torch.ones(1, 2), s)
    loss.backward()
    assert mu.grad[0, 0, 0].abs() > 0                                  # bounded, not silenced
    # the 50-sigma shock and the FLAT step receive the SAME bounded gradient: the Cauchy kernel's
    # bounded influence - a headline cannot buy more gradient than a quiet candle
    assert float(mu.grad[0, 1, 0].abs()) == pytest.approx(float(mu.grad[0, 0, 0].abs()), rel=0.01)
    assert float(mu.grad[0, 0, 0].abs()) < 2.0


def test_huber_loss_kind_is_still_selectable():
    # the v1 robust loss, via settings.loss = "huber": the adaptive weight shrinks on a shock...
    w = step_weights(torch.tensor([[0.001, 3.0]]), ret_scale=0.01, outlier_softness=8.0)
    assert w[0, 0] == pytest.approx(1.0, abs=0.01)  # a typical move keeps (nearly) full weight
    assert 0.0 < w[0, 1] < 0.01                     # a sudden move is strongly down-weighted
    # ...the magnitude gradient is capped by huber_delta (linear arm, never quadratic)...
    s = _settings(loss="huber", huber_delta=1.0)
    mu = torch.zeros(1, 1, F, requires_grad=True)
    ls = torch.zeros(1, 1, F)
    grads = []
    for val in (0.5, 50.0):
        y = torch.full((1, 1, F), val)
        loss, _ = robust_candle_loss(mu, ls, y, torch.zeros(1, 1), torch.ones(1, 1),
                                     torch.ones(1), torch.ones(1, 1), s)
        loss.backward()
        grads.append(mu.grad.clone().abs().sum().item())
        mu.grad.zero_()
    assert 0 < grads[1] <= 1.0 + 1e-6
    # ...and the direction lesson keeps its full weight even under a zeroed magnitude weight
    for weight in (torch.ones(1, 1), torch.zeros(1, 1)):
        loss, stats = robust_candle_loss(mu, ls, torch.ones(1, 1, F), torch.zeros(1, 1), torch.ones(1, 1),
                                        torch.ones(1), weight, _settings(loss="huber", w_dir=1.0))
        assert stats["direction"] > 0


def test_unknown_loss_kind_is_refused():
    with pytest.raises(ValueError, match="unknown loss kind"):
        robust_candle_loss(torch.zeros(1, 1, F), torch.zeros(1, 1, F), torch.zeros(1, 1, F),
                           torch.zeros(1, 1), torch.ones(1, 1), torch.ones(1), torch.ones(1, 1),
                           _settings(loss="nonsense"))


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
    torch.manual_seed(0)  # fit re-seeds internally, but the model INIT reads the global RNG
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


def test_huber_loss_kind_still_learns_the_stream():
    X, Y, dirs, vX, vY, vdirs = _predictable_stream()
    torch.manual_seed(0)
    model = build_ar_stream_model(_settings(loss="huber"), n_features=F)
    hist = fit_ar_stream(model, X, Y, dirs, vX, vY, vdirs, _settings(loss="huber"))
    assert hist["best_val_dir_acc"] > 0.55  # the selectable v1 path keeps its old quality


def test_shape_mismatch_is_refused():
    X, Y, dirs, vX, vY, vdirs = _predictable_stream(300)  # data horizon 8
    model = build_ar_stream_model(_settings(horizon_len=16), n_features=F)
    with pytest.raises(ValueError, match="do not fit"):
        fit_ar_stream(model, X, Y, dirs, vX, vY, vdirs, _settings(horizon_len=16))  # 16 > the data's 8
    with pytest.raises(ValueError, match="do not fit"):
        fit_ar_stream(model, X, Y, dirs, vX, vY, vdirs, _settings(context_len=8, horizon_len=8))
