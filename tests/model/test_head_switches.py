"""FX1 head switches (NIGHead / OrderedMeans ``scale`` = model ``nig_scale``, ``order_detach_close``): default = the TF
formula bit for bit; on = the change it names and nothing else."""

import pytest
import torch
import torch.nn.functional as F

from nig_timenet_torch.model.architectures import ModelSettings, build_model
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.model.nig_layers import OrderedMeans


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(8, 1, generator=g) for _ in range(3))


def test_default_is_the_tf_formula_bit_for_bit():
    mu_c, raw_h, raw_l = _inputs()
    h, c, low = OrderedMeans()(mu_c, raw_h, raw_l)
    assert torch.equal(h, mu_c + F.softplus(raw_h)) and torch.equal(c, mu_c)
    assert torch.equal(low, mu_c - F.softplus(raw_l))


def test_scale_offsets_are_softplus_of_the_unscaled_raw_and_keep_the_order():
    mu_c, raw_h, raw_l = _inputs()
    s_ = 0.02
    h, c, low = OrderedMeans(scale=s_)(s_ * mu_c, s_ * raw_h, s_ * raw_l)
    ref = OrderedMeans()(mu_c, raw_h, raw_l)
    assert all(torch.allclose(a, s_ * b, atol=1e-7) for a, b in zip((h, c, low), ref, strict=True))
    assert bool((h > c).all() and (c > low).all())


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_scale_must_be_positive(bad):
    with pytest.raises(ValueError, match="scale"):
        OrderedMeans(scale=bad)


@pytest.mark.parametrize("detach", [False, True])
def test_detach_close_cuts_only_the_high_low_gradient_into_mu_close(detach):
    mu_c, raw_h, raw_l = (t.clone().requires_grad_(True) for t in _inputs())
    h, c, low = OrderedMeans(detach_close=detach)(mu_c, raw_h, raw_l)
    ref = OrderedMeans()(mu_c, raw_h, raw_l)
    assert all(torch.equal(a, b) for a, b in zip((h, c, low), ref, strict=True))  # forward unchanged
    (g_from_high,) = torch.autograd.grad(h.sum(), mu_c, retain_graph=True, allow_unused=True)
    g_from_high = torch.zeros_like(mu_c) if g_from_high is None else g_from_high
    (g_from_close,) = torch.autograd.grad(c.sum(), mu_c)
    assert torch.equal(g_from_high, torch.zeros_like(mu_c) if detach else torch.ones_like(mu_c))
    assert torch.equal(g_from_close, torch.ones_like(mu_c))


def _mu(arch, overrides, x):
    torch.manual_seed(0)
    model = build_model(ModelSettings(arch, overrides), 32, 3).eval()
    with torch.no_grad():
        out = model(x)
    return {t: out[head_spec("nig_regression", t).output_keys["mu"]] for t in ("high", "close", "low")}


@pytest.mark.parametrize("arch", ["nig_timenet_v2", "tiny_mlp"])
def test_model_default_overrides_reproduce_the_model_and_detach_keeps_the_forward(arch):
    x = torch.randn(4, 32, 3, generator=torch.Generator().manual_seed(1))
    base = _mu(arch, {}, x)
    same = _mu(arch, {"nig_scale": None, "order_detach_close": False}, x)
    assert all(torch.equal(base[t], same[t]) for t in base)
    detached = _mu(arch, {"order_detach_close": True}, x)
    assert all(torch.equal(detached[t], base[t]) for t in base)  # forward identical, only gradients differ


def test_nig_head_scale_is_training_on_scaled_labels_with_outputs_in_label_units():
    from nig_timenet_torch.model.nig_layers import NIGHead
    torch.manual_seed(0)
    plain = NIGHead(5, 8, 0.0).eval()
    scaled = NIGHead(5, 8, 0.0, scale=0.01).eval()
    scaled.load_state_dict(plain.state_dict())
    h = torch.randn(6, 5)
    mu, nu, alpha, beta = plain(h)
    mu_s, nu_s, alpha_s, beta_s = scaled(h)
    assert torch.allclose(mu_s, 0.01 * mu) and torch.allclose(beta_s, 1e-4 * beta)
    assert torch.equal(nu_s, nu) and torch.equal(alpha_s, alpha)
    with pytest.raises(ValueError, match="scale"):
        NIGHead(5, 8, scale=0.0)


@pytest.mark.parametrize("arch", ["nig_timenet_v2", "tiny_mlp"])
def test_model_nig_scale_scales_every_mean_and_the_ordered_offsets(arch):
    x = torch.randn(4, 32, 3, generator=torch.Generator().manual_seed(2))
    base = _mu(arch, {}, x)
    scaled = _mu(arch, {"nig_scale": 0.01}, x)
    assert all(torch.allclose(scaled[t], 0.01 * base[t], atol=1e-8) for t in base)
