"""FX2 model switches: ``nig_trunk_grad`` (nig_layers.grad_scale) and ``nig_scale_fit`` (model/label_scale.py): the
defaults build the same model; on, each changes what it names (gradient into the trunk / a fitted head scale that is
stored with the weights)."""

import pytest
import torch

from nig_timenet_torch.core.registry import UnknownNameError
from nig_timenet_torch.model.architectures import ANTI_MEMORIZATION_CONFIG, ModelSettings, build_model
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.model.label_scale import LABEL_SCALES, fit_label_scale, label_scale_state
from nig_timenet_torch.model.nig_layers import NIGHead, grad_scale

TARGETS = ("high", "low", "close")


def _model(**overrides):
    torch.manual_seed(0)
    return build_model(ModelSettings("nig_timenet_v2", {**ANTI_MEMORIZATION_CONFIG, **overrides}), 32, 3)


def _x():
    return torch.randn(6, 32, 3, generator=torch.Generator().manual_seed(1))


def _labels(scale=0.02, n=200):
    g = torch.Generator().manual_seed(2)
    return {head_spec("nig_regression", t).label_key: scale * torch.randn(n, generator=g) for t in TARGETS}


@pytest.mark.parametrize("g", [0.0, 0.25, 1.0])
def test_grad_scale_keeps_the_forward_and_scales_the_gradient(g):
    x = torch.randn(4, 3, requires_grad=True)
    y = grad_scale(x, g)
    assert torch.equal(y, x)
    (grad,) = torch.autograd.grad((y * 3.0).sum(), x)
    assert torch.allclose(grad, torch.full_like(x, 3.0 * g))


def test_defaults_build_the_same_model():
    a, b = _model(), _model(nig_trunk_grad=1.0, nig_scale_fit="fixed")
    assert a.state_dict().keys() == b.state_dict().keys()
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    a.eval(), b.eval()
    with torch.no_grad():
        oa, ob = a(_x()), b(_x())
    assert all(torch.equal(oa[k], ob[k]) for k in oa)
    assert fit_label_scale(a, _labels()) is None and label_scale_state(a) is None


def test_nig_trunk_grad_scales_only_the_nig_gradient_into_the_trunk():
    x = _x()
    grads = {}
    for g in (1.0, 0.5):
        m = _model(nig_trunk_grad=g).eval()
        out = m(x)
        loss = out[head_spec("nig_regression", "close").output_keys["mu"]].sum()
        grads[g] = torch.autograd.grad(loss, m.readout_fc.weight)[0]
    assert torch.allclose(grads[0.5], 0.5 * grads[1.0], atol=1e-7)


def test_fitted_scale_is_the_mean_train_std_and_is_stored_with_the_weights():
    m = _model(nig_scale_fit="train_std")
    assert label_scale_state(m) == 1.0                          # built: 1.0 until fitted
    labels = _labels(0.02)
    state = fit_label_scale(m, labels)
    expected = float(torch.stack([v.double().std(unbiased=False) for v in labels.values()]).mean())
    assert state["scale"] == pytest.approx(expected) and label_scale_state(m) == pytest.approx(expected)
    fresh = _model(nig_scale_fit="train_std")
    fresh.load_state_dict(m.state_dict())                       # a restored model reads s from its weights
    assert label_scale_state(fresh) == pytest.approx(expected)
    m.eval(), fresh.eval()
    with torch.no_grad():
        assert all(torch.equal(v, fresh(_x())[k]) for k, v in m(_x()).items())


def test_fitted_scale_moves_the_beta_floor_with_the_label_variance():
    torch.manual_seed(0)
    head = NIGHead(5, 8, 0.0, beta_min=0.01, fitted_scale=True).eval()
    head.label_scale.fill_(0.02)
    head.out.bias.data[3] = -50.0                               # softplus ~ 0: beta sits on its floor
    head.out.weight.data[3] = 0.0
    _, _, _, beta = head(torch.randn(3, 5))
    assert torch.allclose(beta, torch.full_like(beta, 0.01 * 0.02 ** 2), rtol=1e-4)


def test_fitted_and_fixed_scale_are_exclusive_and_unknown_names_raise():
    with pytest.raises(ValueError, match="exclusive"):
        _model(nig_scale=0.01, nig_scale_fit="train_std")
    with pytest.raises(UnknownNameError):
        _model(nig_scale_fit="train_sd")
    assert set(LABEL_SCALES.names()) == {"fixed", "train_std"}
