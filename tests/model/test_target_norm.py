"""model/target_norm.py: the registered target normalisation (rule B4). ``none`` builds the model of today exactly;
``logret_std`` fits per-head mean / std on the TRAIN labels only, stores them with the weights, and the outputs are
the exact affine image of the heads' z-space NIG (labels units for every consumer)."""

import math

import pytest
import torch

from nig_timenet_torch.core.registry import UnknownNameError
from nig_timenet_torch.model.architectures import ANTI_MEMORIZATION_CONFIG, ModelSettings, build_model
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.model.label_scale import fit_label_scale, label_scale_state
from nig_timenet_torch.model.target_norm import (
    TARGET_NORMS,
    TargetNorm,
    fit_target_norm,
    normalized_labels,
    target_norm_state,
)
from nig_timenet_torch.stats.nig import nig_nll

TARGETS = ("high", "low", "close")
MEANS = {"high": 0.02, "low": -0.02, "close": 0.0}
STDS = {"high": 0.03, "low": 0.025, "close": 0.04}


def _model(**overrides):
    torch.manual_seed(0)
    return build_model(ModelSettings("nig_timenet_v2", {**ANTI_MEMORIZATION_CONFIG, **overrides}), 32, 3)


def _x(n=6):
    return torch.randn(n, 32, 3, generator=torch.Generator().manual_seed(1))


def _labels(n=500):
    g = torch.Generator().manual_seed(2)
    return {head_spec("nig_regression", t).label_key:
            MEANS[t] + STDS[t] * torch.randn(n, generator=g, dtype=torch.float64) for t in TARGETS}


def _key(t, k):
    return head_spec("nig_regression", t).output_keys[k]


def test_registry_names_and_unknown_name_raises():
    assert set(TARGET_NORMS.names()) == {"none", "logret_std"}
    with pytest.raises(UnknownNameError):
        _model(target_norm="logret_sd")


def test_none_builds_the_same_model_and_fit_is_a_no_op():
    a, b = _model(), _model(target_norm="none")
    assert a.state_dict().keys() == b.state_dict().keys()
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    a.eval(), b.eval()
    with torch.no_grad():
        oa, ob = a(_x()), b(_x())
    assert all(torch.equal(oa[k], ob[k]) for k in oa)
    assert fit_target_norm(a, _labels()) is None and target_norm_state(a) is None
    labels = {"close": [1.0, 2.0]}
    assert normalized_labels(a, labels) is labels


def test_unfitted_model_raises_on_forward():
    m = _model(target_norm="logret_std").eval()
    with pytest.raises(RuntimeError, match="target_norm is not fitted"):
        m(_x())


def test_fit_uses_the_train_labels_and_is_stored_with_the_weights():
    m = _model(target_norm="logret_std")
    y = _labels()
    state = fit_target_norm(m, y)
    for t in TARGETS:
        v = y[head_spec("nig_regression", t).label_key]
        assert state[t]["mean"] == pytest.approx(float(v.mean()))
        assert state[t]["std"] == pytest.approx(float(v.std(unbiased=False)))
    fresh = _model(target_norm="logret_std")
    fresh.load_state_dict(m.state_dict())           # a restored model reads the state from its weights
    assert target_norm_state(fresh) == target_norm_state(m)
    m.eval(), fresh.eval()
    with torch.no_grad():
        assert all(torch.equal(v, fresh(_x())[k]) for k, v in m(_x()).items())


def test_outputs_are_the_affine_image_of_the_z_space_heads():
    """Same weights with and without the normalisation: mu = mean + std mu_z, beta = std^2 beta_z, nu / alpha
    unchanged, so the NIG NLL in label units = NLL in z units + log std (training is the same as on z labels)."""
    plain, norm = _model(), _model(target_norm="logret_std")
    norm.load_state_dict({**norm.state_dict(), **plain.state_dict()})
    fit_target_norm(norm, _labels())
    st = target_norm_state(norm)
    plain.eval(), norm.eval()
    with torch.no_grad():
        oz, oy = plain(_x()), norm(_x())
    y = torch.randn(6, 1, generator=torch.Generator().manual_seed(3))
    for t in TARGETS:
        m, s = st[t]["mean"], st[t]["std"]
        assert torch.allclose(oy[_key(t, "mu")], m + s * oz[_key(t, "mu")], atol=1e-6)
        assert torch.allclose(oy[_key(t, "beta")], s * s * oz[_key(t, "beta")], rtol=1e-5)
        for k in ("nu", "alpha"):
            assert torch.equal(oy[_key(t, k)], oz[_key(t, k)])
        py = [oy[_key(t, k)].double().numpy() for k in ("mu", "nu", "alpha", "beta")]
        pz = [oz[_key(t, k)].double().numpy() for k in ("mu", "nu", "alpha", "beta")]
        nll_y, nll_z = nig_nll((m + s * y).double().numpy(), *py), nig_nll(y.double().numpy(), *pz)
        assert abs(nll_y - (nll_z + math.log(s))).max() < 1e-4


def test_label_scale_is_fitted_on_the_normalised_labels():
    m = _model(target_norm="logret_std", nig_scale_fit="train_std")
    y = _labels()
    fit_target_norm(m, y)
    fit_label_scale(m, y)
    assert label_scale_state(m) == pytest.approx(1.0, abs=1e-6)   # z labels have std 1 per head


def test_transform_and_inverse_are_each_others_inverse():
    tn = TargetNorm(TARGETS)
    tn.fit({t: MEANS[t] + STDS[t] * torch.randn(300, generator=torch.Generator().manual_seed(4)) for t in TARGETS})
    y = torch.randn(10)
    for t in TARGETS:
        mu, beta = tn.inverse(tn.transform(y, t), torch.ones(10), t)
        assert torch.allclose(mu, y, atol=1e-6)
        assert torch.allclose(beta, tn.std[TARGETS.index(t)] ** 2 * torch.ones(10))
    with pytest.raises(KeyError):
        tn.transform(y, "open")
    with pytest.raises(ValueError, match="positive std"):
        TargetNorm(("close",)).fit({"close": torch.ones(5)})


def test_order_in_label_units_orders_prices_and_keeps_close_and_beta():
    """owner 2026-11-19 (B-079): with ``order_in_label_units`` the order holds on PRICES (mu_low <= mu_close <=
    mu_high in label units, on any input), the close mean and every beta are those of the default normalised model
    with the same weights, and high / low = close +- (their label sd) x the offset of the same raw head value."""
    base = _model(target_norm="logret_std")
    lab = _model(target_norm="logret_std", order_in_label_units=True)
    lab.load_state_dict(base.state_dict())          # no new parameter: the same weights fit both
    fit_target_norm(base, _labels()), fit_target_norm(lab, _labels())
    base.eval(), lab.eval()
    x = 5.0 * _x(64)                                  # large inputs: confident means of both signs
    with torch.no_grad():
        ob, ol = base(x), lab(x)
    mu = {t: ol[_key(t, "mu")] for t in TARGETS}
    assert bool((mu["low"] <= mu["close"]).all()) and bool((mu["close"] <= mu["high"]).all())
    assert torch.allclose(mu["close"], ob[_key("close", "mu")], atol=1e-6)
    for t in TARGETS:
        assert torch.allclose(ol[_key(t, "beta")], ob[_key(t, "beta")], rtol=1e-5)
    st = target_norm_state(lab)
    off = (ob[_key("high", "mu")] - st["high"]["mean"]) / st["high"]["std"] - (
        ob[_key("close", "mu")] - st["close"]["mean"]) / st["close"]["std"]   # the z-space offset of the default
    assert torch.allclose(mu["high"] - mu["close"], st["high"]["std"] * off, atol=1e-5)


def test_order_in_label_units_needs_a_target_norm_and_the_ordered_means():
    with pytest.raises(ValueError, match="order_in_label_units needs"):
        _model(order_in_label_units=True)
    with pytest.raises(ValueError, match="order_in_label_units needs"):
        _model(target_norm="logret_std", order_in_label_units=True, enforce_order=False)
