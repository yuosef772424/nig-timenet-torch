"""H3 switch ``har_range`` (model/har_range.py): the terms are the HAR range means of the RAW window, the default graph
is unchanged, and with the switch on the readout sees the window's range level (which the instance norm erases)."""

import math

import pytest
import torch

from nig_timenet_torch.model.architectures import ModelSettings, build_model
from nig_timenet_torch.model.har_range import HarRange, har_terms
from nig_timenet_torch.model.scalers import fit_scalers
from nig_timenet_torch.workflow.settings import default_model_settings

T = 32


def _model(**overrides):
    base = default_model_settings()
    torch.manual_seed(0)
    m = build_model(ModelSettings(base.architecture, {**base.overrides, **overrides}), T, 3)
    x = torch.randn(64, T, 3) * 0.02
    fit_scalers(m, x, {k: torch.randn(64) * 0.01 for k in ("y_high_reg", "y_low_reg", "y_close_reg")})
    return m


def test_har_terms_are_the_range_means_and_their_logs():
    x = torch.zeros(2, 8, 3)
    x[:, :, 0] = torch.arange(1, 9, dtype=torch.float32) * 0.01      # high: 0.01 .. 0.08
    x[:, :, 1] = -0.01                                                # low
    t = har_terms(x, 0, 1)[0]
    rng = [0.02 + 0.01 * i for i in range(8)]                         # high - low per bar
    want = [rng[-1], sum(rng[-6:]) / 6, sum(rng) / 8]
    assert t[:3].tolist() == pytest.approx(want) and t[3:].tolist() == pytest.approx([math.log(v) for v in want])
    flat = har_terms(torch.zeros(1, 8, 3), 0, 1)[0]                   # a flat window: logs clamped, finite
    assert torch.isfinite(flat).all()


def test_default_graph_has_no_har_module_and_the_switch_adds_one():
    plain, har = _model(), _model(har_range=[0, 1])
    assert not [k for k in plain.state_dict() if "har" in k]
    extra = {k for k in har.state_dict()} - set(plain.state_dict())
    assert extra and all(k.startswith("har_range.") for k in extra)
    assert har.readout_fc.in_features == plain.readout_fc.in_features + plain.readout_fc.out_features   # + d_model


def test_the_readout_sees_the_window_range_level():
    """Scaling a window's high / low channels changes its range level, which the instance norm removes; with the
    switch the model's output responds to it far more than without."""
    x = torch.randn(32, T, 3) * 0.02
    x[..., 0], x[..., 1] = x[..., 0].abs(), -x[..., 1].abs()
    wide = x.clone()
    wide[..., :2] *= 3.0
    gaps = {}
    for name, m in (("plain", _model()), ("har", _model(har_range=[0, 1]))):
        m.eval()
        with torch.no_grad():
            h, hw = m.trunk(x), m.trunk(wide)
        gaps[name] = float((h - hw).norm() / h.norm())
    assert gaps["har"] > gaps["plain"]


def test_bad_channels_raise():
    with pytest.raises(ValueError, match="two different input channels"):
        HarRange((1, 1), 8)
