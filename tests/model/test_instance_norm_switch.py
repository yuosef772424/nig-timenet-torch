"""``use_instance_norm=False`` (owner 2026-11-18): the per-window normalisation is skipped, everything else (stats,
parameter names) unchanged; the default model is the same as before."""

import torch

from nig_timenet_torch.model.architectures import ModelSettings, build_model
from nig_timenet_torch.model.layers import InstanceNorm
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


def test_without_normalisation_z_is_the_input_and_the_stats_are_unchanged():
    x = torch.randn(4, T, 3) * 0.3 + 1.0
    on, off = InstanceNorm(3, stats_mode="symlog"), InstanceNorm(3, stats_mode="symlog", normalize=False)
    (z_on, s_on), (z_off, s_off) = on(x), off(x)
    assert torch.equal(z_off, x) and not torch.allclose(z_on, x)
    assert torch.equal(s_on, s_off)


def test_the_switch_keeps_the_parameter_names_and_lets_the_level_through():
    plain, off = _model(), _model(use_instance_norm=False)
    assert plain.state_dict().keys() == off.state_dict().keys()
    x = torch.randn(32, T, 3) * 0.02
    with torch.no_grad():
        scaled_plain, scaled_off = plain.input_scaler(x), off.input_scaler(x)
        clip = off.instance_norm.clip                                          # the soft clip still applies
        want = clip * torch.tanh(scaled_off / clip) if clip else scaled_off
        assert torch.allclose(off.instance_norm(scaled_off)[0], want)        # the scaled input reaches the encoder
        assert not torch.allclose(plain.instance_norm(scaled_plain)[0], scaled_plain)
