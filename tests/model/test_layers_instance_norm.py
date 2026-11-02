"""P08: InstanceNorm vs layers.json #1 'instance_norm' (input: #0 input_sequence = the raw model input)."""

import numpy as np
import pytest
import torch

from nig_timenet_torch.model.layers import InstanceNorm, SymLog
from tests.oracle import oracle_loader as ol


def test_instance_norm_matches_oracle():
    w, lo = ol.load_weights(), ol.load_layer_outputs()
    layer = InstanceNorm(n_features=6, eps=1e-4)
    with torch.no_grad():
        layer.gamma.copy_(w["instance_norm.gamma"])
        layer.beta.copy_(w["instance_norm.beta"])
    assert torch.equal(layer.gamma, w["instance_norm.gamma"])  # loaded weights: exact
    with torch.no_grad():
        z, stats = layer(ol.load_inputs())
    ol.assert_close(z, lo["001_instance_norm__0"], "forward", "instance_norm z")
    ol.assert_close(stats, lo["001_instance_norm__1"], "forward", "instance_norm stats")


def test_instance_norm_clip_and_modes():
    x = torch.randn(2, 7, 3)
    z_plain, _ = InstanceNorm(3)(x)
    z_clip, _ = InstanceNorm(3, clip=0.5)(x)
    torch.testing.assert_close(z_clip, 0.5 * torch.tanh(z_plain / 0.5))
    with pytest.raises(ValueError):
        InstanceNorm(3, stats_mode="bogus")
    assert InstanceNorm(3, stats_mode="none")(x)[1] is None
    _, full = InstanceNorm(3)(x)
    _, sym = InstanceNorm(3, stats_mode="symlog", stats_clip=0.1)(x)
    m = full[:, :3]
    torch.testing.assert_close(sym, torch.cat([torch.sign(m) * torch.log1p(m.abs()), full[:, 3:].clamp(-0.1, 0.1)], -1))


def test_symlog_formula():
    x = torch.tensor([-7e4, -2.0, -0.5, 0.0, 0.5, 2.0, 7e4])
    expected = np.clip(np.sign(x.numpy()) * np.log1p(np.abs(x.numpy())), -8.0, 8.0)
    torch.testing.assert_close(SymLog()(x), torch.from_numpy(expected))
    torch.testing.assert_close(SymLog(clip=None)(x)[-1], torch.log1p(torch.tensor(7e4)))
