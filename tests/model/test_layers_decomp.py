"""P09: CausalMultiScaleDecomp vs layers.json #2 'decomp'.

Input: layers.json #2 'decomp' has inputs == ['instance_norm[0]'], i.e. layer_outputs['001_instance_norm__0'] (the
normalised z), not the raw model input.
"""

import torch

from nig_timenet_torch.model.layers import CausalMultiScaleDecomp, keras_dense_weight_to_torch
from tests.oracle import oracle_loader as ol


def build_decomp():
    w = ol.load_weights()
    layer = CausalMultiScaleDecomp(n_features=6, kernel_sizes=(3, 5, 9, 17))
    with torch.no_grad():
        layer.gate.weight.copy_(keras_dense_weight_to_torch(w["decomp.gate.kernel"]))
        layer.gate.bias.copy_(w["decomp.gate.bias"])
    assert torch.equal(layer.gate.weight.t(), w["decomp.gate.kernel"])
    return layer


def test_decomp_input_is_instance_norm_z():
    meta = {m["name"]: m for m in ol.load_layers_meta()}
    assert meta["decomp"]["inputs"] == ["instance_norm[0]"]


def test_decomp_matches_oracle():
    lo = ol.load_layer_outputs()
    with torch.no_grad():
        trend, seasonal = build_decomp()(lo["001_instance_norm__0"])
    ol.assert_close(trend, lo["002_decomp__0"], "forward", "decomp trend")
    ol.assert_close(seasonal, lo["002_decomp__1"], "forward", "decomp seasonal")


def test_decomp_is_causal():
    """Changing a future step must not change the trend at earlier steps (past-only moving averages)."""
    layer = build_decomp()
    x = torch.randn(1, 32, 6)
    y = x.clone()
    y[:, 20:] += 5.0
    with torch.no_grad():
        # the gate reads the last step, so compare the per-kernel moving averages via a fixed gate
        layer.gate.weight.zero_()
        t1, _ = layer(x)
        t2, _ = layer(y)
    torch.testing.assert_close(t1[:, :20], t2[:, :20])
