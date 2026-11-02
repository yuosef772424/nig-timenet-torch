"""P10: num_patches + PatchEmbedding vs layers.json #3 'trend_seasonal' (concat) and #4 'patch_embed'."""

import pytest
import torch

from nig_timenet_torch.model.layers import PatchEmbedding, keras_conv1d_weight_to_torch, num_patches
from tests.oracle import oracle_loader as ol


def test_num_patches_matches_15_tokens_pad_0():
    assert num_patches(32, 4, 2) == (15, 0)
    assert num_patches(33, 4, 2) == (16, 1)  # pad branch: last patch still ends at the last step


def test_keras_conv1d_weight_layout():
    w = torch.arange(4 * 3 * 5, dtype=torch.float32).reshape(4, 3, 5)  # (kernel, in, out)
    t = keras_conv1d_weight_to_torch(w)
    assert t.shape == (5, 3, 4) and t[4, 2, 1] == w[1, 2, 4]


def test_patch_embed_matches_oracle():
    w, lo = ol.load_weights(), ol.load_layer_outputs()
    x = torch.cat([lo["002_decomp__0"], lo["002_decomp__1"]], dim=-1)
    ol.assert_close(x, lo["003_trend_seasonal"], "weights", "trend_seasonal is a plain concat (exact)")
    layer = PatchEmbedding(in_features=12, d_model=16, patch_len=4, stride=2, seq_len=32)
    with torch.no_grad():
        layer.proj.weight.copy_(keras_conv1d_weight_to_torch(w["patch_embed.proj.kernel"]))
        layer.proj.bias.copy_(w["patch_embed.proj.bias"])
        layer.pos.copy_(w["patch_embed.pos_emb"])
        out = layer(x)
    assert torch.equal(layer.pos, w["patch_embed.pos_emb"])
    ol.assert_close(out, lo["004_patch_embed"], "forward", "patch_embed")


def test_patch_embed_lazy_build_and_pad():
    layer = PatchEmbedding(in_features=3, d_model=8)
    assert layer.pos is None
    out = layer(torch.randn(2, 33, 3))
    assert out.shape == (2, 16, 8) and layer.pad == 1
    with pytest.raises(ValueError):
        layer(torch.randn(2, 32, 3))
