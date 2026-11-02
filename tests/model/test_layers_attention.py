"""P11: RMSNorm + RelativeGQAttention vs sublayers.npz block_1/01..07 (input: block_1/00_block_input)."""

import pytest
import torch

from nig_timenet_torch.model.layers import RelativeGQAttention, RMSNorm, keras_dense_weight_to_torch
from tests.oracle import oracle_loader as ol

D, H, HKV, MAX_REL = 16, 4, 2, 16


@pytest.fixture(scope="module")
def sub():
    return ol.load_sublayers()


@pytest.fixture(scope="module")
def norm1():
    layer = RMSNorm(D)
    with torch.no_grad():
        layer.scale.copy_(ol.load_weights()["block_1.norm1.scale"])
    return layer.eval()


@pytest.fixture(scope="module")
def attn():
    w = ol.load_weights()
    layer = RelativeGQAttention(D, H, HKV, max_rel_pos=MAX_REL)
    with torch.no_grad():
        for name in ("wq", "wk", "wv", "wo"):
            getattr(layer, name).weight.copy_(keras_dense_weight_to_torch(w[f"block_1.attn.{name}.kernel"]))
        layer.rel_bias.copy_(w["block_1.attn.rel_bias"])
    assert torch.equal(layer.wq.weight.t(), w["block_1.attn.wq.kernel"])
    return layer.eval()


@pytest.fixture(scope="module")
def attn_run(attn, sub):
    with torch.no_grad():
        out = attn(sub["block_1/01_norm1_out"])
    return out, attn.last_scores_pre_softmax, attn.last_attn_weights


def test_norm1_matches_oracle(norm1, sub):
    with torch.no_grad():
        ol.assert_close(norm1(sub["block_1/00_block_input"]), sub["block_1/01_norm1_out"], "forward", "norm1")


def _proj(attn, sub, proj, key):
    with torch.no_grad():
        ol.assert_close(getattr(attn, proj)(sub["block_1/01_norm1_out"]), sub[f"block_1/{key}"], "forward", key)


def test_q_proj_matches_oracle(attn, sub):
    _proj(attn, sub, "wq", "02_attn_q_proj")


def test_k_proj_matches_oracle(attn, sub):
    _proj(attn, sub, "wk", "03_attn_k_proj")


def test_v_proj_matches_oracle(attn, sub):
    _proj(attn, sub, "wv", "04_attn_v_proj")


def test_scores_pre_softmax_matches_oracle(attn_run, sub):
    ol.assert_close(attn_run[1], sub["block_1/05_attn_scores_pre_softmax"], "forward", "scores")


def test_attn_weights_matches_oracle(attn_run, sub):
    ol.assert_close(attn_run[2], sub["block_1/06_attn_weights"], "forward", "attn weights")


def test_attn_out_matches_oracle(attn_run, sub):
    ol.assert_close(attn_run[0], sub["block_1/07_attn_out"], "forward", "attn out")


def test_mask_branch_causal_and_window():
    """Unreachable in the oracle (causal=False, window=None), so checked structurally here (consumer: P34)."""
    torch.manual_seed(0)
    layer = RelativeGQAttention(8, 2, 1, max_rel_pos=4, causal=True, window=2).eval()
    with torch.no_grad():
        layer(torch.randn(1, 6, 8))
    w = layer.last_attn_weights[0, 0]
    i, j = torch.meshgrid(torch.arange(6), torch.arange(6), indexing="ij")
    assert torch.all(w[(j > i) | ((i - j) > 2)] == 0)
    torch.testing.assert_close(w.sum(-1), torch.ones(6))
