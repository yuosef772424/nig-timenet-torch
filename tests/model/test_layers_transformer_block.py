"""P12: SwiGLU + TransformerBlock vs sublayers block_N/08..11 and layer_outputs 005_block_1 / 006_block_2.

Wiring read from layers.json (asserted below): block_1 <- patch_embed, block_2 <- block_1. Eval mode only.
"""

import pytest
import torch

from nig_timenet_torch.model.layers import TransformerBlock, keras_dense_weight_to_torch
from tests.oracle import oracle_loader as ol

D, H, HKV, MAX_REL = 16, 4, 2, 16


def load_block(name: str) -> TransformerBlock:
    w = ol.load_weights()
    block = TransformerBlock(D, H, HKV, max_rel_pos=MAX_REL, dropout=0.1)
    with torch.no_grad():
        block.norm1.scale.copy_(w[f"{name}.norm1.scale"])
        block.norm2.scale.copy_(w[f"{name}.norm2.scale"])
        block.attn.rel_bias.copy_(w[f"{name}.attn.rel_bias"])
        for p in ("wq", "wk", "wv", "wo"):
            getattr(block.attn, p).weight.copy_(keras_dense_weight_to_torch(w[f"{name}.attn.{p}.kernel"]))
        for p in ("w1", "w2", "w3"):
            getattr(block.ffn, p).weight.copy_(keras_dense_weight_to_torch(w[f"{name}.ffn.{p}.kernel"]))
    return block.eval()


@pytest.fixture(scope="module")
def sub():
    return ol.load_sublayers()


@pytest.fixture(scope="module")
def block1():
    return load_block("block_1")


def test_dff_default():
    assert load_block("block_1").ffn.dff == 40


def test_residual1_matches_oracle(block1, sub):
    x = sub["block_1/00_block_input"]
    with torch.no_grad():
        out = x + block1.drop1(block1.attn(block1.norm1(x)))
    ol.assert_close(out, sub["block_1/08_residual1"], "forward", "residual1")


def test_norm2_matches_oracle(block1, sub):
    with torch.no_grad():
        ol.assert_close(block1.norm2(sub["block_1/08_residual1"]), sub["block_1/09_norm2_out"], "forward", "norm2")


def test_swiglu_matches_oracle(block1, sub):
    with torch.no_grad():
        ol.assert_close(block1.ffn(sub["block_1/09_norm2_out"]), sub["block_1/10_swiglu_out"], "forward", "swiglu")


def test_block_output_matches_oracle(block1, sub):
    r = sub["block_1/08_residual1"]
    with torch.no_grad():
        out = r + block1.drop2(block1.ffn(block1.norm2(r)))
    ol.assert_close(out, sub["block_1/11_block_output"], "forward", "block output")


@pytest.mark.parametrize("name,key,src",
                         [("block_1", "005_block_1", "patch_embed"), ("block_2", "006_block_2", "block_1")])
def test_transformer_block_forward_matches_layer_outputs(name, key, src):
    meta = {m["name"]: m for m in ol.load_layers_meta()}
    assert meta[name]["inputs"] == [src]
    lo = ol.load_layer_outputs()
    x = lo[meta[src]["output_keys"][0]]
    with torch.no_grad():
        out = load_block(name)(x)
    ol.assert_close(out, lo[key], "forward", name)
