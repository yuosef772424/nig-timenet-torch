"""P13: readout layers vs layer_outputs 007_final_norm .. 017_trunk_drop.

Chained on ORACLE values: each test feeds the oracle output of the layer(s) named in layers.json 'inputs', so a
wrong layer turns only its own test red. Dense layers are torch Linear (+ exact GELU where upstream uses
activation="gelu": stats_proj, readout_fc; linear_path is plain linear), as builder.py _branch builds them.
"""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nig_timenet_torch.model.layers import (
    AttentionPool,
    FiLM,
    GatedFusion,
    LastToken,
    RMSNorm,
    keras_dense_weight_to_torch,
)
from tests.oracle import oracle_loader as ol

D = 16


@pytest.fixture(scope="module")
def w():
    return ol.load_weights()


@pytest.fixture(scope="module")
def lo():
    return ol.load_layer_outputs()


@pytest.fixture(scope="module")
def meta():
    return {m["name"]: m for m in ol.load_layers_meta()}


def inputs(name, meta, lo):
    """Oracle tensors feeding layer `name`, resolved from layers.json 'inputs' ('x[i]' = i-th output of x)."""
    out = []
    for ref in meta[name]["inputs"]:
        src, _, idx = ref.partition("[")
        out.append(lo[meta[src]["output_keys"][int(idx.rstrip("]")) if idx else 0]])
    return out


def key(i, name):
    return f"{i:03d}_{name}"


def linear(w, name, n_in, n_out):
    lin = nn.Linear(n_in, n_out)
    with torch.no_grad():
        lin.weight.copy_(keras_dense_weight_to_torch(w[f"{name}.kernel"]))
        lin.bias.copy_(w[f"{name}.bias"])
    return lin


def rms(w, name):
    n = RMSNorm(D)
    with torch.no_grad():
        n.scale.copy_(w[f"{name}.scale"])
    return n


def check(out, lo, i, name):
    ol.assert_close(out, lo[key(i, name)], "forward", name)


@torch.no_grad()
def test_final_norm(w, lo, meta):
    check(rms(w, "final_norm")(*inputs("final_norm", meta, lo)), lo, 7, "final_norm")


@torch.no_grad()
def test_last_token(lo, meta):
    check(LastToken()(*inputs("last_token", meta, lo)), lo, 8, "last_token")


@torch.no_grad()
def test_attn_pool(w, lo, meta):
    pool = AttentionPool(D)
    pool.query.copy_(w["attn_pool.query"])
    check(pool(*inputs("attn_pool", meta, lo)), lo, 9, "attn_pool")


@torch.no_grad()
def test_stats_proj(w, lo, meta):
    assert meta["stats_proj"]["inputs"] == ["instance_norm[1]"]
    check(F.gelu(linear(w, "stats_proj", 12, D)(*inputs("stats_proj", meta, lo))), lo, 10, "stats_proj")


@torch.no_grad()
def test_readout_concat(lo, meta):
    assert meta["readout_concat"]["inputs"] == ["last_token", "attn_pool", "stats_proj"]
    check(torch.cat(inputs("readout_concat", meta, lo), dim=-1), lo, 11, "readout_concat")


@torch.no_grad()
def test_flatten_window(lo, meta):
    assert meta["flatten_window"]["inputs"] == ["instance_norm[0]"]
    (z,) = inputs("flatten_window", meta, lo)
    check(z.flatten(1), lo, 12, "flatten_window")


@torch.no_grad()
def test_readout_fc(w, lo, meta):
    check(F.gelu(linear(w, "readout_fc", 3 * D, D)(*inputs("readout_fc", meta, lo))), lo, 13, "readout_fc")


@torch.no_grad()
def test_linear_path(w, lo, meta):
    check(linear(w, "linear_path", 192, D)(*inputs("linear_path", meta, lo)), lo, 14, "linear_path")


@torch.no_grad()
def test_add_linear_path(lo, meta):
    a, b = inputs("add_linear_path", meta, lo)
    check(a + b, lo, 15, "add_linear_path")


@torch.no_grad()
def test_trunk_norm(w, lo, meta):
    check(rms(w, "trunk_norm")(*inputs("trunk_norm", meta, lo)), lo, 16, "trunk_norm")


@torch.no_grad()
def test_trunk_drop(lo, meta):
    check(nn.Dropout(0.1).eval()(*inputs("trunk_drop", meta, lo)), lo, 17, "trunk_drop")


def test_film_and_gated_fusion_shapes_and_identity_init():
    """Dead code until P34; only shape/identity-at-init behaviour is checked."""
    x, e = torch.randn(2, 5, D), torch.randn(2, D)
    torch.testing.assert_close(FiLM(D)(x, e), x)
    a, b = torch.randn(2, D), torch.randn(2, D)
    assert GatedFusion(D)(a, b).shape == (2, D)
