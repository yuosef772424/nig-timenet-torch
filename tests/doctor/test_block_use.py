"""doctor/block_use.py (card A, transformer part): sublayer switches are exact and leave the model as it was, a
planted dead block is the one flagged, and the ``ffn`` option of TransformerBlock (attention-only blocks)."""

import numpy as np
import pytest
import torch

from nig_timenet_torch.core.schema import TARGET_COLUMNS
from nig_timenet_torch.doctor.block_use import block_use, switched_off, transformer_blocks
from nig_timenet_torch.model.builder import build_nig_timenet_v2
from nig_timenet_torch.model.layers import TransformerBlock

KW = dict(d_model=16, num_layers=2, num_heads=4, num_kv_heads=2, head_hidden=16, dropout=0.0)


def _model(**kw):
    torch.manual_seed(0)
    return build_nig_timenet_v2(32, 3, **{**KW, **kw})


def _xy(n=96):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(n, 32, 3, generator=g)
    return x, {t: np.random.default_rng(2).normal(size=n) for t in TARGET_COLUMNS}


def test_rows_per_block_and_part_and_the_model_is_left_as_it_was():
    model, (x, y) = _model(), _xy()
    before = model.eval()(x)["y_close"].clone()
    checks, rows = block_use(model, x, y, TARGET_COLUMNS)
    assert [(r["block"], r["part"]) for r in rows] == [
        *((b, p) for b in ("block_1", "block_2") for p in ("attn", "ffn", "block")), ("all blocks", "block")]
    assert len(checks) == len(rows) and all(c.card == "A" for c in checks)
    assert all(0.0 < r["attn_entropy"] <= 1.0 + 1e-6 and r["mu_change"] >= 0.0 for r in rows[:-1])
    assert rows[-1]["mu_change"] >= max(r["mu_change"] for r in rows[:-1] if r["part"] == "block") - 1e-9
    assert all(r["update_ratio"] > 0 for r in rows if r["part"] != "block")
    assert torch.equal(model(x)["y_close"], before)                     # hooks removed, weights untouched


def test_switching_a_part_off_equals_removing_its_update():
    block = TransformerBlock(16, 4, 2, dropout=0.0).eval()
    x = torch.randn(3, 7, 16)
    with switched_off(block, ("attn", "ffn")):
        assert torch.equal(block(x), x)
    with switched_off(block, ("ffn",)):
        assert torch.allclose(block(x), x + block.attn(block.norm1(x)))
    assert not torch.equal(block(x), x)


def test_a_planted_dead_block_is_the_one_flagged():
    model, (x, y) = _model(), _xy()
    with torch.no_grad():                                               # block_2 adds nothing to the stream
        model.block_2.attn.wo.weight.zero_()
        model.block_2.ffn.w2.weight.zero_()
    checks, rows = block_use(model, x, y, TARGET_COLUMNS)
    flagged = [c.stage for c in checks if c.verdict == "warn"]
    assert flagged == ["model/block_2"]
    assert {r["block"]: r["mu_change"] for r in rows if r["part"] == "block"}["block_1"] > 0.01


def test_no_transformer_block_gives_no_rows():
    assert block_use(_model(encoder="gru"), *_xy(), TARGET_COLUMNS) == ([], [])


@pytest.mark.parametrize("encoder", ["transformer", "itransformer"])
def test_ffn_none_builds_attention_only_blocks_and_default_keeps_the_keys(encoder):
    full, bare = _model(encoder=encoder), _model(encoder=encoder, ffn="none")
    lost = set(full.state_dict()) - set(bare.state_dict())
    assert lost and all(".ffn." in k or ".norm2." in k for k in lost)
    assert set(bare.state_dict()) <= set(full.state_dict())
    assert all(b.ffn is None for b in transformer_blocks(bare).values())
    out = bare(torch.randn(2, 32, 3))
    assert all(torch.isfinite(v).all() for v in out.values())
    if encoder == "transformer":
        _, rows = block_use(bare, *_xy(), TARGET_COLUMNS)
        assert [r["part"] for r in rows] == ["attn", "block"] * 2 + ["block"]


def test_ffn_dim_sets_the_swiglu_width_and_unknown_kind_raises():
    assert _model(ffn_dim=8).block_1.ffn.w1.weight.shape == (8, 16)
    with pytest.raises(ValueError, match="ffn must be one of"):
        _model(ffn="mlp")
