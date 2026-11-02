"""P34: input_skip seam. "none" is the P15 graph bit for bit; the four other variants build, run, and are causal
at the seam (no oracle exists for them, and no result is claimed)."""

import pytest
import torch

from nig_timenet_torch.core.registry import UnknownNameError
from nig_timenet_torch.model.builder import build_nig_timenet_v2
from nig_timenet_torch.model.parts import INPUT_SKIP_VARIANTS, LATE_RAW_K
from tests.oracle import oracle_loader as ol

VARIANTS = ("concat_each_block", "dense_skip", "late_raw", "no_proj")


def kwargs(**over):
    kw = dict(ol.load_config()["constructor_kwargs"])
    return {**kw, **over}


def test_registered_variants_and_default():
    assert set(INPUT_SKIP_VARIANTS.names()) == {"none", *VARIANTS}
    assert build_nig_timenet_v2.__kwdefaults__["input_skip"] == "none"
    with pytest.raises(UnknownNameError):
        build_nig_timenet_v2(32, 6, input_skip="concat_every_block")


def test_input_skip_none_matches_p15_oracle_exactly():
    torch.manual_seed(0)
    default = build_nig_timenet_v2(**kwargs())
    explicit = build_nig_timenet_v2(**kwargs(input_skip="none"))
    shapes = {n: p.shape for n, p in default.named_parameters()}
    assert shapes == {n: p.shape for n, p in explicit.named_parameters()}
    weights = ol.load_weights()
    for m in (default, explicit):
        m.load_oracle_weights(weights)
        m.eval()
    x = ol.load_inputs()
    with torch.no_grad():
        a, b = default(x), explicit(x)
    expected = ol.load_outputs()
    for k in expected:
        assert torch.equal(a[k], b[k]), k
        ol.assert_close(b[k], expected[k], "forward", k)


@pytest.mark.parametrize("variant", VARIANTS)
def test_each_variant_builds_and_runs(variant):
    torch.manual_seed(0)
    model = build_nig_timenet_v2(**kwargs(input_skip=variant)).eval()
    with torch.no_grad():
        out = model(ol.load_inputs())
    assert len(out) == 24 and all(torch.isfinite(v).all() for v in out.values())
    # same for the fixture's width (18 features -> 36 token channels, still divisible by num_heads for no_proj)
    torch.manual_seed(0)
    m18 = build_nig_timenet_v2(**kwargs(seq_len=16, n_features=18, input_skip=variant)).eval()
    with torch.no_grad():
        assert all(torch.isfinite(v).all() for v in m18(torch.randn(3, 16, 18)).values())


def _last_bar_of_each_position(model, seq_len: int) -> list[int]:
    if model.input_skip == "no_proj":
        return list(range(seq_len))
    pe = model.patch_embed
    return [j * pe.stride + pe.patch_len - 1 - pe.pad for j in range(pe.n_tokens)]


@pytest.mark.parametrize("variant", ("none", *VARIANTS))
def test_causality_no_future_leakage(variant):
    """Standard causality probe at the seam: with causal blocks, zeroing every bar after t leaves each encoder
    output position whose last bar is <= t unchanged (exactly). The probe is on the encoder sequence because the
    full model is window-global by design (InstanceNorm statistics, non-causal default attention, pooling)."""
    torch.manual_seed(0)
    kw = kwargs(input_skip=variant, causal=True)
    model = build_nig_timenet_v2(**kw).eval()
    seq_len, width = kw["seq_len"], 2 * kw["n_features"]
    tok = torch.randn(4, seq_len, width)
    ends = _last_bar_of_each_position(model, seq_len)
    with torch.no_grad():
        full = model.encode_tokens(tok)
        for t in (5, 12, 20, seq_len - 2):
            cut = tok.clone()
            cut[:, t + 1:] = 0.0
            part = model.encode_tokens(cut)
            keep = [j for j, e in enumerate(ends) if e <= t]
            assert keep and torch.equal(full[:, keep], part[:, keep]), (variant, t)
            later = [j for j, e in enumerate(ends) if e > t]
            assert not torch.equal(full[:, later], part[:, later])  # the probe can detect a change


def test_late_raw_reads_only_the_last_k_bars():
    torch.manual_seed(0)
    model = build_nig_timenet_v2(**kwargs(input_skip="late_raw")).eval()
    tok = torch.randn(2, 32, 12)
    extra = model.skip.readout_extra(tok)
    assert torch.equal(extra, tok[:, -LATE_RAW_K:].flatten(1))
