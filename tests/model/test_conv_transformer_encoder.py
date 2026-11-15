"""The conv_transformer encoder (model/encoders.py): per-bar tokens + causal conv stem + attention blocks. Builds
through the registry on the project's default model, finite outputs independent of the batch, every encoder weight
gets a gradient, the stem is causal, and it works with the attention NIG head and with attention-only blocks."""

import pytest
import torch

from nig_timenet_torch.model.architectures import ModelSettings, build_model
from nig_timenet_torch.model.scalers import fit_scalers
from nig_timenet_torch.model.taps import stream_taps
from nig_timenet_torch.workflow.settings import default_model_settings


def _model(**extra):
    b = default_model_settings()
    torch.manual_seed(0)
    m = build_model(ModelSettings(b.architecture, {**b.overrides, "encoder": "conv_transformer", **extra}), 32, 3)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 32, 3, generator=g) * 0.02
    fit_scalers(m, x, {f"y_{t}_reg": torch.randn(64, generator=g) * 0.03 for t in ("high", "low", "close")})
    return m, x


@pytest.mark.parametrize("extra", [{}, {"ffn": "none"}, {"nig_head": "scalar_joint"}])
def test_rows_are_independent_of_their_batch_and_finite(extra):
    m, x = _model(**extra)
    m.eval()
    with torch.no_grad():
        full, alone = m(x), m(x[3:4])
    for k, v in full.items():
        assert torch.isfinite(v).all()
        torch.testing.assert_close(v[3:4], alone[k], rtol=1e-4, atol=1e-5)


def test_every_encoder_weight_gets_a_gradient():
    m, x = _model()
    m.train()
    sum(v.sum() for v in m(x[:16]).values()).backward()
    enc = [(n, p) for n, p in m.named_parameters() if n.startswith(("step_proj", "stem_", "block_"))]
    assert len(enc) > 10 and all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in enc)


def test_one_token_per_bar_and_the_stem_does_not_look_ahead():
    m, _ = _model()
    m.eval()
    tok = torch.randn(2, 32, m.step_proj.in_features)
    changed = tok.clone()
    changed[:, 20:] += 1.0                                   # bars 20.. change: stem tokens 0..19 must not move
    with torch.no_grad(), stream_taps(m) as a:
        m.encode_tokens(tok)
    with torch.no_grad(), stream_taps(m) as b:
        m.encode_tokens(changed)
    assert a["enc_in"].shape == (2, 32, 64)
    torch.testing.assert_close(a["enc_in"][:, :20], b["enc_in"][:, :20])
    assert not torch.allclose(a["enc_in"][:, 20:], b["enc_in"][:, 20:])
