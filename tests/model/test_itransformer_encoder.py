"""The iTransformer encoder (model/encoders.py, ported from the sequential-gradient branch): builds through the
registry on the project's default model (with its fitted scalers), finite outputs, a row's output does not depend on
the other rows of its batch, every encoder weight gets a gradient, the attention head refuses it."""

import pytest
import torch

from nig_timenet_torch.model.architectures import ModelSettings, build_model
from nig_timenet_torch.model.scalers import fit_scalers
from nig_timenet_torch.workflow.settings import default_model_settings


def _model(**extra):
    b = default_model_settings()
    torch.manual_seed(0)
    m = build_model(ModelSettings(b.architecture, {**b.overrides, "encoder": "itransformer", **extra}), 32, 3)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 32, 3, generator=g) * 0.02
    fit_scalers(m, x, {f"y_{t}_reg": torch.randn(64, generator=g) * 0.03 for t in ("high", "low", "close")})
    return m, x


def test_rows_are_independent_of_their_batch_and_finite():
    m, x = _model()
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
    enc = [(n, p) for n, p in m.named_parameters() if n.startswith(("var_embed", "iblock_"))]
    assert enc and all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in enc)


def test_the_attention_head_is_not_defined_for_it():
    with pytest.raises(NotImplementedError, match="itransformer"):
        _model(nig_head="scalar_joint")
