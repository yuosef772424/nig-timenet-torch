"""The iTransformer and TimesNet encoders (model/encoders.py): build through the registry, finite outputs, a row's
output does not depend on the other rows of its batch, and the gradient reaches every encoder weight."""

import pytest
import torch

from nig_timenet_torch.model.architectures import ModelSettings, build_model
from nig_timenet_torch.model.encoders import TimesBlock
from nig_timenet_torch.workflow.settings import default_model_settings


def _model(encoder):
    b = default_model_settings()
    torch.manual_seed(0)
    return build_model(ModelSettings(b.architecture, {**b.overrides, "encoder": encoder}), 32, 3).eval()


@pytest.mark.parametrize("encoder", ["itransformer", "timesnet"])
def test_rows_are_independent_of_their_batch(encoder):
    m = _model(encoder)
    x = torch.randn(16, 32, 3)
    with torch.no_grad():
        full = m(x)
        alone = m(x[3:4])
    for k, v in full.items():
        torch.testing.assert_close(v[3:4], alone[k], rtol=1e-4, atol=1e-5)
        assert torch.isfinite(v).all()


@pytest.mark.parametrize("encoder", ["itransformer", "timesnet"])
def test_every_encoder_weight_gets_a_gradient(encoder):
    m = _model(encoder).train()
    out = m(torch.randn(8, 32, 3))
    sum(v.sum() for v in out.values()).backward()
    names = ("var_embed", "iblock_") if encoder == "itransformer" else ("step_proj", "times_")
    enc = [(n, p) for n, p in m.named_parameters() if n.startswith(names)]
    assert enc and all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in enc)


def test_times_block_mixes_only_the_periods_a_row_chose():
    blk = TimesBlock(4, 4, top_k=1).eval()
    t = torch.arange(24, dtype=torch.float32)
    rows = torch.stack([torch.sin(2 * torch.pi * t / 6), torch.sin(2 * torch.pi * t / 4)])[:, :, None].expand(2, 24, 4)
    with torch.no_grad():
        both = blk(rows.contiguous())
        one = blk(rows[:1].contiguous())
    torch.testing.assert_close(both[:1], one)
