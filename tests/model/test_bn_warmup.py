"""FX1 ``bn_warmup``: BatchNormChannelsLast moving statistics = the plain mean of the first batches (default: Keras
momentum from the 0 / 1 init, state_dict keys unchanged)."""

import torch

from nig_timenet_torch.model.architectures import ModelSettings, build_model
from nig_timenet_torch.model.layers import BatchNormChannelsLast
from nig_timenet_torch.workflow.settings import default_model_settings


def _batches(n=5):
    g = torch.Generator().manual_seed(0)
    return [0.01 * torch.randn(64, 8, 3, generator=g) + 0.5 for _ in range(n)]


def test_default_is_keras_momentum_from_the_init_and_has_no_new_buffer():
    bn = BatchNormChannelsLast(3).train()
    xs = _batches()
    mean, var = torch.zeros(3), torch.ones(3)
    for x in xs:
        bn(x)
        mean = 0.99 * mean + 0.01 * x.mean(dim=(0, 1))
        var = 0.99 * var + 0.01 * x.var(dim=(0, 1), unbiased=False)
    torch.testing.assert_close(bn.moving_mean, mean)
    torch.testing.assert_close(bn.moving_variance, var)
    assert set(bn.state_dict()) == {"gamma", "beta", "moving_mean", "moving_variance"}


def test_warmup_moving_stats_are_the_mean_of_the_batches_seen():
    bn = BatchNormChannelsLast(3, warmup=True).train()
    xs = _batches()
    for x in xs:
        bn(x)
    torch.testing.assert_close(bn.moving_mean, torch.stack([x.mean(dim=(0, 1)) for x in xs]).mean(0))
    torch.testing.assert_close(bn.moving_variance,
                               torch.stack([x.var(dim=(0, 1), unbiased=False) for x in xs]).mean(0))
    assert int(bn.num_updates) == len(xs)
    bn.eval()
    before = bn.moving_mean.clone()
    bn(xs[0])
    assert torch.equal(bn.moving_mean, before) and int(bn.num_updates) == len(xs)  # eval never updates


def test_model_override_reaches_both_batch_norms_and_default_keeps_the_state_dict():
    base = default_model_settings()
    plain = build_model(ModelSettings(base.architecture, {**base.overrides, "bn_warmup": False}), 32, 3)
    warm = build_model(ModelSettings(base.architecture, {**base.overrides, "bn_warmup": True}), 32, 3)
    bns = [m for m in warm.modules() if isinstance(m, BatchNormChannelsLast)]
    assert len(bns) == 2 and all(m.warmup for m in bns)
    assert not any(m.warmup for m in plain.modules() if isinstance(m, BatchNormChannelsLast))
    assert set(warm.state_dict()) - set(plain.state_dict()) == {"level_bn.num_updates", "stats_bn.num_updates"}
