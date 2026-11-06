"""The panel autoregressive candle transformer contract: shapes, the two causal directions (a coin's
own future, and the OTHER coins' future) and the panel-wide lockstep rollout."""

import pytest
import torch

from nig_timenet_torch.model.panel_ar_candle import PanelArCandleTransformer, rollout


def _model(n_coins: int = 3, seed: int = 0) -> PanelArCandleTransformer:
    torch.manual_seed(seed)
    return PanelArCandleTransformer(n_features=5, n_coins=n_coins, context_len=8, horizon_len=4,
                                    d_model=32, num_layers=1, num_heads=4, num_kv_heads=2, cross_layers=1)


def test_forward_shapes():
    m = _model()
    x = torch.randn(2, 3, 12, 5)
    out = m(x)
    assert out["mu"].shape == (2, 3, 12, 5) and out["log_sigma"].shape == (2, 3, 12, 5)
    assert out["dir_logit"].shape == (2, 3, 12)


def test_causal_no_future_leak_own_and_cross():
    m = _model().eval()
    x = torch.randn(2, 3, 12, 5)
    with torch.no_grad():
        before = m(x)
        x2 = x.clone()
        x2[:, 1:, 6:] += 5.0  # shock coins 2..3 (and coin 1 untouched) AFTER position 6 only
        after = m(x2)
    # every coin's predictions up to 6 are unchanged (own future and the other coins' future both masked)
    assert torch.allclose(before["mu"][:, :, :6], after["mu"][:, :, :6], atol=1e-5)
    assert not torch.allclose(before["mu"][:, :, 6:], after["mu"][:, :, 6:], atol=1e-5)


def test_cross_coin_attention_actually_mixes():
    m = _model().eval()
    x = torch.randn(1, 3, 8, 5)
    with torch.no_grad():
        out = m(x)
        x2 = x.clone()
        x2[:, 1] += 3.0  # a different coin's present changes this coin's prediction: cross-coin mixing
        out2 = m(x2)
    assert not torch.allclose(out["mu"][:, 0, -1], out2["mu"][:, 0, -1], atol=1e-6)


def test_rollout_shapes_and_lockstep_append():
    m = _model().eval()
    ctx = torch.randn(2, 3, 8, 5)
    out = rollout(m, ctx, horizon_len=4)
    assert out["mu"].shape == (2, 3, 4, 5) and out["dir_logit"].shape == (2, 3, 4)


def test_padded_slots_are_masked():
    m = _model().eval()
    x = torch.randn(2, 3, 8, 5)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    with torch.no_grad():
        out = m(x, mask)
    assert bool((out["mu"][0, 2] == 0).all())  # batch 0's third slot is padding: zeroed
    assert bool((out["mu"][1, 2] != 0).any())   # batch 1's third slot is a real coin: alive


def test_zero_coins_refused():
    with pytest.raises(ValueError, match="n_coins"):
        PanelArCandleTransformer(n_features=5, n_coins=0)


def test_gru_encoder_swap_keeps_the_panel_contract():
    """The temporal trunk is pluggable in the panel too: encoder='gru' keeps shapes, both causal
    directions and the lockstep rollout; an unknown name is a loud error."""
    torch.manual_seed(0)
    m = PanelArCandleTransformer(n_features=5, n_coins=3, context_len=8, horizon_len=4, d_model=32,
                                 num_layers=2, cross_layers=1, encoder="gru").eval()
    x = torch.randn(2, 3, 12, 5)
    out = m(x)
    assert out["mu"].shape == (2, 3, 12, 5)
    with torch.no_grad():
        before = m(x)
        x2 = x.clone()
        x2[:, 1:, 6:] += 5.0
        after = m(x2)
    assert torch.allclose(before["mu"][:, :, :6], after["mu"][:, :, :6], atol=1e-5)
    assert rollout(m, torch.randn(2, 3, 8, 5), horizon_len=4)["mu"].shape == (2, 3, 4, 5)
    with pytest.raises(ValueError, match="unknown encoder"):
        PanelArCandleTransformer(n_features=5, n_coins=3, encoder="lstm")
