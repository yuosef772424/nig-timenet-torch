"""The autoregressive candle transformer contract: shapes, the causal mask (no future leakage) and the
text-style rollout that feeds the model's own prediction back."""

import pytest
import torch

from nig_timenet_torch.model.ar_candle_transformer import ArCandleTransformer, rollout


def _model(context: int = 32, horizon: int = 16, seed: int = 0) -> ArCandleTransformer:
    torch.manual_seed(seed)
    return ArCandleTransformer(n_features=5, context_len=context, horizon_len=horizon,
                               d_model=32, num_layers=2, num_heads=4, num_kv_heads=2)


def test_forward_shapes():
    m = _model()
    x = torch.randn(3, 48, 5)
    out = m(x)
    assert out["mu"].shape == (3, 48, 5) and out["log_sigma"].shape == (3, 48, 5)
    assert out["dir_logit"].shape == (3, 48)


def test_causal_no_future_leakage():
    m = _model().eval()
    x = torch.randn(2, 48, 5)
    with torch.no_grad():
        before = m(x)
        x2 = x.clone()
        x2[:, 20:] += 5.0  # shock only the candles after position 20
        after = m(x2)
    assert torch.allclose(before["mu"][:, :20], after["mu"][:, :20], atol=1e-6)
    assert not torch.allclose(before["mu"][:, 20:], after["mu"][:, 20:], atol=1e-6)


def test_rollout_shapes_and_appends_own_predictions():
    m = _model().eval()
    ctx = torch.randn(4, 32, 5)
    out = rollout(m, ctx, horizon_len=16)
    assert out["mu"].shape == (4, 16, 5) and out["dir_logit"].shape == (4, 16)
    assert out["inputs"].shape == (4, 16, 5)
    # the second fed-back input is the model's own first prediction (generation, like a text model)
    assert torch.allclose(out["inputs"][:, 0], out["mu"][:, 0])


def test_rollout_any_horizon():
    m = _model().eval()
    ctx = torch.randn(2, 32, 5)
    for h in (1, 4, 8):
        assert rollout(m, ctx, horizon_len=h)["mu"].shape == (2, h, 5)


def test_rollout_longer_than_the_trained_horizon_extends_positions():
    """A FREE eval horizon may exceed the trained one: the position table extends itself by repeating
    its last entry, the rollout still produces a step per requested candle and the past is unchanged."""
    m = _model(context=8, horizon=4).eval()
    x = torch.randn(2, 12, 5)
    out = m(x)  # 12 > 8 + 4: padded, not refused
    assert out["mu"].shape == (2, 12, 5)
    again = m(x)  # a second padded pass on the SAME input is identical in eval
    assert torch.allclose(out["mu"], again["mu"])
    r = rollout(m, torch.randn(2, 8, 5), horizon_len=7)
    assert r["mu"].shape == (2, 7, 5)


def test_gru_encoder_swap():
    """The trunk is pluggable: encoder='gru' keeps the same contract (shapes, causality, rollout) - the
    ANTI_MEMORIZATION preset's cheap trunk - and an unknown name is a loud error."""
    m = ArCandleTransformer(n_features=5, context_len=8, horizon_len=4, d_model=32, num_layers=2,
                           encoder="gru").eval()
    x = torch.randn(2, 12, 5)
    out = m(x)
    assert out["mu"].shape == (2, 12, 5) and out["dir_logit"].shape == (2, 12)
    with torch.no_grad():
        before = m(x)
        x2 = x.clone()
        x2[:, 6:] += 5.0
        after = m(x2)
    assert torch.allclose(before["mu"][:, :6], after["mu"][:, :6], atol=1e-5)  # GRU is causal by design
    assert rollout(m, torch.randn(2, 8, 5), horizon_len=4)["mu"].shape == (2, 4, 5)
    with pytest.raises(ValueError, match="unknown encoder"):
        ArCandleTransformer(n_features=5, encoder="lstm")
