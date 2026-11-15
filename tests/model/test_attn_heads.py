"""model/attn_heads.py (experiment, owner 2026-11-15): the attention NIG head. The default ``nig_head='dense'`` is the
old model (state_dict keys unchanged); the scalar heads keep the output contract, honour the label scale switch and
their attention probabilities are proper distributions."""

import pytest
import torch

from nig_timenet_torch.model.architectures import ANTI_MEMORIZATION_CONFIG, ModelSettings, build_model
from nig_timenet_torch.model.attn_heads import ScalarNIGHead, ScalarOutputAttention, n_encoder_tokens
from nig_timenet_torch.model.nig_layers import NIGHead
from nig_timenet_torch.model.scalers import fit_scalers

T, F_ = 32, 3


def _model(head, encoder="gru", **over):
    ms = ModelSettings("nig_timenet_v2", {**ANTI_MEMORIZATION_CONFIG, "encoder": encoder, "nig_head": head, **over})
    return build_model(ms, T, F_)


def test_dense_is_the_old_model_and_the_scalar_heads_add_only_their_own_weights():
    dense = _model("dense")
    explicit = build_model(ModelSettings("nig_timenet_v2", dict(ANTI_MEMORIZATION_CONFIG)), T, F_)
    assert list(dense.state_dict()) == list(explicit.state_dict())
    sc = _model("scalar_joint")
    assert any(k.startswith("head_seq_norm") for k in sc.state_dict()) and not hasattr(dense, "head_seq_norm")
    assert not any(k.startswith("nig_high.fc") for k in sc.state_dict())


@pytest.mark.parametrize("encoder", ["gru", "transformer", "tcn"])
@pytest.mark.parametrize("head", ["scalar_joint", "scalar_factorized"])
def test_output_contract_equals_dense_and_gradients_flow(encoder, head):
    dense, sc = _model("dense", encoder), _model(head, encoder)
    x = torch.randn(6, T, F_) * 0.02
    od, os_ = dense(x), sc(x)
    assert set(od) == set(os_) and all(od[k].shape == os_[k].shape for k in od)
    assert all(torch.isfinite(v).all() for v in os_.values())
    sum(v.sum() for v in os_.values()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for n, p in sc.named_parameters()
               if n.startswith("nig_") and p.requires_grad)
    assert torch.isfinite(sc.regularization_loss()) and sc.regularization_loss().item() > 0


@pytest.mark.parametrize("encoder", ["gru", "transformer", "tcn"])
def test_token_count_matches_the_encoder(encoder):
    m = _model("scalar_joint", encoder)
    head = m.nig_high.attn
    assert head.t == 1 + n_encoder_tokens(encoder, T, 4, 2)
    m(torch.randn(2, T, F_) * 0.02)                                # the length check inside the head passes


@pytest.mark.parametrize("mode", ["joint", "factorized"])
def test_attention_probabilities_are_distributions(mode):
    a = ScalarOutputAttention(16, 5, 4, mode)
    torch.nn.init.normal_(a.time_bias, std=1.0)
    torch.nn.init.normal_(a.feat_logits, std=1.0)
    x = torch.randn(3, 5, 16)
    p = a.probs(a.wk(x))
    assert p.shape == (3, 4, 5, 16) and (p >= 0).all()
    assert torch.allclose(p.sum(dim=(2, 3)), torch.ones(3, 4), atol=1e-5)
    with pytest.raises(ValueError):
        a(torch.randn(3, 6, 16))                                   # a wrong sequence length raises


def test_scale_switch_matches_nighead_semantics():
    h = ScalarNIGHead(8, 3, scale=0.5)
    base = ScalarNIGHead(8, 3, scale=None)
    base.load_state_dict({k: v for k, v in h.state_dict().items() if "fitted" not in k}, strict=False)
    x = torch.randn(4, 3, 8)
    (mu, nu, al, be), (mu0, nu0, al0, be0) = h(x), base(x)
    assert torch.allclose(mu, 0.5 * mu0) and torch.allclose(be, 0.25 * be0)
    assert torch.allclose(nu, nu0) and torch.allclose(al, al0)
    assert ScalarNIGHead(8, 3).nu_min == NIGHead(8).nu_min and ScalarNIGHead(8, 3).alpha_min == NIGHead(8).alpha_min


def test_fitted_label_scale_and_input_scaler_work_with_the_scalar_head():
    m = _model("scalar_joint", nig_scale="train_std", input_scaler="train_std")
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, T, F_, generator=g) * 0.02
    y = {f"y_{t}_reg": torch.randn(64, generator=g) * 0.03 for t in ("high", "low", "close")}
    fit_scalers(m, x, y)
    assert all(torch.isfinite(v).all() for v in m(x).values())


def test_bad_options_raise():
    with pytest.raises(ValueError, match="nig_head"):
        _model("dense_ish")
    ms = ModelSettings("nig_timenet_v2", {**ANTI_MEMORIZATION_CONFIG, "nig_head": "scalar_joint"})
    with pytest.raises(NotImplementedError, match="one timeframe"):
        build_model(ms, {"4h": T, "1d": T}, {"4h": F_, "1d": F_})
    with pytest.raises(ValueError, match="mode"):
        ScalarOutputAttention(8, 3, 4, "nope")


def test_linear_head_is_one_layer_with_the_dense_contract_and_reads_the_trunk():
    dense, lin = _model("dense"), _model("linear")
    keys = [k for k in lin.state_dict() if k.startswith("nig_high.")]
    assert keys == ["nig_high.out.weight", "nig_high.out.bias"] and lin.nig_high.out.weight.shape == (4, 64)
    assert not hasattr(lin, "head_seq_norm") and float(lin.regularization_loss().detach()) > 0  # linear_path only
    x = torch.randn(6, T, F_) * 0.02
    od, ol = dense(x), lin(x)
    assert set(od) == set(ol) and all(torch.isfinite(v).all() for v in ol.values())
    assert (ol["y_high_nu"] > 0.1).all() and (ol["y_high_alpha"] > 2.0).all()
    assert "nig_high.fc" not in lin.activation_points


def test_unknown_nig_head_raises():
    with pytest.raises(ValueError, match="nig_head must be one of"):
        _model("mlp")
