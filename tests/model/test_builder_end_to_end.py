"""P15 (M2 exit gate): the full torch model vs the TF oracle (weights exact, all 24 outputs at forward tolerance).
Built with reference/data/config.json constructor_kwargs verbatim."""

import pytest
import torch

from nig_timenet_torch.model.builder import build_nig_timenet_v2, torch_param_name
from tests.oracle import oracle_loader as ol


def constructor_kwargs() -> dict:
    return dict(ol.load_config()["constructor_kwargs"])


@pytest.fixture(scope="module")
def model():
    kw = constructor_kwargs()
    m = build_nig_timenet_v2(kw.pop("seq_len"), kw.pop("n_features"), **kw)
    m.load_oracle_weights(ol.load_weights())
    return m.eval()


def test_weight_count_matches_oracle(model):
    model_entries = [m for m in ol.load_weights_meta() if not m["name"].startswith("trainer/")]
    assert len(model_entries) == 72
    assert len(list(model.parameters())) == 72


def test_load_oracle_weights_exact(model):
    params = dict(model.named_parameters())
    for path, value in ol.load_weights().items():
        if path.startswith("trainer/"):
            continue
        p = params[torch_param_name(path)]
        if path.endswith(".kernel"):
            expected = value.permute(2, 1, 0) if value.ndim == 3 else value.t()
        else:
            expected = value
        ol.assert_close(p, expected, "weights", path)


def test_load_oracle_weights_rejects_partial():
    kw = constructor_kwargs()
    m = build_nig_timenet_v2(kw.pop("seq_len"), kw.pop("n_features"), **kw)
    w = dict(ol.load_weights())
    w.pop("block_1.attn.wq.kernel")
    with pytest.raises(KeyError, match="block_1.attn.wq.weight"):
        m.load_oracle_weights(w)


def test_unported_options_raise():
    with pytest.raises(NotImplementedError, match="P34"):
        build_nig_timenet_v2(32, 6, encoder="gru")
    with pytest.raises(NotImplementedError, match="P33b"):
        build_nig_timenet_v2(32, 6, stats_mode="symlog")


def test_forward_matches_outputs_npz(model):
    expected = ol.load_outputs()
    with torch.no_grad():
        got = model(ol.load_inputs())
    assert sorted(got) == sorted(expected) and len(got) == 24
    for k in sorted(expected):
        ol.assert_close(got[k], expected[k], "forward", k)
