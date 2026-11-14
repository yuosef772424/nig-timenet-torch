"""P33b: the GRU / level-passthrough (anti-memorization) encoder vs the SECOND oracle, reference/data_default/
(built with its constructor_kwargs verbatim: encoder="gru", stats_mode="symlog", level_passthrough=True,
level_norm="batch", ...). Weights exact (BatchNorm moving statistics included, as buffers), the 24 outputs and the
recorded per-layer outputs at TOLERANCE["forward"] in eval mode."""

from pathlib import Path

import pytest
import torch

from nig_timenet_torch.model.architectures import ANTI_MEMORIZATION_CONFIG, ARCHITECTURES
from nig_timenet_torch.model.builder import build_nig_timenet_v2
from tests.oracle import oracle_loader as ol

D = Path(__file__).resolve().parents[2] / "reference" / "data_default"


@pytest.fixture(scope="module")
def model():
    kw = dict(ol.load_config(data=D)["constructor_kwargs"])
    m = build_nig_timenet_v2(kw.pop("seq_len"), kw.pop("n_features"), **kw)
    m.load_oracle_weights(ol.load_weights(data=D))
    return m.eval()


def test_weight_count_matches_default_oracle(model):
    entries = [m for m in ol.load_weights_meta(data=D) if not m["name"].startswith("trainer/")]
    assert len(entries) == 67
    n_buffers = len(list(model.buffers()))  # 2 BatchNorms x (moving_mean, moving_variance)
    assert n_buffers == 4
    gru_torch_extra = 2  # each Keras GRU bias (2, 3u) becomes torch bias_ih + bias_hh
    assert len(list(model.parameters())) + n_buffers == len(entries) + gru_torch_extra


def test_gru_weights_round_trip_exact(model):
    w = ol.load_weights(data=D)
    gru = model.gru_1.cell
    k, rk, b = w["gru_1.cell.kernel"], w["gru_1.cell.recurrent_kernel"], w["gru_1.cell.bias"]
    u = gru.hidden_size
    for keras_gate, torch_gate in ((0, 1), (1, 0), (2, 2)):  # Keras z|r|h -> torch r|z|n
        ks, ts = slice(keras_gate * u, (keras_gate + 1) * u), slice(torch_gate * u, (torch_gate + 1) * u)
        ol.assert_close(gru.weight_ih_l0[ts], k[:, ks].t(), "weights", "weight_ih")
        ol.assert_close(gru.weight_hh_l0[ts], rk[:, ks].t(), "weights", "weight_hh")
        ol.assert_close(gru.bias_ih_l0[ts], b[0, ks], "weights", "bias_ih")
        ol.assert_close(gru.bias_hh_l0[ts], b[1, ks], "weights", "bias_hh")
    ol.assert_close(model.level_bn.moving_variance, w["level_bn.moving_variance"], "weights", "level_bn var")


def test_forward_matches_default_oracle_outputs(model):
    expected = ol.load_outputs(data=D)
    with torch.no_grad():
        got = model(ol.load_inputs(data=D))
    assert sorted(got) == sorted(expected) and len(got) == 24
    for k in sorted(expected):
        ol.assert_close(got[k], expected[k], "forward", k)


# each new P33b layer fed the oracle's OWN recorded input (isolates the layer; chaining the whole trunk from the raw
# input accumulates float32 differences — TF's approximate float tanh in input_clip alone is ~1.6e-6 off exact)
ISOLATED = [("level_symlog", "000_input_sequence"), ("level_bn", "003_level_symlog"),
            ("step_proj", "010_shape_and_level"), ("gru_norm_1", "011_step_proj"), ("gru_1", "012_gru_norm_1"),
            ("gru_2", "016_gru_norm_2"), ("final_norm", "019_gru_res_2"), ("stats_bn", "001_instance_norm__1")]


@pytest.mark.parametrize("name,src", ISOLATED)
def test_new_layers_match_default_oracle_in_isolation(model, name, src):
    lo = ol.load_layer_outputs(data=D)
    out_key = {m["name"]: m["output_keys"][0] for m in ol.load_layers_meta(data=D)}[name]
    with torch.no_grad():
        got = getattr(model, name)(lo[src])
    ol.assert_close(got, lo[out_key], "forward", name)


def test_gru_residual_wiring_matches_default_oracle(model):
    lo = ol.load_layer_outputs(data=D)
    with torch.no_grad():
        got = lo["011_step_proj"] + model.gru_1(model.gru_norm_1(lo["011_step_proj"]))
    ol.assert_close(got, lo["015_gru_res_1"], "forward", "gru_res_1")


def test_anti_memorization_preset_is_the_default_oracle_overrides():
    assert ANTI_MEMORIZATION_CONFIG == ol.load_config(data=D)["overrides"]
    meta = ARCHITECTURES.entry("nig_timenet_v2").meta
    assert meta["anti_memorization_config"] == ANTI_MEMORIZATION_CONFIG
    assert ARCHITECTURES.names() == ("nig_timenet_v2", "tiny_mlp", "chart_cnn", "conv1d_net",
                                    "no_dense_net")  # a preset, not a second entry


def test_default_config_plus_preset_rebuilds_the_default_oracle():
    from nig_timenet_torch.model.architectures import NIG_TIMENET_V2_DEFAULT_CONFIG

    kw = dict(ol.load_config(data=D)["constructor_kwargs"])
    kw.pop("seq_len"), kw.pop("n_features")
    assert {**NIG_TIMENET_V2_DEFAULT_CONFIG, **ANTI_MEMORIZATION_CONFIG} == kw
