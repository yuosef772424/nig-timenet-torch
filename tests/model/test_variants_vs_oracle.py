"""B9 (P40-P43): the model variants vs their TF oracles in reference/data_variants/<case>/
(reference/export_reference_variants.py). Weights exact, outputs at TOLERANCE["forward"] in eval mode, the multiclass
loss at TOLERANCE["loss"]."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nig_timenet_torch.model.builder import build_nig_timenet_v2, tcn_n_blocks
from nig_timenet_torch.model.panel import build_panel_model, panel_encoder
from nig_timenet_torch.trainer.tasks import classification_task_loss, with_defaults
from tests.oracle import oracle_loader as ol

D = Path(__file__).resolve().parents[2] / "reference" / "data_variants"


def _npz(case, name):
    with np.load(D / case / name, allow_pickle=False) as z:
        return {k: torch.from_numpy(np.array(z[k], copy=True)) for k in z.files}


def _cfg(case):
    return json.loads((D / case / "config.json").read_text())


def _nig_model(case):
    cfg = _cfg(case)
    m = build_nig_timenet_v2(cfg["seq_len"], cfg["n_features"], **cfg["constructor_kwargs"])
    m.load_oracle_weights(_npz(case, "weights.npz"))
    return m.eval()


def _assert_outputs(got, case):
    expected = _npz(case, "outputs.npz")
    assert sorted(got) == sorted(expected)
    for k in sorted(expected):
        ol.assert_close(got[k], expected[k], "forward", f"{case}:{k}")


@pytest.mark.parametrize("case", ["tcn", "multitf", "multiclass"])
def test_every_oracle_weight_loads_and_every_parameter_is_assigned(case):
    m = _nig_model(case)  # load_oracle_weights raises on any unmatched / missing parameter
    assert len(list(m.parameters())) == len(_npz(case, "weights.npz"))


def test_tcn_forward_matches_oracle():
    m = _nig_model("tcn")
    assert m.branch_blocks[""] == tcn_n_blocks(32) == 4
    with torch.no_grad():
        got = m(_npz("tcn", "inputs.npz")["input_sequence"])
    assert len(got) == 24
    _assert_outputs(got, "tcn")


def test_tcn_is_causal_along_the_sequence():
    m = build_nig_timenet_v2(32, 6, d_model=16, num_layers=2, encoder="tcn").eval()
    x = torch.randn(2, 32, 16)
    with torch.no_grad():
        full = x.clone()
        for i in range(m.branch_blocks[""]):
            full = full + m._m(f"tcn_conv_{i + 1}")(m._m(f"tcn_norm_{i + 1}")(full))
        cut = x.clone()
        cut[:, 20:] = 99.0  # changing bars after 19 must not move positions <= 19
        for i in range(m.branch_blocks[""]):
            cut = cut + m._m(f"tcn_conv_{i + 1}")(m._m(f"tcn_norm_{i + 1}")(cut))
    torch.testing.assert_close(full[:, :20], cut[:, :20])


def test_multitf_forward_matches_oracle_and_input_forms_agree():
    m = _nig_model("multitf")
    xs = _npz("multitf", "inputs.npz")
    assert m.timeframes == ("1d", "4h")
    with torch.no_grad():
        od = m(xs)
        ol_ = m([xs["1d"], xs["4h"]])
        ot = m((xs["1d"], xs["4h"]))
    _assert_outputs(od, "multitf")
    for k in od:
        torch.testing.assert_close(od[k], ol_[k], rtol=0, atol=0)
        torch.testing.assert_close(od[k], ot[k], rtol=0, atol=0)


def test_multitf_names_are_suffixed_per_timeframe_and_branches_are_separate():
    # adapted from upstream tests/test_multi_tf.py::test_multi_tf_build_and_forward
    cfg = dict(d_model=16, num_layers=2, encoder="gru", stats_mode="symlog", level_passthrough=True,
               level_norm="batch", feature_dropout=0.1, input_noise_std=0.1)
    one = build_nig_timenet_v2(32, 7, **cfg)
    two = build_nig_timenet_v2({"1h": 32, "4h": 24}, 7, **cfg).eval()
    names = {n.split(".")[0] for n, _ in two.named_modules() if n}
    for tf in ("1h", "4h"):
        for base in ("instance_norm", "feature_dropout", "input_noise", "level_symlog", "level_bn", "gru_1",
                     "readout_fc", "linear_path", "stats_bn"):
            assert f"{base}_{tf}" in names, f"{base}_{tf}"
    assert "instance_norm" not in names
    n_one = sum(p.numel() for p in one.parameters())
    assert sum(p.numel() for p in two.parameters()) > 1.7 * n_one
    x1, x4 = torch.randn(6, 32, 7), torch.randn(6, 24, 7)
    with torch.no_grad():
        od, ob = two({"1h": x1, "4h": x4}), two({"1h": x1, "4h": x4 + 1.0})
        o16 = two({"1h": x1.half(), "4h": x4.half()})
    assert set(od) == set(one.eval()(x1)) and all(v.shape == (6, 1) for v in od.values())
    assert not torch.allclose(od["y_high"], ob["y_high"])  # the second branch reaches the output
    for k in od:
        torch.testing.assert_close(od[k], o16[k], rtol=0, atol=5e-2)


def test_multitf_validation():
    with pytest.raises(ValueError, match="different timeframes"):
        build_nig_timenet_v2({"1h": 32, "4h": 16}, {"1h": 7})
    m = build_nig_timenet_v2({"1h": 32, "4h": 16}, 7, d_model=16, num_layers=1)
    with pytest.raises(KeyError, match="4h"):
        m({"1h": torch.randn(2, 32, 7)})
    single = build_nig_timenet_v2({"1h": 32}, 7, d_model=16, num_layers=1)  # one timeframe = the old graph
    assert single.timeframes is None and hasattr(single, "instance_norm")


def test_multiclass_forward_matches_oracle():
    m = _nig_model("multiclass")
    with torch.no_grad():
        got = m(_npz("multiclass", "inputs.npz")["input_sequence"])
    assert got["y_close_class_logits"].shape == (8, 4)
    torch.testing.assert_close(got["y_close_class_logits"].sum(-1), torch.ones(8), rtol=0, atol=1e-6)
    _assert_outputs(got, "multiclass")


@pytest.mark.parametrize("ls", [0.0, 0.1])
@pytest.mark.parametrize("weighted", [False, True])
def test_multiclass_loss_matches_upstream_classification_task_loss(ls, weighted):
    inp = _npz("multiclass", "inputs.npz")
    out = _npz("multiclass", "outputs.npz")
    exp = json.loads((D / "multiclass" / "losses.json").read_text())[f"ls={ls},weighted={weighted}"]
    cfg = with_defaults("classification", {"output_keys": {"logits": "y_close_class_logits"}, "binary": False,
                                           "label_smoothing": ls})
    loss, stats = classification_task_loss(inp["y_close_class"], out, cfg, {},
                                           sample_weight=inp["sample_weight"] if weighted else None)
    ol.assert_close(loss, torch.tensor(exp["loss"]["value"]), "loss", "multiclass loss")
    ol.assert_close(stats["accuracy"], torch.tensor(exp["accuracy"]["value"]), "loss", "multiclass accuracy")


def test_binary_and_multiclass_on_one_target_is_refused():
    with pytest.raises(ValueError, match="share the Keras layer names"):
        build_nig_timenet_v2(32, 6, head_types={"close": ["binary_classification", "multiclass_classification"]})


def _panel(case):
    cfg = _cfg(case)
    enc = panel_encoder(cfg["seq_len"], cfg["n_features"], **cfg["encoder_constructor_kwargs"])
    m = build_panel_model(enc, **cfg["panel_config"])
    m.load_oracle_weights(_npz(case, "weights.npz"))
    return m.eval()


@pytest.mark.parametrize("case", ["panel", "panel_no_attention"])
def test_panel_forward_matches_oracle(case):
    m = _panel(case)
    n_enc = sum(1 for k in _npz(case, "weights.npz") if k.startswith("encoder."))
    assert n_enc == len(list(m.encoder.parameters()))
    with torch.no_grad():
        got = m(_npz(case, "inputs.npz"))
    assert {k: tuple(v.shape) for k, v in got.items()} == {"logit": (10, 3), "mu": (10, 3)}
    _assert_outputs(got, case)


@pytest.mark.parametrize("attention", [True, False])
def test_panel_masking_day_isolation_and_coin_order(attention):
    # adapted from upstream cross_asset/selftest.py::test_masking
    torch.manual_seed(1)
    enc = panel_encoder(8, 5, d_model=16, num_layers=1)
    m = build_panel_model(enc, d_model=16, num_heads=4, cross_attention=attention, dropout=0.0).eval()
    sizes = (4, 2, 6)
    day = torch.cat([torch.full((n,), d) for d, n in enumerate(sizes)])
    pos = torch.cat([torch.arange(n) for n in sizes])
    x = torch.randn(len(day), 8, 5)
    with torch.no_grad():
        o_big = m({"x": x, "day": day, "pos": pos})["logit"]
        rows = day == 1
        alone = {"x": x[rows], "day": torch.zeros(2, dtype=torch.long), "pos": pos[rows]}
        o_alone = m(alone)["logit"]
        torch.testing.assert_close(o_big[rows], o_alone, rtol=0, atol=1e-5)  # padding / other days invisible
        x2 = alone["x"].clone()
        x2[0] += 3.0
        o2 = m({**alone, "x": x2})["logit"]
        assert (not torch.allclose(o2[1:], o_alone[1:], atol=1e-6)) == attention  # same-day coins interact
        perm = torch.tensor([1, 0])
        o3 = m({**alone, "pos": alone["pos"][perm]})["logit"]
        torch.testing.assert_close(o3, o_alone, rtol=0, atol=1e-5)  # no positional code across coins
    assert torch.isfinite(o_big).all()


def test_panel_with_two_timeframe_encoder():
    enc = panel_encoder({"1h": 16, "4h": 8}, 5, d_model=16, num_layers=1)
    m = build_panel_model(enc, d_model=16, num_heads=4).eval()
    assert hasattr(m, "enc_proj")  # encoder width 32 != d_model 16
    out = m({"x": {"1h": torch.randn(5, 16, 5), "4h": torch.randn(5, 8, 5)}, "day": torch.tensor([0, 0, 1, 1, 1]),
             "pos": torch.tensor([0, 1, 0, 1, 2])})
    assert out["logit"].shape == (5, 3) and torch.isfinite(out["mu"]).all()
