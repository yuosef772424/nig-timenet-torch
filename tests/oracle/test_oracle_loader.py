"""P07: the oracle loader returns every artifact with its declared shape (MAJOR-2 handled) and complete keys."""

import pytest
import torch

from tests.oracle import oracle_loader as ol


def test_load_weights_reshapes_kendall_scalars():
    """ORACLE_REVIEW.md MAJOR-2: stored as (1,), declared () -> loader returns torch scalars."""
    w, g = ol.load_weights(), ol.load_grads()
    for name in ol.MAJOR2_NAMES:
        assert w[name].shape == () and g[name].shape == (), name
    # every other entry already had its declared shape (the loader would have raised otherwise)
    meta = ol.load_weights_meta()
    assert len(w) == len(meta) == 75
    for m in meta:
        key = m.get("attr_path") or m["name"]
        assert tuple(w[key].shape) == tuple(m["shape"]), key
    assert set(g) == {m.get("attr_path") or m["name"] for m in meta if m["trainable"]}


def test_load_layer_outputs_keys_match_layers_json():
    outs = ol.load_layer_outputs()
    keys = [k for m in ol.load_layers_meta() for k in m["output_keys"]]
    assert set(keys) == set(outs)
    for m in ol.load_layers_meta():
        for k, shape in zip(m["output_keys"], m["output_shapes"], strict=True):
            assert tuple(outs[k].shape) == tuple(shape), k


def test_other_artifacts_load():
    assert ol.load_inputs().shape == (8, 32, 6)
    assert set(ol.load_labels()) == {f"y_{t}_{k}" for t in ("high", "low", "close") for k in ("reg", "class")}
    assert len(ol.load_outputs()) == 24
    subs = ol.load_sublayers()
    assert set(subs) == {m["key"] for m in ol.load_sublayers_meta()}
    losses = ol.load_losses()
    assert losses["primary"] in losses["scenarios"]
    assert ol.load_config()["overrides"] == {"d_model": 16, "num_layers": 2}


def test_assert_close_reports_max_diff():
    x = torch.ones(3)
    ol.assert_close(x + 1e-7, x, "forward")
    with pytest.raises(AssertionError, match="max abs diff 1.000e-03"):
        ol.assert_close(x + 1e-3, x, "forward", "probe")
    with pytest.raises(AssertionError, match="shape"):
        ol.assert_close(torch.ones(2), x, "forward")
