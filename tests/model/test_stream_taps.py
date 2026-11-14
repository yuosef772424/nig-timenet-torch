"""model/taps.py stream_taps + NigTimeNetV2._tap: the residual / internal streams of every encoder are recorded in
forward order, recording changes nothing, an edit ablates one stream, and nothing is left over afterwards."""

import pytest
import torch

from nig_timenet_torch.core.seeds import seed_everything
from nig_timenet_torch.model.architectures import ModelSettings, build_model
from nig_timenet_torch.model.taps import stream_taps

SEQ, FEAT = 32, 4
SMALL = {"d_model": 16, "num_layers": 2, "num_heads": 2, "num_kv_heads": 1, "head_hidden": 16, "class_head_hidden": 8}
MUST_HAVE = {"enc_in", "enc_block_1", "enc_out", "pooled", "readout", "linear_path", "branch_out"}


def _model(encoder):
    seed_everything(0)
    return build_model(ModelSettings("nig_timenet_v2", {**SMALL, "encoder": encoder}), SEQ, FEAT).eval()


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("encoder", ["gru", "transformer", "tcn"])
def test_tap_names_in_forward_order(encoder):
    m, x = _model(encoder), torch.randn(6, SEQ, FEAT)
    with torch.no_grad(), stream_taps(m) as taps:
        m(x)
    names = list(taps)
    assert names[:2] == ["input", "inst_norm"] and names[-1] == "trunk"
    assert MUST_HAVE <= set(names)
    order = ["enc_in", "enc_block_1", "enc_out", "pooled", "readout", "linear_path", "branch_out", "trunk"]
    assert [names.index(n) for n in order] == sorted(names.index(n) for n in order)


@pytest.mark.parametrize("encoder", ["gru", "transformer", "tcn"])
def test_recording_does_not_change_outputs_and_leaves_no_state(encoder):
    m, x = _model(encoder), torch.randn(6, SEQ, FEAT)
    with torch.no_grad():
        plain = m(x)
        with stream_taps(m) as taps:
            recorded = m(x)
        assert taps and _same(plain, recorded)
        assert m._stream is None and m._stream_edit == {}
        assert _same(plain, m(x))


@pytest.mark.parametrize("encoder", ["gru", "transformer", "tcn"])
def test_edit_replaces_a_stream_and_leaves_nothing_behind(encoder):
    m, x = _model(encoder), torch.randn(6, SEQ, FEAT)
    with torch.no_grad():
        plain = m(x)
        with stream_taps(m, edit={"branch_out": lambda v: torch.zeros_like(v)}) as taps:
            edited = m(x)
        assert float(taps["branch_out"].abs().sum()) == 0.0
        assert not _same(plain, edited)
        assert _same(plain, m(x))                       # a later plain forward is the original again


def test_edit_of_an_unrecorded_name_changes_nothing():
    m, x = _model("gru"), torch.randn(4, SEQ, FEAT)
    with torch.no_grad():
        plain = m(x)
        with stream_taps(m, edit={"no_such_stream": lambda v: v * 0}):
            assert _same(plain, m(x))


def test_tiny_mlp_has_no_stream_taps():
    seed_everything(0)
    m = build_model(ModelSettings("tiny_mlp"), SEQ, FEAT)
    with pytest.raises(TypeError), stream_taps(m):
        pass
