"""model/taps.py: the inspection-only taps record every named layer of any registered architecture without changing
the outputs, the state_dict or training; tap_summary flags dead / constant features; signal_trace scores a known
signal per layer next to the raw input."""

import numpy as np
import pytest
import torch

from nig_timenet_torch.core.seeds import seed_everything
from nig_timenet_torch.model.architectures import ARCHITECTURES, ModelSettings, build_model
from nig_timenet_torch.model.taps import record_taps, signal_trace, tap_points, tap_summary

SEQ, FEAT = 16, 6


def _model(arch):
    seed_everything(0)
    m = build_model(ModelSettings(arch), SEQ, FEAT)
    return m.eval()


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES.names()))
def test_taps_record_every_layer_and_change_nothing(arch):
    m = _model(arch)
    x = torch.randn(8, SEQ, FEAT)
    with torch.no_grad():
        plain = m(x)
        before = {k: v.clone() for k, v in m.state_dict().items()}
        with record_taps(m) as taps:
            tapped = m(x)
        after = m(x)
    assert taps and set(taps) <= set(tap_points(m))
    assert plain.keys() == tapped.keys() and all(torch.equal(plain[k], tapped[k]) for k in plain)
    assert all(torch.equal(plain[k], after[k]) for k in plain)          # hooks removed, outputs unchanged
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())
    assert not any(m._forward_hooks for m in m.modules())


def test_unknown_tap_points_raise():
    with pytest.raises(KeyError), record_taps(_model("tiny_mlp"), ["no_such_layer"]):
        pass


def test_tap_summary_flags_dead_and_constant_features():
    taps = {"relu": torch.tensor([[0.0, 1.0, 2.0], [0.0, 3.0, 2.0]])}
    row = tap_summary(taps).iloc[0]
    assert row["dead_share"] == pytest.approx(1 / 3) and row["constant_share"] == pytest.approx(2 / 3)


def test_signal_trace_scores_a_planted_signal_on_the_raw_input():
    m = _model("tiny_mlp")
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(400, SEQ, FEAT, generator=gen)
    signal = x[:, -1, 0].numpy().astype(np.float64)                     # exactly linear in the raw window
    rep = signal_trace(m, x, {"planted": signal}, points=tap_points(m, leaves_only=True)[:2])
    raw = rep[rep["probe_point"] == "raw_input"].iloc[0]
    assert raw["mean"] > 0.95 and set(rep["signal"]) == {"planted"}
