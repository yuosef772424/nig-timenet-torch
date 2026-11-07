"""T4 trainer trace: log-only (training is unchanged), per-head gradient share with a known answer, Kendall and
schedule values in the history, identity unaffected."""

import math

import pytest
import torch

from nig_timenet_torch.trainer.trace import head_gradient_share
from nig_timenet_torch.trainer.trainer import FitSettings, Trainer
from tests.trainer.helpers import data, fit, make_trainer  # noqa: F401 (fixture `data`)


def test_trace_is_log_only_training_is_identical(data):  # noqa: F811
    off, on = make_trainer(), make_trainer()
    r_off = fit(off, data, FitSettings(epochs=2, batch_size=32))
    r_on = fit(on, data, FitSettings(epochs=2, batch_size=32, trace=True))
    for a, b in zip(off.model.state_dict().values(), on.model.state_dict().values(), strict=True):
        assert torch.equal(a, b), "trace=True changed the trained weights"
    assert all(not k.startswith("trace_") for k in r_off.history[0])
    base = {k: v for k, v in r_on.history[-1].items() if not k.startswith("trace_")}
    assert base == r_off.history[-1]


def test_trace_columns_and_shares(data):  # noqa: F811
    tr = make_trainer()
    r = fit(tr, data, FitSettings(epochs=2, batch_size=32, trace=True))
    row = r.history[-1]
    shares = {k: v for k, v in row.items() if k.startswith("trace_gradshare_")}
    assert set(shares) == {f"trace_gradshare_{t}_reg" for t in ("high", "low", "close")} | {
        f"trace_gradshare_{t}_class" for t in ("high", "low", "close")}
    assert sum(shares.values()) == pytest.approx(1.0) and all(v >= 0 for v in shares.values())
    for name in tr.scheduled_vars:
        assert f"trace_sched_{name}" in row
    assert tr.scheduled_vars and tr.kendall_task_names, "the oracle trainer config exercises schedules and Kendall"
    for t in tr.kendall_task_names:
        assert f"trace_kendall_weight_{t}" in row and f"trace_kendall_logvar_{t}" in row


def test_gradient_share_known_answer():
    """Two heads on one shared weight: loss_a = 3 w, loss_b = w  ->  shares 3/4 and 1/4."""

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            return {"y_a": self.w * x, "y_b": self.w * x}

    tr = Trainer.__new__(Trainer)
    tr.model, tr.uncertainty_layer, tr.kendall_task_names = Tiny(), None, []
    tr.compute_losses = lambda x, y, **kw: (None, None, {"a": 3 * tr.model.w, "b": 1 * tr.model.w}, {}, {})
    share = head_gradient_share(tr, None, None)
    assert share == pytest.approx({"a": 0.75, "b": 0.25})
    assert math.isclose(sum(share.values()), 1.0)


def test_identity_ignores_trace():
    from nig_timenet_torch.trainer.identity import build_run_identity

    tr = make_trainer()
    kw = dict(model_settings={"m": 1}, trainer_config=tr.config, optimizer=tr.optimizer, data_fingerprint="d",
              code_sha="c")
    a = build_run_identity(FitSettings(epochs=3), **kw)
    b = build_run_identity(FitSettings(epochs=5, trace=True), **kw)
    assert a.fingerprint() == b.fingerprint()
