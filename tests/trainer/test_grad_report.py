"""Gate A (trainer/grad_report.py) and the representative batch set (trainer/batch_set.py): a correct small model
passes the wiring checks, planted defects fail the right check, the report leaves the model and optimizer as they
were, and the trace on the batch set is log-only (ROADMAP G2)."""

import copy

import numpy as np
import pytest
import torch

from nig_timenet_torch.model.architectures import ModelSettings, TinyMLP
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.trainer.batch_set import representative_batches
from nig_timenet_torch.trainer.grad_report import grad_report, worst
from nig_timenet_torch.trainer.trainer import FitSettings, Trainer
from nig_timenet_torch.workflow import flat_run
from nig_timenet_torch.workflow.settings import RunSettings

TARGETS = ("high", "low", "close")
N, T, F = 1024, 8, 3


def _data(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, T, F, generator=g) * 0.02
    vol = x.abs().mean(dim=(1, 2))
    y = {"y_high_reg": vol + 0.01 * torch.rand(N, generator=g), "y_low_reg": -vol - 0.01 * torch.rand(N, generator=g),
         "y_close_reg": 0.02 * torch.randn(N, generator=g)}
    y.update({f"y_{t}_class": (y[f"y_{t}_reg"] > y[f"y_{t}_reg"].median()).float() for t in TARGETS})
    return x, y


def _trainer(model=None, seed=0):
    torch.manual_seed(seed)
    ms = ModelSettings("tiny_mlp", {})
    tr = flat_run.make_trainer(RunSettings(run_dir="x", model=ms), ms, TARGETS, T, F, torch.device("cpu"))
    if model is not None:
        tr = Trainer(model, tr.config, optimizer=lambda ps: torch.optim.AdamW(ps, lr=1e-3, weight_decay=1e-4))
    return tr


def _batches():
    coins = np.repeat(np.arange(32), N // 32)
    times = np.tile(np.arange(N // 32), 32)
    return representative_batches(coins, times, 64, seed=0)


class DetachedClose(TinyMLP):
    """A planted wiring defect: every output of the close NIG head is detached from the graph."""

    def forward(self, x):
        out = super().forward(x)
        keys = set(head_spec("nig_regression", "close").output_keys.values())
        return {k: (v.detach() if k in keys else v) for k, v in out.items()}


def test_batch_set_is_stratified_seeded_and_not_the_first_rows():
    coins = np.repeat(np.arange(40), 100)                       # coin-major, as the artifact
    times = np.tile(np.arange(100), 40)
    b = representative_batches(coins, times, 256, seed=3)
    assert len(b) == 16 and all(len(x) == 256 and len(set(x)) == 256 for x in b)
    assert all(len(np.unique(coins[x])) >= 32 for x in b), "every batch holds rows of >= 32 coins"
    assert all(times[x].min() < 10 and times[x].max() >= 90 for x in b), "every batch spans the time range"
    assert not np.array_equal(b[0], np.arange(256))
    again = representative_batches(coins, times, 256, seed=3)
    assert all(np.array_equal(p, q) for p, q in zip(b, again, strict=True))
    few = representative_batches(np.repeat([0, 1, 2], 10), np.tile(np.arange(10), 3), 25, n_batches=2)
    assert all(len(x) == 25 and len(set(x)) == 25 for x in few), "small coins: topped up, never a repeated row"


def test_correct_small_model_passes_the_wiring_checks_and_state_is_unchanged():
    x, y = _data()
    tr = _trainer()
    model0 = copy.deepcopy(tr.model.state_dict())
    opt0 = copy.deepcopy(tr.optimizer.state_dict())
    rep = grad_report(tr, x, y, _batches())
    c = rep["checks"]
    assert c["A1"]["verdict"] == "pass"
    assert c["A2"]["verdict"] in ("pass", "warn") and set(c["A2"]["dead"]) == set(c["A2"]["scheduled_off"]), \
        "only the confidence heads (lambda_calib starts at 0) are without a gradient at step 0"
    assert all(n.startswith("conf.") for n in c["A2"]["dead"])
    assert c["A4"]["verdict"] in ("pass", "warn") and 1e-5 < c["A4"]["value"] < 1e-1
    assert set(c["A5"]["value"]) == set(tr.target_names) and sum(c["A5"]["value"].values()) == pytest.approx(1.0)
    assert set(rep["checks"]) == {f"A{i}" for i in range(1, 8)} and rep["verdict"] == worst(
        v["verdict"] for v in c.values())
    for k, v in tr.model.state_dict().items():
        assert torch.equal(v, model0[k]), f"grad_report changed {k}"
    assert tr.optimizer.state_dict()["state"] == opt0["state"], "the A4 step was not undone"


def test_detached_head_fails_a2():
    x, y = _data()
    torch.manual_seed(0)
    rep = grad_report(_trainer(DetachedClose(T, F)), x, y, _batches())
    a2 = rep["checks"]["A2"]
    assert a2["verdict"] == "fail"
    # nig.close still gets a gradient through the ordered high / low means; the confidence head of close does not,
    # and is NOT explained by the lambda_calib schedule (it stays dead at the schedule's end value)
    assert {n.split(".")[1] for n in set(a2["dead"]) - set(a2["scheduled_off"])} == {"close"}


def test_non_finite_fails_a1_and_dead_units_fail_a7():
    x, y = _data()
    tr = _trainer()
    with torch.no_grad():
        tr.model.fc1.bias.fill_(-1e3)                       # every fc1 unit's pre-activation is negative
    assert grad_report(tr, x, y, _batches())["checks"]["A7"]["verdict"] == "fail"
    tr = _trainer()
    with torch.no_grad():
        tr.model.fc2.weight[0, 0] = float("nan")
    assert grad_report(tr, x, y, _batches())["checks"]["A1"]["verdict"] == "fail"


def test_trace_on_the_batch_set_is_log_only():
    x, y = _data()
    fs = FitSettings(epochs=2, batch_size=128, trace=True)
    a, b = _trainer(), _trainer()
    ra = a.fit(x, y, x[:256], {k: v[:256] for k, v in y.items()}, fs)
    rb = b.fit(x, y, x[:256], {k: v[:256] for k, v in y.items()}, fs, trace_batches=_batches())
    for k, v in a.model.state_dict().items():
        assert torch.equal(v, b.model.state_dict()[k]), "the trace batch set changed training"
    shares = {k: v for k, v in rb.history[-1].items() if k.startswith("trace_gradshare_")}
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares != {k: v for k, v in ra.history[-1].items() if k.startswith("trace_gradshare_")}
    assert {k for k in ra.history[-1] if not k.startswith("trace_")} == {
        k for k in rb.history[-1] if not k.startswith("trace_")}
