"""``FitSettings.steps_per_execution`` (Keras steps_per_execution): the logs of N steps are read back together; the
updates are the same ones in the same order, so the weights are EXACTLY those of N = 1, the history agrees to
float rounding, the run identity is unchanged, and a resume across groups equals an uninterrupted run."""

import dataclasses

import pytest
import torch

from nig_timenet_torch.core.seeds import seed_everything
from nig_timenet_torch.trainer.identity import configuration_hash
from nig_timenet_torch.trainer.schedules import LrSettings
from nig_timenet_torch.trainer.trainer import FitSettings
from nig_timenet_torch.workflow.settings import TrainSettings
from tests.trainer.helpers import fit, identity_for, make_trainer, state_equal

LR = LrSettings(lr_initial=3e-3, lr_min=1e-5, warmup_epochs=1, kind="cosine", total_epochs=3)
BASE = FitSettings(epochs=2, batch_size=32, lr=LR)


@pytest.mark.parametrize("n", [2, 5, 32, 1000])  # 1000 > batches per epoch: one group = the whole epoch
@pytest.mark.parametrize("extra", [{}, {"grad_clip_norm": 0.5, "ema_decay": 0.9}])
def test_weights_are_bit_identical_to_one_step_per_execution(data, n, extra):
    a, b = make_trainer(), make_trainer()
    ra = fit(a, data, dataclasses.replace(BASE, **extra))
    rb = fit(b, data, dataclasses.replace(BASE, steps_per_execution=n, **extra))
    assert state_equal(a.model, b.model)
    assert ra.best_epoch == rb.best_epoch and len(ra.history) == len(rb.history)
    for row_a, row_b in zip(ra.history, rb.history, strict=True):
        for k, v in row_a.items():
            if isinstance(v, float) and not k.startswith("trace_") and "seconds" not in k:
                assert row_b[k] == pytest.approx(v, rel=1e-6, abs=1e-9), k


def test_identity_ignores_it_and_bad_values_raise(data):
    t = make_trainer()
    h = lambda s: configuration_hash(s, {"m": 1}, t.config, t.optimizer)  # noqa: E731
    assert h(BASE) == h(dataclasses.replace(BASE, steps_per_execution=32))
    assert h(BASE) != h(dataclasses.replace(BASE, batch_size=64))     # the hash still sees real changes
    for bad in (0, -1, 2.0, True):
        with pytest.raises(ValueError, match="steps_per_execution"):
            FitSettings(epochs=1, steps_per_execution=bad)
    with pytest.raises(ValueError, match="steps_per_execution"):
        TrainSettings(steps_per_execution=0)


def test_resume_across_groups_equals_an_uninterrupted_run(data, tmp_path):
    s = FitSettings(epochs=3, batch_size=32, lr=LR, steps_per_execution=5)
    ident = identity_for(s)
    full = make_trainer()
    fr = fit(full, data, s, run_dir=tmp_path / "full", identity=ident)
    fit(make_trainer(), data, dataclasses.replace(s, epochs=1), run_dir=tmp_path / "split", identity=ident)
    second = make_trainer(init_seed=123)
    resumed = fit(second, data, s, run_dir=tmp_path / "split", identity=ident)
    assert resumed.history == fr.history and state_equal(second.model, full.model)


def test_train_step_logs_as_tensors_do_not_change_the_step(data):
    a, b = make_trainer(), make_trainer()
    x, y = data[0][:32], {k: v[:32] for k, v in data[1].items()}
    seed_everything(1)
    la = a.train_step(x, y)
    seed_everything(1)                                   # the same dropout masks for both steps
    lb = b.train_step(x, y, as_tensors=True)
    assert all(isinstance(v, torch.Tensor) and v.dim() == 0 for v in lb.values())
    assert {k: float(v) for k, v in lb.items()} == la and state_equal(a.model, b.model)
