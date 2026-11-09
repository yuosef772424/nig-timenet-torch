"""FX2 loss balancers (trainer/loss_balance.py) and the Kendall start value: the defaults reproduce the trainer before
D-023; ``equal_grad_norm`` equalises the heads' gradient norms on h, refuses Kendall and resumes exactly."""

import copy
import dataclasses

import pytest
import torch

from nig_timenet_torch.core.registry import UnknownNameError
from nig_timenet_torch.trainer.loss_balance import LOSS_BALANCERS, make_balancer
from nig_timenet_torch.trainer.schedules import LrSettings
from nig_timenet_torch.trainer.trainer import FitSettings
from tests.trainer.fixture_setup import trainer_config
from tests.trainer.helpers import fit, identity_for, make_trainer, state_equal

LR = LrSettings(lr_initial=3e-3, lr_min=1e-5, warmup_epochs=1, kind="cosine", total_epochs=3)


def _config(**loss):
    cfg = copy.deepcopy(trainer_config())
    cfg["loss"].update(loss)
    return cfg


def test_registry_names_and_unknown_name_raises():
    assert set(LOSS_BALANCERS.names()) == {"none", "equal_grad_norm"}
    with pytest.raises(UnknownNameError):
        make_balancer("grad_norm", ["a"], [])


def test_explicit_defaults_train_the_same_weights(data):
    a = make_trainer()
    b = make_trainer(config=_config(balance="none", kendall_init_logvar=0.0))
    settings = FitSettings(epochs=2, batch_size=32, lr=LR)
    fit(a, data, settings)
    fit(b, data, settings)
    assert state_equal(a.model, b.model)
    assert set(a.balancer.multipliers().values()) == {1.0}


def test_kendall_init_logvar_sets_the_start_value():
    t = make_trainer(config=_config(kendall_init_logvar=-2.0))
    assert {float(t.uncertainty_layer.log_var(n).detach()) for n in t.kendall_task_names} == {-2.0}


def test_equal_grad_norm_refuses_kendall():
    with pytest.raises(ValueError, match="replaces Kendall"):
        make_trainer(config=_config(balance="equal_grad_norm"))


def _balanced():
    return make_trainer(config=_config(balance="equal_grad_norm", use_uncertainty_weighting=False))


def test_equal_grad_norm_equalises_the_ema_norms(data):
    t = _balanced()
    x, y = data[0][:32], {k: v[:32] for k, v in data[1].items()}
    t.train_step(x, y)
    w = t.balancer.multipliers()
    products = [w[n] * t.balancer.ema[n] for n in t.target_names]
    assert max(products) == pytest.approx(min(products), rel=1e-9)
    assert sum(products) == pytest.approx(sum(t.balancer.ema.values()), rel=1e-9)
    assert len(set(w.values())) > 1          # the heads' norms differ, so the multipliers do


def test_equal_grad_norm_changes_training_and_resumes_exactly(data, tmp_path):
    settings = FitSettings(epochs=3, batch_size=32, lr=LR)
    ident = identity_for(settings)
    plain = make_trainer(config=_config(use_uncertainty_weighting=False))
    fit(plain, data, settings)
    full = _balanced()
    full_result = fit(full, data, settings, run_dir=tmp_path / "full", identity=ident)
    assert not state_equal(plain.model, full.model)
    fit(_balanced(), data, dataclasses.replace(settings, epochs=1), run_dir=tmp_path / "split", identity=ident)
    second = _balanced()
    resumed = fit(second, data, settings, run_dir=tmp_path / "split", identity=ident)
    assert resumed.history == full_result.history and state_equal(second.model, full.model)
    assert second.balancer.state_dict() == full.balancer.state_dict()


def test_default_checkpoint_has_no_balance_key(data, tmp_path):
    from nig_timenet_torch.trainer import checkpoint
    settings = FitSettings(epochs=1, batch_size=32, lr=LR)
    fit(make_trainer(), data, settings, run_dir=tmp_path, identity=identity_for(settings))
    assert "balance" not in checkpoint.load_checkpoint(tmp_path)


def test_grad_report_sees_the_balanced_objective(data):
    """gate A5 reads the objective as optimised: the per-target losses carry the multipliers."""
    t = _balanced()
    x, y = data[0][:32], {k: v[:32] for k, v in data[1].items()}
    t.train_step(x, y)
    t.model.eval()
    _, _, per_target, _, _ = t.compute_losses(x, y)
    w = t.balancer.multipliers()
    plain = make_trainer(config=_config(use_uncertainty_weighting=False))
    plain.model.load_state_dict(t.model.state_dict())
    plain.model.eval()
    _, _, ref, _, _ = plain.compute_losses(x, y)
    for n in t.target_names:
        assert float(per_target[n]) == pytest.approx(float(ref[n]) * w[n], rel=1e-5)
    assert torch.isfinite(torch.stack(list(per_target.values()))).all()
