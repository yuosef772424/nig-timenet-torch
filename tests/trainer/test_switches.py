"""D-021 training switches (trainer/switches.py, FitSettings / TrainSettings): every default reproduces the trainer
before D-021 (same weights, same identity, same trainer config); each switch on changes what it should; resume with
each switch on equals an uninterrupted run."""

import dataclasses

import pytest
import torch

from nig_timenet_torch.core.registry import UnknownNameError
from nig_timenet_torch.trainer import switches
from nig_timenet_torch.trainer.identity import build_run_identity
from nig_timenet_torch.trainer.schedules import LrSettings
from nig_timenet_torch.trainer.trainer import FitSettings
from nig_timenet_torch.workflow import flat_run
from nig_timenet_torch.workflow.settings import TrainSettings
from tests.trainer.helpers import fit, identity_for, make_trainer, state_equal

LR = LrSettings(lr_initial=3e-3, lr_min=1e-5, warmup_epochs=1, kind="cosine", total_epochs=3)
BASE = FitSettings(epochs=2, batch_size=32, lr=LR)


def test_switches_at_their_defaults_give_the_same_weights_and_identity(data):
    a, b = make_trainer(), make_trainer()
    fit(a, data, BASE)
    fit(b, data, dataclasses.replace(BASE, grad_clip_norm=None, ema_decay=None, init="torch"))
    assert state_equal(a.model, b.model)
    assert identity_for(BASE) == identity_for(dataclasses.replace(BASE, init="torch"))


@pytest.mark.parametrize("change", [{"grad_clip_norm": 0.05}, {"ema_decay": 0.9}, {"init": "keras"}])
def test_each_switch_changes_the_weights_and_the_identity(data, change):
    a, b = make_trainer(), make_trainer()
    fit(a, data, BASE)
    fit(b, data, dataclasses.replace(BASE, **change))
    assert not state_equal(a.model, b.model)
    assert identity_for(BASE) != identity_for(dataclasses.replace(BASE, **change))


@pytest.mark.parametrize("change", [{"grad_clip_norm": 0.05}, {"ema_decay": 0.9}, {"init": "keras"}])
def test_resume_with_a_switch_on_equals_an_uninterrupted_run(data, tmp_path, change):
    settings = dataclasses.replace(FitSettings(epochs=3, batch_size=32, lr=LR), **change)
    ident = identity_for(settings)
    full = make_trainer()
    full_result = fit(full, data, settings, run_dir=tmp_path / "full", identity=ident)
    fit(make_trainer(), data, dataclasses.replace(settings, epochs=1), run_dir=tmp_path / "split", identity=ident)
    second = make_trainer(init_seed=123)
    resumed = fit(second, data, settings, run_dir=tmp_path / "split", identity=ident)
    assert resumed.history == full_result.history and state_equal(second.model, full.model)


def test_ema_is_what_the_model_holds_and_validates_with(data):
    t = make_trainer()
    s = dataclasses.replace(BASE, ema_decay=0.5, early_stop=dataclasses.replace(BASE.early_stop,
                                                                                restore_best_weights=False))
    result = fit(t, data, s)
    x_val, y_val = data[2], data[3]
    assert t.evaluate(x_val, y_val, 32)["raw_loss"] == pytest.approx(
        result.history[-1]["val_raw_loss"], rel=1e-6)


def test_gradient_clipping_bounds_the_update(data):
    t = make_trainer()
    x, y = data[0][:32], {k: v[:32] for k, v in data[1].items()}
    t.train_step(x, y, grad_clip_norm=1e-3)
    norm = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in t.parameters() if p.grad is not None))
    assert float(norm) <= 1e-3 * 1.0001


def test_keras_init_is_glorot_with_zero_biases():
    t = make_trainer()
    switches.apply_init("keras", t.model)
    lin = [m for m in t.model.modules() if isinstance(m, torch.nn.Linear)]
    assert lin and all(m.bias is None or float(m.bias.detach().abs().max()) == 0.0 for m in lin)
    w = lin[0].weight
    bound = (6.0 / (w.shape[0] + w.shape[1])) ** 0.5
    assert float(w.abs().max()) <= bound


def test_weight_decay_groups_all_params_is_one_group_and_exclude_splits_norms_and_biases():
    t = make_trainer()
    params = list(t.parameters())
    (only,) = switches.param_groups("all_params", t.model, params, 1e-4)
    assert len(only["params"]) == len(params) and only["weight_decay"] == 1e-4
    decay, keep = switches.param_groups("exclude_norm_bias", t.model, params, 1e-4)
    names = {id(p): n for n, p in t.model.named_parameters()}
    assert all(p.dim() >= 2 for p in decay["params"]) and keep["weight_decay"] == 0.0
    assert any(names.get(id(p), "").endswith("bias") for p in keep["params"])
    assert len(decay["params"]) + len(keep["params"]) == len(params)


def test_unknown_switch_names_raise():
    with pytest.raises(UnknownNameError):
        FitSettings(epochs=1, init="xavier")
    with pytest.raises(UnknownNameError):
        switches.param_groups("nope", torch.nn.Linear(2, 2), [], 0.0)
    with pytest.raises(ValueError):
        FitSettings(epochs=1, ema_decay=1.0)


def test_trainer_config_at_default_switches_is_byte_equal_and_each_loss_switch_lands():
    targets = ("high", "low", "close")
    base = flat_run.trainer_config(targets, 0.1)
    assert flat_run.trainer_config(targets, 0.1, TrainSettings()) == base
    cfg = flat_run.trainer_config(targets, 0.1, TrainSettings(class_loss_weight=3.0, uncertainty_weighting=False,
                                                             lambda_reg_end=0.0, lambda_calib_end=0.2))
    assert cfg["loss"]["use_uncertainty_weighting"] is False
    sched = cfg["loss"]["schedules"]
    assert sched["lambda_reg"]["end"] == 0.0 and sched["lambda_calib"]["end"] == 0.2
    weights = {n: c.get("loss_weight") for n, c in cfg["targets"].items()}
    assert {w for n, w in weights.items() if c_kind(cfg, n) == "classification"} == {3.0}
    assert {w for n, w in weights.items() if c_kind(cfg, n) != "classification"} == {None}


def c_kind(cfg, name):
    return cfg["targets"][name]["task_type"]


def test_identity_ignores_default_switches_only():
    kw = dict(model_settings={"a": 1}, trainer_config={"targets": {}},
              optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1), data_fingerprint="d",
              code_sha="c")
    a = build_run_identity(FitSettings(epochs=1), **kw)
    assert a == build_run_identity(FitSettings(epochs=1, init="torch"), **kw)
    assert a != build_run_identity(FitSettings(epochs=1, grad_clip_norm=1.0), **kw)


def test_ema_warm_up_follows_the_trained_weights_early():
    """D-021 finding: without the warm-up, decay 0.999 over a few hundred steps keeps the EMA at the init weights."""
    m = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(m.weight)
    ema = switches.EmaWeights(m, 0.999)
    with torch.no_grad():
        m.weight.fill_(1.0)
    ema.update(m)
    assert float(ema.weights()["weight"]) == pytest.approx(0.9)          # d_0 = 1/10
    for _ in range(300):
        ema.update(m)
    assert float(ema.weights()["weight"]) > 0.999 and ema.current_decay() < 0.999
