"""P23: RunIdentity + check_resume."""

import copy
import dataclasses
import json

import pytest
import torch

from nig_timenet_torch.model.architectures import ModelSettings
from nig_timenet_torch.trainer.identity import (
    ResumeMismatchError,
    RunIdentity,
    SeedMismatchError,
    build_run_identity,
    check_resume,
    optimizer_settings,
    settings_hash,
)
from nig_timenet_torch.trainer.trainer import FitSettings
from tests.trainer.fixture_setup import trainer_config

BASE = {"loss_weights": {"close_reg": 1.0}, "label_smoothing": 0.0, "lr": 5e-5, "epochs": 10}


def ident(**changes) -> RunIdentity:
    base = RunIdentity(schema_version=1, settings_hash=settings_hash(BASE), seed=42, data_fingerprint="d" * 64,
                       code_sha="c" * 40, parent_hashes=("p1",))
    return dataclasses.replace(base, **changes)


def test_resume_with_changed_settings_hash_raises():
    changed = ident(settings_hash=settings_hash({**BASE, "label_smoothing": 0.1}))
    with pytest.raises(ResumeMismatchError, match="settings_hash"):
        check_resume(ident(), changed)
    with pytest.raises(ResumeMismatchError, match="seed: saved=42 current=7"):
        check_resume(ident(), ident(seed=7))


def test_resume_with_only_allowed_change_passes():
    assert settings_hash({**BASE, "epochs": 99}) == settings_hash(BASE)  # epochs never enters the identity
    check_resume(ident(), ident(settings_hash=settings_hash({**BASE, "epochs": 99})))

    @dataclasses.dataclass(frozen=True)
    class WithEpochs(RunIdentity):  # throwaway field, exercises allow_changes
        epochs: int = 0

    a = WithEpochs(**ident().__dict__, epochs=10)
    b = dataclasses.replace(a, epochs=20)
    check_resume(a, b)
    with pytest.raises(ResumeMismatchError, match="epochs"):
        check_resume(a, b, allow_changes=())


def test_json_round_trip_and_frozen():
    i = ident()
    assert RunIdentity.from_json(i.to_json()) == i
    with pytest.raises(dataclasses.FrozenInstanceError):
        i.seed = 1
    with pytest.raises(ValueError, match="unknown"):
        RunIdentity.from_json({**i.to_json(), "bogus": 1})


# ---------------------------------------------------------------------------------------- build_run_identity (T2)

FP = "f" * 64
SHA = "c" * 40


def make_optimizer(lr=1e-3, weight_decay=0.01, extra_group=False):
    a, b = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    groups = [{"params": list(a.parameters())}]
    if extra_group:
        groups.append({"params": list(b.parameters()), "lr": lr * 10})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def build(fit=None, model=None, trainer=None, optimizer=None, fp=FP, sha=SHA):
    fit = fit or FitSettings(epochs=5, seed=7)
    model = model or ModelSettings(overrides={"d_model": 32})
    trainer = trainer if trainer is not None else trainer_config()
    optimizer = optimizer or make_optimizer()
    return build_run_identity(fit, model, trainer, optimizer, fp, code_sha=sha)


def with_config(path, value):
    """trainer_config() with one entry changed; path = keys down the nested dicts."""
    cfg = copy.deepcopy(trainer_config())
    node = cfg
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    return cfg


FIRST_TARGET = next(iter(trainer_config()["targets"]))
SCHEDULE = next(iter(trainer_config()["loss"]["schedules"]))


def test_build_takes_seed_fingerprint_and_code_sha_from_its_inputs():
    i = build()
    assert (i.schema_version, i.seed, i.data_fingerprint, i.code_sha) == (1, 7, FP, SHA)
    assert i.parent_hashes == () and i.fold is None and i.data_cut is None
    assert RunIdentity.from_json(i.to_json()) == i


def test_build_is_deterministic_for_equal_inputs():
    assert build() == build(trainer=copy.deepcopy(trainer_config()), optimizer=make_optimizer())


def test_epochs_are_not_part_of_the_identity():
    assert build(FitSettings(epochs=1, seed=7)) == build(FitSettings(epochs=50, seed=7))


@pytest.mark.parametrize("change", [
    {"fit": FitSettings(epochs=5, seed=7, batch_size=16)},
    {"fit": FitSettings(epochs=5, seed=8)},
    {"model": ModelSettings(overrides={"d_model": 64})},
    {"model": {"architecture": "other"}},
    {"fp": "e" * 64},
    {"sha": "d" * 40},
])
def test_any_other_change_changes_the_identity(change):
    assert build(**change) != build()


@pytest.mark.parametrize("path, value", [
    (("targets", FIRST_TARGET, "loss_weight"), 7.5),
    (("targets", FIRST_TARGET, "task_type"), "binary_classification"),
    (("loss", "use_uncertainty_weighting"), not trainer_config()["loss"].get("use_uncertainty_weighting", False)),
    (("loss", "schedules", SCHEDULE, "start"), 0.123),
    (("constraints",), [{"name": "x"}]),
], ids=["loss_weight", "task_type", "kendall_switch", "schedule_start", "constraints"])
def test_a_trainer_config_change_changes_the_identity(path, value):
    changed = build(trainer=with_config(path, value))
    assert changed != build()
    assert changed.settings_hash != build().settings_hash and changed.seed == build().seed


@pytest.mark.parametrize("optimizer", [
    make_optimizer(lr=2e-3), make_optimizer(weight_decay=0.1), make_optimizer(extra_group=True),
], ids=["lr", "weight_decay", "param_group"])
def test_an_optimizer_change_changes_the_identity(optimizer):
    assert build(optimizer=optimizer).settings_hash != build().settings_hash


def test_one_param_group_hyper_parameter_changes_the_identity():
    opt = make_optimizer(extra_group=True)
    changed = make_optimizer(extra_group=True)
    changed.param_groups[1]["weight_decay"] = 0.5
    assert build(optimizer=changed).settings_hash != build(optimizer=opt).settings_hash


def test_optimizer_type_changes_the_identity():
    sgd = torch.optim.SGD(torch.nn.Linear(2, 2).parameters(), lr=1e-3, weight_decay=0.01)
    assert build(optimizer=sgd).settings_hash != build().settings_hash
    assert optimizer_settings(sgd)["type"] == "torch.optim.sgd.SGD"


def test_optimizer_settings_lists_every_group_without_tensors():
    s = optimizer_settings(make_optimizer(extra_group=True))
    assert [g["n_params"] for g in s["param_groups"]] == [2, 2] and [g["lr"] for g in s["param_groups"]] == [1e-3, 1e-2]
    assert all("params" not in g for g in s["param_groups"]) and {"lr", "weight_decay"} <= set(s["defaults"])
    json.dumps(s)


def test_model_settings_may_be_a_dataclass_or_the_same_dict():
    as_dict = {"architecture": ModelSettings().architecture, "overrides": {"d_model": 32}}
    assert build(model=as_dict) == build()


@pytest.mark.parametrize("bad", [dict(fp=""), dict(fp=None), dict(sha=""), dict(model=3), dict(fit=3), dict(trainer=3),
                                 dict(trainer={"x": object()}), dict(trainer={"x": float("nan"), "f": len})])
def test_build_refuses_missing_or_unhashable_pieces(bad):
    with pytest.raises((ValueError, TypeError)):
        build(**bad)


def test_optimizer_must_be_an_optimizer_instance_and_is_required():
    with pytest.raises(TypeError, match="torch.optim.Optimizer"):
        build(optimizer={"type": "AdamW", "lr": 1e-3})
    with pytest.raises(TypeError):  # optimizer (and trainer_config) are required, not defaulted
        build_run_identity(FitSettings(epochs=1), ModelSettings(), trainer_config(), FP, code_sha=SHA)
    with pytest.raises(TypeError):
        optimizer_settings(None)


def test_seed_mismatch_error_is_a_value_error():
    assert issubclass(SeedMismatchError, ValueError)
