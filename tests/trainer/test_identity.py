"""P23: RunIdentity + check_resume."""

import dataclasses

import pytest

from nig_timenet_torch.model.architectures import ModelSettings
from nig_timenet_torch.trainer.identity import (
    ResumeMismatchError,
    RunIdentity,
    SeedMismatchError,
    build_run_identity,
    check_resume,
    settings_hash,
)
from nig_timenet_torch.trainer.trainer import FitSettings

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


def build(fit=None, model=None, fp=FP, sha=SHA):
    fit = fit or FitSettings(epochs=5, seed=7)
    model = model or ModelSettings(overrides={"d_model": 32})
    return build_run_identity(fit, model, fp, code_sha=sha)


def test_build_takes_seed_fingerprint_and_code_sha_from_its_inputs():
    i = build()
    assert (i.schema_version, i.seed, i.data_fingerprint, i.code_sha) == (1, 7, FP, SHA)
    assert i.parent_hashes == () and i.fold is None and i.data_cut is None
    assert RunIdentity.from_json(i.to_json()) == i


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


def test_model_settings_may_be_a_dataclass_or_the_same_dict():
    as_dict = {"architecture": ModelSettings().architecture, "overrides": {"d_model": 32}}
    assert build(model=as_dict) == build()


@pytest.mark.parametrize("bad", [dict(fp=""), dict(fp=None), dict(sha=""), dict(model=3), dict(fit=3)])
def test_build_refuses_missing_pieces(bad):
    with pytest.raises((ValueError, TypeError)):
        build(**bad)


def test_seed_mismatch_error_is_a_value_error():
    assert issubclass(SeedMismatchError, ValueError)
