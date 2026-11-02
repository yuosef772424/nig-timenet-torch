"""P23: RunIdentity + check_resume."""

import dataclasses

import pytest

from nig_timenet_torch.trainer.identity import ResumeMismatchError, RunIdentity, check_resume, settings_hash

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
