"""P04: core.config.from_dict — strict, frozen, recursive, defaults only on the dataclass."""

from dataclasses import FrozenInstanceError, dataclass, field

import pytest

from nig_timenet_torch.core.config import from_dict


@dataclass(frozen=True)
class Optim:
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass(frozen=True)
class Run:
    seed: int
    epochs: int = 10
    optim: Optim = field(default_factory=Optim)
    extra: Optim | None = None


def test_unknown_key_raises_with_did_you_mean():
    with pytest.raises(ValueError, match=r"unknown key\(s\) 'epoch' \(did you mean 'epochs'\?\)"):
        from_dict(Run, {"seed": 0, "epoch": 5})
    with pytest.raises(ValueError, match=r"Optim: unknown key\(s\) 'lrr' \(did you mean 'lr'\?\)"):
        from_dict(Run, {"seed": 0, "optim": {"lrr": 0.1}})
    with pytest.raises(ValueError, match="'zzz'"):  # no close match: still named
        from_dict(Run, {"seed": 0, "zzz": 1})


def test_nested_dataclass_builds_recursively():
    r = from_dict(Run, {"seed": 1, "optim": {"lr": 0.5}, "extra": {"weight_decay": 0.1}})
    assert r.optim == Optim(lr=0.5) and r.extra == Optim(weight_decay=0.1)
    assert from_dict(Run, {"seed": 1, "optim": Optim(lr=2.0)}).optim.lr == 2.0  # instance passes through
    with pytest.raises(FrozenInstanceError):
        r.seed = 2


def test_missing_key_uses_dataclass_default():
    r = from_dict(Run, {"seed": 3})
    assert r == Run(seed=3) and r.epochs == 10 and r.optim == Optim() and r.extra is None
    with pytest.raises(ValueError, match=r"missing required key\(s\) \['seed'\]"):
        from_dict(Run, {})


def test_strict_false_ignores_unknown_keys():
    r = from_dict(Run, {"seed": 0, "nope": 1, "optim": {"lr": 0.2, "also_nope": 2}}, strict=False)
    assert r == Run(seed=0, optim=Optim(lr=0.2))


def test_rejects_non_frozen_or_non_dataclass():
    @dataclass
    class Mutable:
        a: int = 0

    with pytest.raises(TypeError, match="frozen"):
        from_dict(Mutable, {})
    with pytest.raises(TypeError, match="not a dataclass"):
        from_dict(dict, {})
