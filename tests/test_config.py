"""P04: core.config.from_dict — strict, frozen, recursive, defaults only on the dataclass."""

from dataclasses import FrozenInstanceError, dataclass, field

import pytest

from nig_timenet_torch.core.config import from_dict, to_dict, updated


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


@dataclass(frozen=True)
class Checked:
    n: int = 1
    tags: tuple[str, ...] = ("a", "b")
    run: Run | None = None

    def __post_init__(self):
        if self.n < 0:
            raise ValueError("n must be >= 0")


def test_to_dict_is_the_plain_nested_inverse_of_from_dict():
    r = from_dict(Run, {"seed": 3, "optim": {"lr": 0.5}, "extra": {"weight_decay": 0.1}})
    d = to_dict(r)
    assert d == {"seed": 3, "epochs": 10, "optim": {"lr": 0.5, "weight_decay": 0.0},
                 "extra": {"lr": 1e-3, "weight_decay": 0.1}}
    assert from_dict(Run, d) == r
    c = to_dict(Checked(tags=("x", "y")))
    assert c == {"n": 1, "tags": ["x", "y"], "run": None} and isinstance(c["tags"], list)
    assert to_dict({"k": (1, {"z": 2})}) == {"k": [1, {"z": 2}]}


def test_updated_returns_a_copy_and_accepts_mapping_keywords_and_dotted_keys():
    r = Run(seed=1)
    assert updated(r, {"epochs": 3}).epochs == 3 and r.epochs == 10          # original untouched
    assert updated(r, epochs=4).epochs == 4
    assert updated(r, {"epochs": 3}, seed=9) == Run(seed=9, epochs=3)
    assert updated(r, {"optim.lr": 0.5}).optim == Optim(lr=0.5)               # dotted key
    merged = updated(r, {"optim": {"weight_decay": 0.2}}).optim              # merged, not replaced
    assert merged == Optim(lr=1e-3, weight_decay=0.2)
    assert updated(r, {"optim": Optim(lr=7.0)}).optim.lr == 7.0               # an instance replaces
    assert updated(r) == r


def test_updated_unknown_keys_raise_with_did_you_mean():
    r = Run(seed=1)
    with pytest.raises(ValueError, match=r"unknown key 'epoch' \(did you mean 'epochs'\?\)"):
        updated(r, {"epoch": 3})
    with pytest.raises(ValueError, match="unknown key 'zzz'"):
        updated(r, zzz=1)
    with pytest.raises(ValueError, match=r"Optim: unknown key 'lrr' \(did you mean 'lr'\?\)"):
        updated(r, {"optim.lrr": 1})
    with pytest.raises(ValueError, match="not a settings section"):
        updated(r, {"epochs.x": 1})


def test_updated_reruns_post_init_and_switches_an_optional_section_on():
    with pytest.raises(ValueError, match="n must be >= 0"):
        updated(Checked(), {"n": -1})
    on = updated(Checked(), {"run": {"seed": 5}})
    assert on.run == Run(seed=5)
    assert updated(Checked(), {"run.seed": 6}).run == Run(seed=6)
    assert updated(on, {"run.epochs": 2}).run == Run(seed=5, epochs=2)
    with pytest.raises(ValueError, match="unknown key"):
        updated(Checked(), {"run": {"sed": 5}})
