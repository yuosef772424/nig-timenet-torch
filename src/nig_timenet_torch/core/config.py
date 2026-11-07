"""PURPOSE: Strict settings primitive: from_dict builds frozen dataclasses (unknown keys raise), to_dict is its inverse
    and updated copies a settings object with dotted-key changes.
TAGS: strict config, settings dataclass, from_dict, to_dict, updated, did you mean
PITFALLS: to_dict turns tuples into lists (from_dict accepts them back); updated re-runs __post_init__ through
    dataclasses.replace, and a mapping switches an optional (None) section on built from its defaults.

Strict settings primitive (BUILD_PLAN.md P04): every ``*Settings`` / ``*Config`` class of this repo is a frozen
dataclass built through :func:`from_dict`. Unknown keys raise (with a did-you-mean), defaults live only on the dataclass
fields (no second default source such as ``d.get(key, default)``). One exception: the trainer config is a nested
dict validated by ``trainer/targets.py`` (``validate_trainer_config``, D-012), not a dataclass.

Shape of the rule from crypto-signal-prediction docs/research/hardening/audit_2026-10-31.md:156-165 (C2);
this is a fresh, smaller primitive, not a port (there is no legacy dict config to retrofit here).
"""

import dataclasses
import difflib
import types
import typing
from collections.abc import Mapping
from typing import Any

__all__ = ["from_dict", "to_dict", "updated"]


def _dataclass_type(hint: Any) -> type | None:
    """The dataclass type a field annotation refers to: ``DC``, ``DC | None`` or ``Optional[DC]``; else None."""
    if isinstance(hint, type) and dataclasses.is_dataclass(hint):
        return hint
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        dcs = [a for a in typing.get_args(hint) if isinstance(a, type) and dataclasses.is_dataclass(a)]
        if len(dcs) == 1:
            return dcs[0]
    return None


def from_dict(cls: type, d: Mapping[str, Any], *, strict: bool = True):
    """Build a frozen dataclass `cls` from `d`. strict=True (default): a key in `d` not in `cls`'s fields raises
    ValueError naming the unknown key + a did-you-mean (reuse core.registry's difflib pattern). Nested dataclass
    fields are built recursively. Missing keys use the dataclass field default (no second default source)."""
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        raise TypeError(f"from_dict: {cls!r} is not a dataclass type")
    if not cls.__dataclass_params__.frozen:
        raise TypeError(f"from_dict: {cls.__name__} must be a frozen dataclass (@dataclass(frozen=True))")
    if not isinstance(d, Mapping):
        raise TypeError(f"from_dict({cls.__name__}): expected a mapping, got {type(d).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    unknown = [k for k in d if k not in fields]
    if unknown and strict:
        known = sorted(fields)
        parts = []
        for k in unknown:
            close = difflib.get_close_matches(str(k), known, n=1)
            parts.append(f"{k!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise ValueError(f"{cls.__name__}: unknown key(s) {', '.join(parts)}. Known keys: {known}")
    missing = [n for n, f in fields.items() if n not in d and f.default is dataclasses.MISSING
               and f.default_factory is dataclasses.MISSING]
    if missing:
        raise ValueError(f"{cls.__name__}: missing required key(s) {missing} (no field default)")
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for name in fields:
        if name not in d:
            continue
        value = d[name]
        sub = _dataclass_type(hints.get(name))
        if sub is not None and isinstance(value, Mapping):
            value = from_dict(sub, value, strict=strict)
        kwargs[name] = value
    return cls(**kwargs)


def to_dict(obj: Any) -> dict[str, Any]:
    """The plain nested dict of a settings dataclass (tuples become lists), the inverse of :func:`from_dict`."""
    def conv(v: Any) -> Any:
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            return {f.name: conv(getattr(v, f.name)) for f in dataclasses.fields(v) if f.init}
        if isinstance(v, Mapping):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        return v
    return conv(obj)


def updated(obj: Any, patch: Mapping[str, Any] | None = None, /, **changes: Any):
    """A copy of the frozen dataclass `obj` with `patch` applied (the notebooks' ``settings.updated({...})``).

    Keys may be dotted (``"train.epochs"``) and a mapping value for a nested-dataclass field is merged into it
    instead of replacing it. Unknown keys raise ValueError with a did-you-mean; the copy is rebuilt through
    ``dataclasses.replace`` so ``__post_init__`` validation runs again."""
    merged = {**(patch or {}), **changes}
    names = [f.name for f in dataclasses.fields(obj) if f.init]
    grouped: dict[str, dict[str, Any]] = {}
    direct: dict[str, Any] = {}
    for key, value in merged.items():
        head, _, rest = str(key).partition(".")
        if head not in names:
            close = difflib.get_close_matches(head, names, n=1)
            raise ValueError(f"{type(obj).__name__}: unknown key {head!r}"
                             + (f" (did you mean {close[0]!r}?)" if close else "") + f". Known keys: {sorted(names)}")
        if rest:
            grouped.setdefault(head, {})[rest] = value
        else:
            direct[head] = value
    out = {}
    hints = typing.get_type_hints(type(obj))
    for head in {*direct, *grouped}:
        current = getattr(obj, head)
        value = direct.get(head, current)
        sub = _dataclass_type(hints.get(head))
        if current is None and sub is not None and (head in grouped or isinstance(value, Mapping)):
            # an optional section that is off (None): a mapping switches it on, built strictly from its defaults
            value = from_dict(sub, {**(value if isinstance(value, Mapping) else {}), **grouped.get(head, {})})
        elif head in grouped:
            if not dataclasses.is_dataclass(value):
                raise ValueError(f"{type(obj).__name__}.{head} is not a settings section; dotted keys need one")
            value = updated(value, grouped[head])
        elif dataclasses.is_dataclass(current) and isinstance(value, Mapping):
            value = updated(current, value)
        out[head] = value
    return dataclasses.replace(obj, **out)
