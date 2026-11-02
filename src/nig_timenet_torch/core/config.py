"""Strict settings primitive (BUILD_PLAN.md P04): every ``*Settings`` / ``*Config`` class of this repo is a frozen
dataclass built through :func:`from_dict`. Unknown keys raise (with a did-you-mean), defaults live only on the dataclass
fields (no second default source such as ``d.get(key, default)``).

Shape of the rule from crypto-signal-prediction docs/research/hardening/audit_2026-10-31.md:156-165 (C2);
this is a fresh, smaller primitive, not a port (there is no legacy dict config to retrofit here).
"""

import dataclasses
import difflib
import types
import typing
from collections.abc import Mapping
from typing import Any

__all__ = ["from_dict"]


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
