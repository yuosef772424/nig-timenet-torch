"""The generic Registry primitive: named components with a declared contract (tier 0, stdlib only)

SOURCE:   Ported close to verbatim from yuosef772424/crypto-signal-prediction core/registry.py at
          aeafc564f25c78f252936affbd374f68aa49b8d9 (BUILD_PLAN.md P02). Only changes: this provenance line,
          `from __future__ import annotations` dropped (Python >= 3.11), the card re-wrapped and its TF-repo user
          list replaced by a pointer to the registries table of docs/MAP.md (H1, HF-02).
PURPOSE:  The ONE generic registry: `Registry(kind, contract)` registers objects by name with a declared contract
          (checked AT REGISTRATION); raises on a duplicate name and on an unknown name (known names + did-you-mean).
TAGS:     registry, Registry, Contract, register, unknown name, duplicate name, did you mean, one component one
          interface, plugin, tier 0
PITFALLS: The 7 registries of this repo (kind, variable, module, names) are listed in the Registries table of
          docs/MAP.md. Dict-style READ API (`reg[name]`, `reg.get(name, default)`, `in`, `len`, iteration = names,
          `keys/values/items`) returns the entry's VALUE = `view(entry)` (default: the registered object);
          `get(name)` without a default raises on an unknown name; `entries()` is the Entry list; there is no item
          assignment or pop (register / unregister are the only writers). NO overwrite flag: registering a name
          twice raises, so a module that registers at import time must not be re-executed in one process (use
          `unregister` in tests or a fresh Registry). The validator runs once, at `register`; it must be cheap and
          return a list of problem strings (or raise); `check_all()` re-runs it over every entry.
          `UnknownNameError` subclasses KeyError and ValueError; its str() is the plain message. Metadata is stored
          by reference, not copied.

Usage:
    TASKS = Registry("task", Contract("task", required_meta=("stat_keys",), validator=check_task))
    TASKS.register("nig", nig_loss, version=1, stat_keys=("mae", "nll"))
    TASKS.get("nig")             # the object; TASKS.get("nigg") raises UnknownNameError (known names + "did you mean 'nig'")
"""
import difflib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Tuple

__all__ = ["Registry", "Contract", "Entry", "RegistryError", "DuplicateNameError", "UnknownNameError", "ContractError"]


class RegistryError(Exception):
    """Base class of every error raised by this module."""


class DuplicateNameError(RegistryError, ValueError):
    """A name was registered twice."""


class UnknownNameError(RegistryError, KeyError, ValueError):
    """A lookup of a name that is not registered; the message lists the known names."""

    def __str__(self) -> str:  # KeyError would show the repr (quoted) of its message
        return str(self.args[0]) if self.args else ""


class ContractError(RegistryError, ValueError):
    """The object or its metadata violates the registry's declared contract."""


_MISSING = object()

#: validator(name, obj, meta) -> None | iterable of problem strings (empty = fine); may also raise.
Validator = Callable[[str, Any, Mapping[str, Any]], Optional[Iterable[str]]]


@dataclass(frozen=True)
class Contract:
    """What every entry of one registry must declare.

    kind:          what is registered ("task", "head", "normalizer", ...); used in error messages.
    required_meta: metadata keys every registration must pass (besides name and version).
    optional_meta: extra keys allowed; None = any extra key is accepted, a tuple = only these (unknown keys raise).
    validator:     optional check run at registration (fail early, not hours into a run).
    """
    kind: str
    required_meta: Tuple[str, ...] = ()
    optional_meta: Optional[Tuple[str, ...]] = None
    validator: Optional[Validator] = None

    def __post_init__(self):
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ContractError("Contract.kind must be a non-empty string")
        object.__setattr__(self, "required_meta", tuple(self.required_meta))
        if self.optional_meta is not None:
            object.__setattr__(self, "optional_meta", tuple(self.optional_meta))
        reserved = {"name", "version", "obj"} & set(self.required_meta + (self.optional_meta or ()))
        if reserved:
            raise ContractError(f"Contract({self.kind!r}): {sorted(reserved)} are reserved (name/version/obj are not metadata)")

    def check(self, name: str, obj: Any, version: Any, meta: Mapping[str, Any]) -> None:
        """Raise ContractError listing EVERY problem found (all at once, not one per attempt)."""
        problems: list[str] = []
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            problems.append(f"name must be a non-empty string without surrounding spaces, got {name!r}")
        ok_int = isinstance(version, int) and not isinstance(version, bool) and version >= 1
        ok_str = isinstance(version, str) and bool(version.strip())
        if not (ok_int or ok_str):
            problems.append(f"version must be a positive int or a non-empty string, got {version!r}")
        missing = [k for k in self.required_meta if k not in meta]
        if missing:
            problems.append(f"missing required metadata {missing} (required: {list(self.required_meta)})")
        if self.optional_meta is not None:
            unknown = sorted(set(meta) - set(self.required_meta) - set(self.optional_meta))
            if unknown:
                allowed = list(self.required_meta + self.optional_meta)
                problems.append(f"unknown metadata {unknown} (allowed: {allowed})")
        if not problems and self.validator is not None:
            try:
                found = self.validator(name, obj, meta)
            except Exception as e:  # a validator that raises is a violation, with the cause chained
                raise ContractError(f"{self.kind} {name!r}: contract validator raised {type(e).__name__}: {e}") from e
            problems.extend(str(p) for p in (found or ()))
        if problems:
            raise ContractError(f"{self.kind} {name!r} violates its contract: " + "; ".join(problems))


@dataclass(frozen=True)
class Entry:
    """One registered component: the object plus its declared identity and metadata."""
    name: str
    version: Any
    obj: Any = field(repr=False)
    meta: Mapping[str, Any] = field(default_factory=dict)


class Registry:
    """Name -> object registry with a declared contract. Insertion order is kept (`names()`)."""

    def __init__(self, kind: str, contract: Optional[Contract] = None, *, view: Optional[Callable[["Entry"], Any]] = None):
        """`view(entry) -> value` is what dict-style reads (`reg[name]`, `get`, `values`, `items`) return; default = the object."""
        if contract is None:
            contract = Contract(kind)
        elif contract.kind != kind:
            raise ContractError(f"Registry kind {kind!r} != its contract kind {contract.kind!r}")
        self.kind = kind
        self.contract = contract
        self._view = view if view is not None else (lambda e: e.obj)
        self._entries: dict[str, Entry] = {}

    # ------------------------------------------------------------------ registration
    def register(self, name: str, obj: Any, *, version: Any, **meta: Any) -> Any:
        """Register `obj` under `name`; returns `obj` (so it can be used inline). Raises on a duplicate name or on any
        contract violation, BEFORE anything is stored (a failed registration leaves the registry unchanged)."""
        self.contract.check(name, obj, version, meta)
        if name in self._entries:
            old = self._entries[name]
            raise DuplicateNameError(f"{self.kind} {name!r} is already registered (version {old.version!r}); "
                                     f"a name is registered once: pick a new name (or unregister it in a test)")
        self._entries[name] = Entry(name, version, obj, dict(meta))
        return obj

    def decorator(self, name: str, *, version: Any, **meta: Any) -> Callable[[Any], Any]:
        """`@REG.decorator("name", version=1, key=...)` registers the decorated function/class and returns it unchanged."""
        def deco(obj: Any) -> Any:
            return self.register(name, obj, version=version, **meta)
        return deco

    def unregister(self, name: str) -> Entry:
        """Remove and return an entry (tests / explicit reload only). Unknown name raises like `get`."""
        self._require(name)
        return self._entries.pop(name)

    # ------------------------------------------------------------------ lookup
    def _require(self, name: Any) -> None:
        if name in self._entries:
            return
        known = sorted(self._entries)
        close = difflib.get_close_matches(str(name), known, n=1) if known else []
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise UnknownNameError(f"unknown {self.kind} {name!r}. Known {self.kind}s: {known}.{hint}")

    def entry(self, name: str) -> Entry:
        self._require(name)
        return self._entries[name]

    def get(self, name: str, default: Any = _MISSING) -> Any:
        """The entry's value (the registered object unless a `view` was given); an unknown name raises UnknownNameError
        (known names + did-you-mean) unless a `default` is passed (dict-compatible form)."""
        if default is not _MISSING and name not in self._entries:
            return default
        return self._view(self.entry(name))

    def __getitem__(self, name: str) -> Any:
        return self._view(self.entry(name))

    def names(self) -> Tuple[str, ...]:
        """Registered names in registration order."""
        return tuple(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def keys(self) -> Tuple[str, ...]:
        return tuple(self._entries)

    def values(self) -> list[Any]:
        return [self._view(e) for e in self._entries.values()]

    def items(self) -> list[tuple[str, Any]]:
        """(name, value) pairs, dict-style (value = `view(entry)`)."""
        return [(n, self._view(e)) for n, e in self._entries.items()]

    def entries(self) -> list[Entry]:
        """The Entry records (name, version, obj, meta) in registration order."""
        return list(self._entries.values())

    # ------------------------------------------------------------------ whole-registry views
    def describe(self) -> list[dict[str, Any]]:
        """[{name, version, **meta}] in registration order: the content to fingerprint before/after a migration."""
        return [{"name": e.name, "version": e.version, **e.meta} for e in self._entries.values()]

    def check_all(self) -> list[str]:
        """Re-run the contract on every entry; the list of violation messages (empty = the registry honours its contract)."""
        out = []
        for e in self._entries.values():
            try:
                self.contract.check(e.name, e.obj, e.version, e.meta)
            except ContractError as err:
                out.append(str(err))
        return out

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {list(self._entries)})"
