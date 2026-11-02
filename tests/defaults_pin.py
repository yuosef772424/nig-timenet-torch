"""Collect and compare the DEFAULTS of the package (rule B1; used by tests/test_decisions.py).

A "default" is any value the code uses when the caller says nothing:
  * a parameter default of a public module-level function or of a public class constructor (nn.Module ``__init__``
    and dataclass fields alike)            -> key ``module.name(param)``
  * a ``default_*`` metadata value of a registry entry (``default_config`` is expanded per key)
                                             -> key ``registry:<kind>:<entry>.<meta>[.<key>]``
  * a module constant whose name contains DEFAULT or CONFIG (dict constants are expanded per key)
                                             -> key ``module.CONSTANT[.<key>]``

``python -m tests.defaults_pin`` prints the current map as JSON; ``--add`` writes keys missing from
``tests/pins/defaults.json`` into it (it never changes or removes an existing key: that needs a decision file).
"""

import dataclasses
import enum
import hashlib
import inspect
import json
import re
import sys
from pathlib import Path

from tests.scan import PKG, ROOT, import_all

PIN_FILE = ROOT / "tests" / "pins" / "defaults.json"
DECISIONS_DIR = ROOT / "docs" / "decisions"
DECISION_NAME = re.compile(r"^D-(\d{3})-[a-z0-9][a-z0-9-]*\.md$")
#: ``- `key`: `old` -> `new` `` (backticks around key and both values; values may contain spaces)
DECISION_LINE = re.compile(r"^\s*[-*]\s+`([^`]+)`:\s+`([^`]*)`\s+->\s+`([^`]*)`\s*$")


def _norm(v) -> str:
    """A stable text for a default value (no memory addresses, dict keys sorted)."""
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_norm(k)}: {_norm(x)}" for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))) + "}"
    if isinstance(v, (list, tuple, set, frozenset)):
        items = sorted(map(_norm, v)) if isinstance(v, (set, frozenset)) else [_norm(x) for x in v]
        open_, close = ("(", ")") if isinstance(v, tuple) else ("[", "]") if isinstance(v, list) else ("{", "}")
        return open_ + ", ".join(items) + close
    if callable(v) and not isinstance(v, type):
        return f"<callable {getattr(v, '__module__', '?')}.{getattr(v, '__qualname__', type(v).__name__)}>"
    r = repr(v)
    return f"<{type(v).__module__}.{type(v).__qualname__}>" if " at 0x" in r else r


def _signature_defaults(prefix: str, obj) -> dict[str, str]:
    out = {}
    if inspect.isclass(obj) and dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            if not f.init:
                continue
            if f.default is not dataclasses.MISSING:
                out[f"{prefix}({f.name})"] = _norm(f.default)
            elif f.default_factory is not dataclasses.MISSING:
                out[f"{prefix}({f.name})"] = _norm(f.default_factory())
        return out
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return out
    for p in sig.parameters.values():
        if p.default is not inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            out[f"{prefix}({p.name})"] = _norm(p.default)
    return out


def _expand(prefix: str, value) -> dict[str, str]:
    if isinstance(value, dict):
        return {f"{prefix}.{k}": _norm(v) for k, v in value.items()}
    return {prefix: _norm(value)}


def collect_defaults() -> dict[str, str]:
    from nig_timenet_torch.core.registry import Registry

    import_all()
    out: dict[str, str] = {}
    for mod_name, mod in sorted(sys.modules.items()):
        if not (mod_name == PKG or mod_name.startswith(f"{PKG}.")):
            continue
        for name, obj in sorted(vars(mod).items()):
            if isinstance(obj, Registry):
                for entry in obj.entries():
                    for meta_key, meta_val in entry.meta.items():
                        if meta_key.startswith("default"):
                            out |= _expand(f"registry:{obj.kind}:{entry.name}.{meta_key}", meta_val)
                continue
            if name.startswith("_"):
                continue
            if re.search("DEFAULT|CONFIG", name) and name.isupper():
                out |= _expand(f"{mod_name}.{name}", obj)
            elif inspect.isclass(obj) and issubclass(obj, enum.Enum):
                continue  # an Enum's signature is the functional API (names, module, ...), not a default of ours
            elif (inspect.isfunction(obj) or inspect.isclass(obj)) and getattr(obj, "__module__", None) == mod_name:
                out |= _signature_defaults(f"{mod_name}.{name}", obj)
    return out


def fingerprint(defaults: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(sorted(defaults.items())).encode()).hexdigest()[:16]


def load_pin() -> dict[str, str]:
    return json.loads(PIN_FILE.read_text(encoding="utf-8"))["defaults"]


def parse_decision(path: Path) -> list[tuple[str, str, str]]:
    """The ``(key, old, new)`` lines of one decision file."""
    return [m.groups() for line in path.read_text(encoding="utf-8").splitlines() if (m := DECISION_LINE.match(line))]


def decision_files(folder: Path = DECISIONS_DIR) -> list[Path]:
    return sorted(p for p in folder.glob("D-*.md"))


def apply_decisions(pinned: dict[str, str], folder: Path = DECISIONS_DIR) -> dict[str, str]:
    """The defaults in force: the pin file with every decision applied in filename order. A decision whose old
    value is not the value in force at that point raises (it describes a change that did not happen that way)."""
    effective = dict(pinned)
    for path in decision_files(folder):
        for key, old, new in parse_decision(path):
            if effective.get(key) != old:
                raise ValueError(f"{path.name}: `{key}` is {effective.get(key)!r} when this decision applies, "
                                 f"but the file says the old value is {old!r}")
            if new == "<removed>":
                del effective[key]
            else:
                effective[key] = new
    return effective


def compare(current: dict[str, str], effective: dict[str, str]) -> tuple[dict[str, tuple], list[str]]:
    """``(changed, added)``: ``changed`` maps a key in force to (value in force, current value or None if gone);
    ``added`` lists the current keys that are not pinned."""
    changed = {k: (v, current.get(k)) for k, v in effective.items() if current.get(k) != v}
    added = sorted(set(current) - set(effective))
    return changed, added


def main(argv: list[str]) -> None:
    current = collect_defaults()
    if "--add" not in argv:
        print(json.dumps(current, indent=1, sort_keys=True))
        return
    pinned = load_pin() if PIN_FILE.exists() else {}
    merged = {**current, **pinned}  # an existing pinned value always wins: changing it needs a decision file
    PIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    PIN_FILE.write_text(json.dumps({"fingerprint": fingerprint(merged), "defaults": dict(sorted(merged.items()))},
                                   indent=1) + "\n", encoding="utf-8")
    print(f"{len(set(current) - set(pinned))} key(s) added; {len(merged)} pinned")


if __name__ == "__main__":
    main(sys.argv[1:])
