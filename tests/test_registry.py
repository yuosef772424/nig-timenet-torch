"""P02: behaviour of core.registry (port of crypto-signal-prediction core/registry.py @ aeafc56)."""

import ast
from pathlib import Path

import pytest

from nig_timenet_torch.core.registry import (
    Contract,
    ContractError,
    DuplicateNameError,
    Entry,
    Registry,
    RegistryError,
    UnknownNameError,
)


def f(x):
    return x


def test_duplicate_name_raises():
    reg = Registry("task")
    reg.register("nig", f, version=1)
    with pytest.raises(DuplicateNameError, match="already registered"):
        reg.register("nig", f, version=2)
    assert isinstance(DuplicateNameError("x"), (RegistryError, ValueError))
    assert reg.entry("nig").version == 1  # failed registration left the registry unchanged


def test_unknown_name_raises_with_did_you_mean():
    reg = Registry("task")
    reg.register("nig", f, version=1)
    reg.register("binary", f, version=1)
    with pytest.raises(UnknownNameError) as ei:
        reg.get("nigg")
    msg = str(ei.value)
    assert "Did you mean 'nig'?" in msg and "['binary', 'nig']" in msg
    assert not msg.startswith('"')  # plain message, not KeyError's quoted repr
    for exc in (KeyError, ValueError):
        with pytest.raises(exc):
            reg["nigg"]
    with pytest.raises(UnknownNameError):
        reg.unregister("nope")


def test_contract_required_meta_enforced():
    reg = Registry("task", Contract("task", required_meta=("stat_keys",), optional_meta=("doc",)))
    with pytest.raises(ContractError, match=r"missing required metadata \['stat_keys'\]"):
        reg.register("nig", f, version=1)
    with pytest.raises(ContractError, match="unknown metadata"):
        reg.register("nig", f, version=1, stat_keys=(), extra=1)
    with pytest.raises(ContractError, match="version must be"):
        reg.register("nig", f, version=0, stat_keys=())
    with pytest.raises(ContractError, match="name must be"):
        reg.register(" nig", f, version=1, stat_keys=())
    with pytest.raises(ContractError, match="reserved"):
        Contract("task", required_meta=("version",))
    with pytest.raises(ContractError, match="kind"):
        Registry("task", Contract("head"))
    assert len(reg) == 0
    reg.register("nig", f, version="v1", stat_keys=("mae",), doc="x")
    assert reg.entry("nig").meta == {"stat_keys": ("mae",), "doc": "x"}


def test_decorator_registers_and_returns_object_unchanged():
    reg = Registry("head")

    @reg.decorator("dense", version=1, units=4)
    def build():
        return 42

    assert build() == 42 and reg["dense"] is build
    assert reg.entries() == [Entry("dense", 1, build, {"units": 4})]


def test_dict_style_reads_get_items_values():
    reg = Registry("task", view=lambda e: {"fn": e.obj, **e.meta})
    reg.register("a", f, version=1, k=1)
    reg.register("b", len, version=1, k=2)
    assert reg.names() == reg.keys() == ("a", "b") and list(reg) == ["a", "b"]
    assert "a" in reg and "z" not in reg and len(reg) == 2
    assert reg["a"] == {"fn": f, "k": 1} and reg.get("a") == reg["a"]
    assert reg.get("z", None) is None
    assert reg.values() == [{"fn": f, "k": 1}, {"fn": len, "k": 2}]
    assert reg.items() == [("a", {"fn": f, "k": 1}), ("b", {"fn": len, "k": 2})]
    assert reg.describe() == [{"name": "a", "version": 1, "k": 1}, {"name": "b", "version": 1, "k": 2}]
    plain = Registry("x")
    plain.register("a", f, version=1)
    assert plain["a"] is f  # default view = the object
    assert reg.unregister("a").obj is f and reg.names() == ("b",)


def test_check_all_reports_contract_violations():
    state = {"strict": False}

    def validator(name, obj, meta):
        return ["obj must be len"] if state["strict"] and obj is not len else []

    reg = Registry("task", Contract("task", validator=validator))
    reg.register("a", f, version=1)
    reg.register("b", len, version=1)
    assert reg.check_all() == []
    state["strict"] = True
    out = reg.check_all()
    assert len(out) == 1 and "'a' violates its contract: obj must be len" in out[0]

    def boom(name, obj, meta):
        raise RuntimeError("bad")

    with pytest.raises(ContractError, match="validator raised RuntimeError"):
        Registry("t", Contract("t", validator=boom)).register("x", f, version=1)


def test_registry_is_stdlib_only():
    """P02 DoD: core/registry.py imports only difflib, dataclasses, typing."""
    path = Path(__file__).resolve().parent.parent / "src" / "nig_timenet_torch" / "core" / "registry.py"
    mods = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module.split(".")[0])
    assert mods == {"difflib", "dataclasses", "typing"}
