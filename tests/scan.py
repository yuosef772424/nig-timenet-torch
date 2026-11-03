"""Static scans of ``src/nig_timenet_torch`` shared by the section-G ratchet tests (docs/RULES.md G1-G8) and the
C2 metric-ownership test. Pure ``ast`` + one import pass for registry names; nothing here is a test.

Every scan returns a plain list of ``"relative/path.py:line: what"`` strings (or a dict), so a failing test prints
the exact places and ``tests/ratchets.json`` pins only ``len(...)``.
"""

import ast
import importlib
import pkgutil
import re
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = ROOT / "src" / "nig_timenet_torch"
PKG = "nig_timenet_torch"


@cache
def sources() -> dict[str, tuple[str, ast.Module]]:
    """{path relative to the package, (text, parsed tree)} for every ``.py`` file of the package."""
    out = {}
    for p in sorted(PKG_DIR.rglob("*.py")):
        text = p.read_text(encoding="utf-8")
        out[p.relative_to(PKG_DIR).as_posix()] = (text, ast.parse(text))
    return out


def layer_of(rel: str) -> str:
    """Top-level folder of a module ("core", "model", ...); "" for files directly in the package."""
    parts = rel.split("/")
    return parts[0] if len(parts) > 1 else ""


# ---------------------------------------------------------------------------------------------------- G3 (size)
def module_lines() -> dict[str, int]:
    return {rel: len(text.splitlines()) for rel, (text, _) in sources().items()}


def function_lengths() -> dict[str, int]:
    """{"path:qualified_name": number of lines} for every function and method."""
    out = {}

    def walk(rel, node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[f"{rel}:{prefix}{child.name}"] = child.end_lineno - child.lineno + 1
                walk(rel, child, f"{prefix}{child.name}.")
            elif isinstance(child, ast.ClassDef):
                walk(rel, child, f"{prefix}{child.name}.")
            else:
                walk(rel, child, prefix)

    for rel, (_, tree) in sources().items():
        walk(rel, tree, "")
    return out


# ---------------------------------------------------------------------------------------- G1 (output names)
OUTPUT_NAME = re.compile(r"^y_(\{\}|high|low|close)(_[a-z_{}]+)?$")
#: the one module that owns the head/output vocabulary (HeadSpec + HEAD_REGISTRY templates)
OUTPUT_NAME_HOME = "model/heads.py"


def _template(node: ast.AST) -> str | None:
    """The text of a str constant or f-string, with every ``{expr}`` replaced by ``{}``; None for other nodes."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)
    return None


def output_name_literals() -> list[str]:
    """String literals that spell a model output key (``y_high``, ``y_{}_nu``, ...) outside OUTPUT_NAME_HOME."""
    found = []
    for rel, (_, tree) in sources().items():
        if rel == OUTPUT_NAME_HOME:
            continue
        inside_fstring = {id(v) for n in ast.walk(tree) if isinstance(n, ast.JoinedStr) for v in n.values}
        for node in ast.walk(tree):
            if id(node) in inside_fstring:
                continue
            t = _template(node)
            if t is not None and OUTPUT_NAME.match(t):
                found.append(f"{rel}:{node.lineno}: {t}")
    return found


# ----------------------------------------------------------------------------------- G2 (mode-name branching)
@cache
def import_all() -> None:
    package = importlib.import_module(PKG)
    for m in pkgutil.walk_packages(package.__path__, f"{PKG}."):
        importlib.import_module(m.name)


def registered_names() -> set[str]:
    """Every name registered in any ``Registry`` instance held by a module-level constant of the package."""
    import sys

    from nig_timenet_torch.core.registry import Registry

    import_all()
    names: set[str] = set()
    for mod_name, mod in list(sys.modules.items()):
        if mod_name == PKG or mod_name.startswith(f"{PKG}."):
            names |= {n for v in vars(mod).values() if isinstance(v, Registry) for n in v.names()}
    return names


def mode_name_comparisons() -> list[str]:
    """``x == "<registered name>"`` / ``!=`` / ``in ("<registered name>", ...)`` comparisons (name-based: a shared
    word such as "none" counts wherever it is compared)."""
    names = registered_names()
    found = []
    for rel, (_, tree) in sources().items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            lefts = [node.left, *node.comparators[:-1]]
            for op, left, right in zip(node.ops, lefts, node.comparators, strict=True):
                if not isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)):
                    continue
                consts = []
                for side in (left, right):
                    consts += list(side.elts) if isinstance(side, (ast.Tuple, ast.List, ast.Set)) else [side]
                hit = [c.value for c in consts if isinstance(c, ast.Constant) and isinstance(c.value, str)
                       and c.value in names]
                if hit:
                    found.append(f"{rel}:{node.lineno}: compares with registered name {hit[0]!r}")
    return found


# ---------------------------------------------------------------------------------------- G4 (dead settings)
SETTINGS_SUFFIXES = ("Settings", "Config")


def _is_dataclass(cls: ast.ClassDef) -> bool:
    for d in cls.decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        if (isinstance(target, ast.Name) and target.id == "dataclass") or (
                isinstance(target, ast.Attribute) and target.attr == "dataclass"):
            return True
    return False


def dead_settings() -> list[str]:
    """Fields of ``*Settings`` / ``*Config`` dataclasses that no code outside the class reads as ``obj.field``."""
    dead = []
    for rel, (_, tree) in sources().items():
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            if not (cls.name.endswith(SETTINGS_SUFFIXES) and _is_dataclass(cls)):
                continue
            fields = [s.target.id for s in cls.body if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)]
            inside = {id(n) for n in ast.walk(cls)}
            for field in fields:
                read = any(isinstance(n, ast.Attribute) and n.attr == field and isinstance(n.ctx, ast.Load)
                           and id(n) not in inside
                           for _, (_, t) in sources().items() for n in ast.walk(t))
                if not read:
                    dead.append(f"{rel}: {cls.name}.{field}")
    return dead


# ---------------------------------------------------------------------------------- G5 (one trainer, one step)
def optimizer_calls_outside_trainer() -> list[str]:
    """``.step()`` / ``.backward()`` / ``.zero_grad()`` calls on anything outside ``trainer/``."""
    found = []
    for rel, (_, tree) in sources().items():
        if layer_of(rel) == "trainer":
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("step", "backward", "zero_grad")):
                found.append(f"{rel}:{node.lineno}: .{node.func.attr}()")
    return found


def trainer_classes() -> list[str]:
    """Classes whose name ends in ``Trainer`` (G5: exactly one)."""
    return [f"{rel}: {n.name}" for rel, (_, tree) in sources().items() for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and n.name.endswith("Trainer")]


# ------------------------------------------------------------------------------------------ G7 (layer imports)
#: layer -> package layers it may import (None = anything). Files directly in the package are not layered.
ALLOWED_IMPORTS: dict[str, set[str] | None] = {
    "core": set(),
    "stats": {"core"},
    "data_io": None,
    "eval": None,
    "model": None,
    "trainer": None,
    "workflow": None,
    "colab": None,
    "jobs": None,
}
#: T3: the Colab layers sit on top; nothing below them imports them (colab/ and jobs/ may import everything)
_ON_TOP = {"workflow", "colab", "jobs"}
FORBIDDEN_IMPORTS = {"model": _ON_TOP, "trainer": _ON_TOP, "data_io": {"colab", "jobs"}, "eval": {"colab", "jobs"}}


def layer_import_violations() -> list[str]:
    """Imports (module level or lazy inside a function) that break the fixed layers of RULES.md G7."""
    found = []
    for rel, (_, tree) in sources().items():
        layer = layer_of(rel)
        if layer not in ALLOWED_IMPORTS:
            continue
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                targets = [node.module]
            elif isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            for t in targets:
                parts = t.split(".")
                if parts[0] != PKG or len(parts) < 2 or parts[1] == layer:
                    continue
                dst = parts[1]
                allowed = ALLOWED_IMPORTS[layer]
                if (allowed is not None and dst not in allowed) or dst in FORBIDDEN_IMPORTS.get(layer, set()):
                    found.append(f"{rel}:{node.lineno}: {layer} imports {dst} ({t})")
    return found


# ------------------------------------------------------------------------------------ G8 (registry folders)
def registry_folders() -> set[str]:
    """Folders in which some module instantiates ``Registry(...)``."""
    out = set()
    for rel, (_, tree) in sources().items():
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Registry"
                    and layer_of(rel)):
                out.add(layer_of(rel))
    return out


def _module_name(rel: str) -> str:
    parts = rel[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join([PKG, *parts])


def orphan_modules() -> list[str]:
    """Modules of a registry folder that neither call ``.register(`` / ``.decorator(`` nor are imported by any other
    module of the package. Component = one file + one registration line, with explicit imports (no auto-discovery):
    a file nobody imports can never get registered."""
    imported: set[str] = set()
    for _, tree in sources().values():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module)
                imported |= {f"{node.module}.{a.name}" for a in node.names}
            elif isinstance(node, ast.Import):
                imported |= {a.name for a in node.names}
    folders = registry_folders()
    orphans = []
    for rel, (_, tree) in sources().items():
        if layer_of(rel) not in folders or rel.endswith("__init__.py"):
            continue
        registers = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr in ("register", "decorator") for n in ast.walk(tree))
        if not registers and _module_name(rel) not in imported:
            orphans.append(rel)
    return orphans


# ------------------------------------------------------------------------------------ C2 (metric ownership)
METRIC_HOME = "stats/metrics.py"
#: names of metrics (whole-word match inside a function name) that must be defined only in METRIC_HOME
METRIC_WORDS = ("auc", "rank_ic", "pearson", "spearman", "ece", "mae", "rmse", "pinball", "wilson", "r2", "corr",
                "average_ranks")
METRIC_FUNCTION = re.compile(rf"(^|_)({'|'.join(METRIC_WORDS)})($|_)")
#: external implementations that would be a second copy of a metric
METRIC_IMPORTS = ("sklearn.metrics", "scipy.stats")


def _calls_metrics_module(fn: ast.AST) -> bool:
    """True if the function body uses ``metrics.<name>`` (an adapter around stats.metrics, not a second copy)."""
    return any(isinstance(n, ast.Name) and n.id == "metrics" for n in ast.walk(fn))


def metric_definitions_outside_stats() -> list[str]:
    """Function definitions that look like a metric, and imports of a third-party metric, outside ``stats/``."""
    found = []
    for rel, (_, tree) in sources().items():
        if layer_of(rel) == "stats":
            continue
        for node in ast.walk(tree):
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and METRIC_FUNCTION.search(node.name)
                    and not _calls_metrics_module(node)):
                found.append(f"{rel}:{node.lineno}: def {node.name}")
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(METRIC_IMPORTS):
                found.append(f"{rel}:{node.lineno}: from {node.module} import ...")
            elif isinstance(node, ast.Import):
                found += [f"{rel}:{node.lineno}: import {a.name}" for a in node.names
                          if a.name.startswith(METRIC_IMPORTS)]
            elif (isinstance(node, ast.Attribute) and node.attr == "corrcoef"):
                found.append(f"{rel}:{node.lineno}: corrcoef")
    return found
