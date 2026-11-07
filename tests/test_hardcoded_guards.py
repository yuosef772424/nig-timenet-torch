"""PURPOSE: Guards for the class of failures behind Colab runs #21-#23 (rule B3): schema facts spelled once, no integer
    column subscript on last_candles, target-name tuples ratcheted, autograd outside the train step under cudnn off.
TAGS: hardcoded guard, schema single source, last candles subscript, target tuple ratchet, cudnn autograd
PITFALLS: every scan takes {path: parsed tree} so a fake-source test proves it finds the pattern; the target-tuple
    count is pinned in ratchets.json and may only shrink.
"""

import ast

from nig_timenet_torch.core.schema import TARGET_COLUMNS
from tests import scan
from tests.ratchet import ratchet as _ratchet

SCHEMA_HOME = "core/schema.py"
SCHEMA_NAMES = {"TS_COL", "LAST_COLUMNS", "LAST_COLUMN_INDEX"}
TRAIN_STEP_HOME = "trainer/trainer.py"


def trees() -> dict[str, ast.Module]:
    return {rel: tree for rel, (_, tree) in scan.sources().items()}


def schema_redefinitions(srcs: dict[str, ast.Module]) -> list[str]:
    found = []
    for rel, tree in srcs.items():
        if rel == SCHEMA_HOME:
            continue
        for n in ast.walk(tree):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, ast.AnnAssign) else []
            found += [f"{rel}:{n.lineno}: {t.id} assigned" for t in targets
                      if isinstance(t, ast.Name) and t.id in SCHEMA_NAMES]
    return found


def integer_column_subscripts(srcs: dict[str, ast.Module]) -> list[str]:
    """``<expr naming last_candles>[..., <int>]``: a column taken by position."""
    found = []
    for rel, tree in srcs.items():
        for n in ast.walk(tree):
            if (isinstance(n, ast.Subscript) and "last_candles" in ast.unparse(n.value)
                    and isinstance(n.slice, ast.Tuple) and len(n.slice.elts) == 2
                    and isinstance(n.slice.elts[1], ast.Constant) and isinstance(n.slice.elts[1].value, int)):
                found.append(f"{rel}:{n.lineno}: {ast.unparse(n)}")
    return found


def target_tuple_literals(srcs: dict[str, ast.Module]) -> list[str]:
    """tuple / list / set literals of 2+ distinct price-target names, outside core/schema.py."""
    found = []
    for rel, tree in srcs.items():
        if rel == SCHEMA_HOME:
            continue
        for n in ast.walk(tree):
            if isinstance(n, (ast.Tuple, ast.List, ast.Set)):
                names = {e.value for e in n.elts if isinstance(e, ast.Constant) and e.value in TARGET_COLUMNS}
                if len(names) >= 2:
                    found.append(f"{rel}:{n.lineno}: {ast.unparse(n)}")
    return found


def _is_autograd_call(n: ast.AST) -> bool:
    return (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and ((n.func.attr == "grad" and ast.unparse(n.func.value).endswith("autograd"))
                 or n.func.attr == "backward"))


def _cudnn_off(w: ast.With) -> bool:
    return any("cudnn.flags" in ast.unparse(i.context_expr) and "enabled=False" in ast.unparse(i.context_expr)
               for i in w.items)


def autograd_without_cudnn_off(srcs: dict[str, ast.Module]) -> list[str]:
    """autograd.grad / .backward calls outside ``Trainer.train_step`` not inside ``with ...cudnn.flags(enabled=False)``
    (the #23 class: cuDNN RNN double backward)."""
    found = []

    def walk(rel, node, safe, in_step):
        if isinstance(node, ast.FunctionDef) and node.name == "train_step" and rel == TRAIN_STEP_HOME:
            in_step = True
        if isinstance(node, ast.With) and _cudnn_off(node):
            safe = True
        if _is_autograd_call(node) and not safe and not in_step:
            found.append(f"{rel}:{node.lineno}: {ast.unparse(node.func)}(...) outside cudnn.flags(enabled=False)")
        for c in ast.iter_child_nodes(node):
            walk(rel, c, safe, in_step)

    for rel, tree in srcs.items():
        walk(rel, tree, False, False)
    return found


def parse(code: str, rel: str = "x/y.py") -> dict[str, ast.Module]:
    return {rel: ast.parse(code)}


def test_schema_columns_are_defined_only_in_core_schema():
    assert schema_redefinitions(trees()) == []
    assert schema_redefinitions(parse("TS_COL = 3\nLAST_COLUMNS: tuple = ()")) != []


def test_no_integer_column_subscript_on_last_candles():
    assert integer_column_subscripts(trees()) == []
    assert integer_column_subscripts(parse("ts = np.asarray(d['last_candles'])[:, 3]"))
    assert not integer_column_subscripts(parse("ts = np.asarray(d['last_candles'])[:, TS_COL]"))


def test_price_target_tuple_literals_ratchet():
    _ratchet("G1_price_target_literals_outside_schema", target_tuple_literals(trees()))
    assert target_tuple_literals(parse("T = ('high', 'low', 'close')"))
    assert not target_tuple_literals(parse("T = ('high', 'high')"))


def test_autograd_outside_the_train_step_runs_with_cudnn_off():
    assert autograd_without_cudnn_off(trees()) == []
    bad = "def f():\n    torch.autograd.grad(l, p)\n    l.backward()\n"
    assert len(autograd_without_cudnn_off(parse(bad))) == 2
    good = ("def f():\n    with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):\n"
            "        torch.autograd.grad(l, p)\n")
    assert autograd_without_cudnn_off(parse(good)) == []
    step = "class Trainer:\n    def train_step(self):\n        total.backward()\n"
    assert autograd_without_cudnn_off(parse(step, "trainer/trainer.py")) == []
    assert autograd_without_cudnn_off(parse(step, "other.py"))
