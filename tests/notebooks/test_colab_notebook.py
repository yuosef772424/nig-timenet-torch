"""T3: the Colab notebook is a THIN caller of ``nig_timenet_torch.colab`` (rules E4/G6; owner: every Colab run starts
from a notebook, no pasted cells). Nothing here needs a network or Colab: the notebook is checked statically.

  * committed without outputs, no token in it, no ``def``/``class``, short cells;
  * the three sections exist, in order;
  * every function the cells call exists with a compatible signature (an imported function, or a name taken from the
    exec'd ``bootstrap.py`` through ``boot["name"]``), so a renamed function breaks THIS test, not the owner's run;
  * every variable a cell reads is defined by an earlier cell (a typo in a setting breaks this test).
"""

import ast
import builtins
import inspect
import json
from importlib import import_module

import pytest

from nig_timenet_torch.colab import bootstrap
from tests.notebooks.helpers import MAX_CELL_LINES, code_cells, logic_lines
from tests.scan import ROOT

NB = ROOT / "notebooks" / "colab_nig_timenet_torch.ipynb"
SECTIONS = ("## 1. Setup", "## 2. Data", "## 3. Runner")


@pytest.fixture(scope="module")
def nb():
    return json.loads(NB.read_text(encoding="utf-8"))




def test_notebook_is_clean_and_thin(nb):
    assert nb["nbformat"] == 4 and code_cells(nb)
    for i, c, src in code_cells(nb):
        assert c["outputs"] == [] and c["execution_count"] is None, f"cell {i}: commit the notebook without outputs"
        assert "ghp_" not in src and "github_pat_" not in src, "no tokens in the notebook"
        n_lines = logic_lines(src)
        assert n_lines <= MAX_CELL_LINES, f"cell {i} is {n_lines} lines: move logic into colab/"
        tree = ast.parse(src)  # plain Python: no ! or % magics at all
        defs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        assert not defs, f"cell {i}: def/class belongs in a module (rule G6)"


def test_three_sections_in_order(nb):
    text = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    pos = [text.find(s) for s in SECTIONS]
    assert all(p >= 0 for p in pos) and pos == sorted(pos), pos
    assert "colab-queue" in text and "colab-results" in text and "republish" in text.lower()


def _calls(tree):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


def _check_call(call, fn, where):
    n_pos = len(call.args)
    assert not any(isinstance(a, ast.Starred) for a in call.args) and not any(k.arg is None for k in call.keywords)
    kwargs = {k.arg: 0 for k in call.keywords}
    try:
        inspect.signature(fn).bind(*([0] * n_pos), **kwargs)
    except TypeError as e:
        raise AssertionError(f"{where}: call does not match {fn.__name__}{inspect.signature(fn)}: {e}") from None


def test_every_call_into_the_package_matches_a_real_signature(nb):
    checked = 0
    for i, _, src in code_cells(nb):
        tree = ast.parse(src)
        imported = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("nig_timenet_torch"):
                mod = import_module(n.module)
                for a in n.names:
                    assert hasattr(mod, a.name), f"cell {i}: {n.module} has no {a.name}"
                    imported[a.asname or a.name] = getattr(mod, a.name)
        for call in _calls(tree):
            f = call.func
            if isinstance(f, ast.Name) and f.id in imported:
                _check_call(call, imported[f.id], f"cell {i} {f.id}")
                checked += 1
            elif (isinstance(f, ast.Subscript) and isinstance(f.value, ast.Name) and f.value.id == "boot"
                  and isinstance(f.slice, ast.Constant)):
                assert hasattr(bootstrap, f.slice.value), f"cell {i}: bootstrap.py has no {f.slice.value}"
                _check_call(call, getattr(bootstrap, f.slice.value), f"cell {i} boot[{f.slice.value!r}]")
                checked += 1
    assert checked >= 6, ("expected calls to setup_clones, install_package, acquire_token, check_environment, "
                          "prepare_artifact, Runner and republish")


def test_every_name_a_cell_reads_is_defined_earlier(nb):
    defined = set(dir(builtins)) | {"__name__"}
    for i, _, src in code_cells(nb):
        tree = ast.parse(src)
        for n in ast.walk(tree):  # names this cell defines (assign, import, for/with/except targets, lambda args)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                defined.add(n.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                defined |= {(a.asname or a.name).split(".")[0] for a in n.names}
            elif isinstance(n, ast.arg):
                defined.add(n.arg)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                defined.add(n.name)
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        assert used <= defined, f"cell {i} reads undefined {sorted(used - defined)}"


def test_the_settings_cell_names_what_the_runner_needs(nb):
    settings = next(src for _, c, src in code_cells(nb) if "settings" in c["metadata"].get("tags", []))
    names = {t.id for n in ast.walk(ast.parse(settings)) if isinstance(n, ast.Assign) for tgt in n.targets
             for t in ([tgt] if isinstance(tgt, ast.Name) else getattr(tgt, "elts", []))}
    need = {"TORCH_REPO", "TF_REPO", "EXPECT_DEVICE", "TOKEN_PATH", "WORKDIR", "DRIVE_RUNS", "SOURCE", "ARTIFACT"}
    assert need <= names
    assert 'EXPECT_DEVICE = "cuda"' in settings, "the default run is a GPU run (rule E1)"
    tree = ast.parse(settings)
    literals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert not [v for v in literals if v.startswith("/tmp")], "rule E2: no path under /tmp"
