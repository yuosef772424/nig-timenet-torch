"""The data pipeline and main notebooks: thin, static-checked, and RUN headless.

(The cross-asset notebook follows with I4.)

Static: committed without outputs, no ``def``/``class``, short cells, every call into the package matches a real
signature, every name a cell reads is defined by an earlier cell, the settings cell holds every option the cells use.
Run: the cells after the Colab-only setup section are executed here on a tiny synthetic dataset with the settings
cell's variables overridden, so a renamed option or function breaks THIS test, not the owner's Colab run.
"""

import ast
import builtins
import inspect
from importlib import import_module

import pytest

from nig_timenet_torch.colab import bootstrap
from tests.notebooks.helpers import (
    MAX_CELL_LINES,
    SETTINGS_MAX_LINES,
    code_cells,
    is_settings,
    load,
    logic_lines,
    run_notebook,
)

NAMES = ("data_pipeline", "main")
COMMON = {"TORCH_REPO", "TORCH_REF", "EXPECT_DEVICE", "TOKEN_PATH", "WORKDIR", "TORCH_DIR", "DRIVE_RUNS",
          "EXTRA_MODULES"}
NEEDS = {"data_pipeline": {"RAW_ROOT", "OUT_DIR", "NAME", "PRESETS", "CHANGES", "FETCH"},
         "main": {"ARTIFACT", "RUN_NAME", "WORK_ON", "DRIVE_RUNS", "LOCAL_RUNS", "RESTORE_FROM", "CHANGES",
                  "BATTERY_FOLDER", "BATTERY_COST", "BATTERY_VARIANTS", "REPO_SOURCE", "REPO_DIR", "UPDATE_DRIVE_COPY",
                  "DISCARD_LOCAL_CHANGES", "USE_ENV_CACHE", "ENV_CACHE_DIR"}}


def _check_call(call, fn, where):
    """The call binds to ``fn``'s signature; with ``*args`` / ``**kwargs`` only the names that are given are checked."""
    params = inspect.signature(fn).parameters
    names = {k.arg for k in call.keywords if k.arg}
    open_ended = any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords)
    if not open_ended:
        inspect.signature(fn).bind(*([0] * len(call.args)), **dict.fromkeys(names, 0))
    elif not any(p.kind is p.VAR_KEYWORD for p in params.values()):
        assert names <= set(params), f"{where}: {sorted(names - set(params))} not in {inspect.signature(fn)}"


@pytest.mark.parametrize("name", NAMES)
def test_clean_thin_and_the_settings_cell_is_complete(name):
    nb = load(name)
    assert nb["nbformat"] == 4
    for i, c, src in code_cells(nb):
        assert c["outputs"] == [] and c["execution_count"] is None, f"{name} cell {i}: commit without outputs"
        assert "ghp_" not in src and "github_pat_" not in src
        cap = SETTINGS_MAX_LINES if is_settings(c) else MAX_CELL_LINES
        assert logic_lines(src) <= cap, f"{name} cell {i}: move logic into the package"
        tree = ast.parse(src)
        assert not [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    tagged = [src for _, c, src in code_cells(nb) if "settings" in c["metadata"].get("tags", [])]
    assert len(tagged) == 1
    assigned = {t.id for n in ast.walk(ast.parse(tagged[0])) if isinstance(n, ast.Assign) for tgt in n.targets
                for t in ([tgt] if isinstance(tgt, ast.Name) else getattr(tgt, "elts", []))}
    assert (COMMON | NEEDS[name]) <= assigned
    assert not [n.value for n in ast.walk(ast.parse(tagged[0])) if isinstance(n, ast.Constant)
                and isinstance(n.value, str) and n.value.startswith("/tmp")]


@pytest.mark.parametrize("name", NAMES)
def test_every_call_matches_a_real_signature_and_every_name_is_defined_earlier(name):
    nb = load(name)
    defined, checked = set(dir(builtins)) | {"__name__"}, 0
    for i, _, src in code_cells(nb):
        tree = ast.parse(src)
        imported = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("nig_timenet_torch"):
                mod = import_module(n.module)
                for a in n.names:
                    if not hasattr(mod, a.name):
                        import_module(f"{n.module}.{a.name}")  # a submodule: from package import module
                    assert hasattr(mod, a.name), f"{name} cell {i}: {n.module} has no {a.name}"
                    imported[a.asname or a.name] = getattr(mod, a.name)
        for call in [n for n in ast.walk(tree) if isinstance(n, ast.Call)]:
            f = call.func
            if isinstance(f, ast.Name) and f.id in imported:
                _check_call(call, imported[f.id], f"{name} cell {i} {f.id}")
                checked += 1
            elif (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                  and inspect.ismodule(imported.get(f.value.id))):  # steps.prepare(...)
                assert hasattr(imported[f.value.id], f.attr), f"{name} cell {i}: no {f.value.id}.{f.attr}"
                _check_call(call, getattr(imported[f.value.id], f.attr), f"{name} cell {i} {f.value.id}.{f.attr}")
                checked += 1
            elif (isinstance(f, ast.Subscript) and isinstance(f.value, ast.Name) and f.value.id == "boot"
                  and isinstance(f.slice, ast.Constant)):
                assert hasattr(bootstrap, f.slice.value)
                _check_call(call, getattr(bootstrap, f.slice.value), f"{name} cell {i} boot[{f.slice.value!r}]")
                checked += 1
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                defined.add(n.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                defined |= {(a.asname or a.name).split(".")[0] for a in n.names}
            elif isinstance(n, ast.arg):
                defined.add(n.arg)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                defined.add(n.name)
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        assert used <= defined, f"{name} cell {i} reads undefined {sorted(used - defined)}"
    assert checked >= 4


def test_data_notebook_builds_a_named_dataset_that_the_loader_reads_back(tmp_path):
    base = "preprocessing_output_20261027"
    data = run_notebook("data_pipeline", {"OUT_DIR": str(tmp_path), "PRESETS": ("tiny",), "NAME": base})
    assert (tmp_path / f"{base}_latest.dataset").is_dir()
    assert any(p.name.startswith(f"{base}_2") for p in tmp_path.iterdir())
    assert data["loaded"].path.name == f"{base}_latest.dataset" and data["manifest"]["export"]["rows"] > 0
