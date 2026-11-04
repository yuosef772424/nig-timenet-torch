"""The three pipeline notebooks (data_pipeline, main, cross_asset_pipeline): thin, static-checked, and RUN headless.

Static: committed without outputs, no ``def``/``class``, short cells, every call into the package matches a real
signature, every name a cell reads is defined by an earlier cell, the settings cell holds every option the cells use.
Run: the cells after the Colab-only setup section are executed here on a tiny synthetic dataset with the settings
cell's variables overridden, so a renamed option or function breaks THIS test, not the owner's Colab run.
"""

import ast
import builtins
import inspect
import json
from importlib import import_module

import pytest

from nig_timenet_torch.colab import bootstrap
from tests.notebooks.test_colab_notebook import MAX_CELL_LINES, code_cells
from tests.scan import ROOT

NAMES = ("data_pipeline", "main", "cross_asset_pipeline")
COMMON = {"TORCH_REPO", "TORCH_REF", "EXPECT_DEVICE", "TOKEN_PATH", "WORKDIR", "TORCH_DIR", "DRIVE_RUNS",
          "EXTRA_MODULES"}
NEEDS = {"data_pipeline": {"RAW_ROOT", "OUT_DIR", "NAME", "PRESETS", "CHANGES", "FETCH"},
         "main": {"DATA_DIR", "DATA_NAME", "RUN_DIR", "CHANGES", "ROLLING", "RETRAIN"},
         "cross_asset_pipeline": {"RAW_ROOT", "DATA_DIR", "DATA_NAME", "RUN_DIR", "BUILD", "DATA_PRESETS",
                                  "DATA_CHANGES", "CHANGES", "ROLLING", "RETRAIN"}}
TINY_TRAIN = {"train.epochs": 1, "train.batch_size": 8, "train.device": "cpu", "train.verbose": False}


def _check_call(call, fn, where):
    """The call binds to ``fn``'s signature; with ``*args`` / ``**kwargs`` only the names that are given are checked."""
    params = inspect.signature(fn).parameters
    names = {k.arg for k in call.keywords if k.arg}
    open_ended = any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords)
    if not open_ended:
        inspect.signature(fn).bind(*([0] * len(call.args)), **dict.fromkeys(names, 0))
    elif not any(p.kind is p.VAR_KEYWORD for p in params.values()):
        assert names <= set(params), f"{where}: {sorted(names - set(params))} not in {inspect.signature(fn)}"


def load(name):
    return json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text(encoding="utf-8"))


def sections(nb):
    """(is_setup, source) of every code cell: the cells between '## 1. Setup' and the next heading need Colab."""
    out, in_setup = [], False
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "markdown":
            in_setup = src.lstrip().startswith("## 1. Setup")
        else:
            out.append((in_setup, src, c))
    return out


@pytest.mark.parametrize("name", NAMES)
def test_clean_thin_and_the_settings_cell_is_complete(name):
    nb = load(name)
    assert nb["nbformat"] == 4
    for i, c, src in code_cells(nb):
        assert c["outputs"] == [] and c["execution_count"] is None, f"{name} cell {i}: commit without outputs"
        assert "ghp_" not in src and "github_pat_" not in src
        assert len(src.splitlines()) <= MAX_CELL_LINES, f"{name} cell {i}: move logic into the package"
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
                    assert hasattr(mod, a.name), f"{name} cell {i}: {n.module} has no {a.name}"
                    imported[a.asname or a.name] = getattr(mod, a.name)
        for call in [n for n in ast.walk(tree) if isinstance(n, ast.Call)]:
            f = call.func
            if isinstance(f, ast.Name) and f.id in imported:
                _check_call(call, imported[f.id], f"{name} cell {i} {f.id}")
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


def run_notebook(name, overrides):
    """Run the settings cell, apply ``overrides`` over its variables, then every non-Colab cell in one namespace."""
    ns = {"__name__": "__notebook__"}
    for in_setup, src, c in sections(load(name)):
        if in_setup:
            continue
        exec(compile(src, f"{name}-cell", "exec"), ns)  # noqa: S102
        if "settings" in c["metadata"].get("tags", []):
            ns.update(overrides)
    return ns


def test_data_notebook_builds_a_named_dataset_that_the_main_notebook_trains_on(tmp_path):
    base = "preprocessing_output_20261027"
    data = run_notebook("data_pipeline", {"OUT_DIR": str(tmp_path), "PRESETS": ("tiny",), "NAME": base})
    assert (tmp_path / f"{base}_latest.dataset").is_dir()
    assert any(p.name.startswith(f"{base}_2") for p in tmp_path.iterdir())
    assert data["loaded"].path.name == f"{base}_latest.dataset" and data["manifest"]["export"]["rows"] > 0
    ns = run_notebook("main", {"DATA_DIR": str(tmp_path), "DATA_NAME": base, "RUN_DIR": str(tmp_path / "r"),
                               "CHANGES": TINY_TRAIN, "RETRAIN": {"train.seed": 1}})
    assert ns["result"].fit.epochs_done == 1 and ns["second"].settings.train.seed == 1 and (tmp_path / "r").exists()
    full = next(p.name for p in tmp_path.iterdir() if p.name.startswith(f"{base}_2"))
    ns = run_notebook("main", {"DATA_DIR": str(tmp_path), "DATA_NAME": full, "RUN_DIR": "", "CHANGES": TINY_TRAIN})
    assert ns["loaded"].path.name == full and ns["second"] is None


def test_cross_asset_notebook_builds_prepares_grids_trains_and_retrains(tmp_path):
    ns = run_notebook("cross_asset_pipeline", {
        "BUILD": True, "DATA_PRESETS": ("tiny",), "DATA_DIR": str(tmp_path), "RUN_DIR": str(tmp_path / "r"),
        "CHANGES": {"kind": "panel", **TINY_TRAIN}, "RETRAIN": {"panel_loss.w_ic": 1.0}})
    assert ns["manifest"]["export"]["rows"] > 0 and ns["grid"].mask.sum() > 0 and ns["grid"].y_reg.ndim == 3
    assert ns["result"].report["n_rows"] > 0 and ns["second"].settings.panel_loss.w_ic == 1.0


def test_main_notebook_runs_rolling_when_asked(tmp_path):
    from nig_timenet_torch.data.pipeline import build_artifact
    from tests.data.helpers import longer_settings
    s = longer_settings(100).updated({"output.directory": str(tmp_path)})
    build_artifact(s)
    rolling = {"train_span": "30D", "val_span": "6D", "test_span": "8D", "step": "20D", "embargo": "1h"}
    ns = run_notebook("main", {"DATA_DIR": str(tmp_path), "RUN_DIR": str(tmp_path / "r"), "CHANGES": TINY_TRAIN,
                               "ROLLING": rolling})
    assert len(ns["result"].folds) >= 2 and ns["settings"].rolling is not None
