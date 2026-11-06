"""The autoregressive-stream Colab notebooks' contract: thin (rules E4/G6), every call matches a real
signature, every name is defined earlier, and the whole thing RUNS from the first attempt on the
synthetic market with tiny training (the same guarantee the owner gets on Colab). The helpers are
shared with the panel notebook's test (tests/notebooks/test_panel_ar_stream_notebook.py)."""

import ast
import builtins
import inspect
import json
from importlib import import_module

import pytest

from tests.scan import ROOT

MAX_CELL_LINES = 22
SETTINGS_VARS = {"TORCH_REPO", "TORCH_REF", "EXPECT_DEVICE", "TOKEN_PATH", "WORKDIR", "TORCH_DIR", "DRIVE_RUNS",
                 "EXTRA_MODULES", "SOURCE", "COINS", "DAYS", "INTERVAL", "RAW_ROOT", "CONTEXT_LEN", "HORIZON_LEN",
                 "EVAL_HORIZON", "LOSS", "CHANGES", "VAL_SHARE", "TEST_SHARE"}
TINY = {"epochs": 2, "d_model": 32, "num_layers": 1, "batch_size": 128, "patience": 2, "verbose": False}


def load(name: str) -> dict:
    return json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text(encoding="utf-8"))


def code_cells(nb):
    """(is_setup, cell, source) for every code cell; the Setup section (after its heading, before the
    next heading) is Colab-only and skipped by the headless run."""
    in_setup, out = False, []
    for c in nb["cells"]:
        if c["cell_type"] == "markdown":
            src = "".join(c["source"])
            in_setup = src.startswith("## 1. Setup") or (in_setup and not src.startswith("## "))
            continue
        out.append((in_setup, c, "".join(c["source"])))
    return out


def check_call(call, fn, where):
    """Arity check only (keyword values become 0): the notebook's calls must fit the real signatures."""
    sig = inspect.signature(fn)
    open_ended = any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords)
    if not open_ended:
        kwargs = {k.arg: 0 for k in call.keywords if k.arg}
        try:
            sig.bind(*call.args, **kwargs)
        except TypeError as exc:
            raise AssertionError(f"{where}: {exc}") from exc


def check_static(name: str, settings_vars=SETTINGS_VARS, min_checked: int = 6):
    """The static half of the contract: clean, thin, complete settings, real signatures, defined names."""
    nb = load(name)
    assert nb["nbformat"] == 4
    for i, c, src in code_cells(nb):
        assert c["outputs"] == [] and c["execution_count"] is None, f"{name} cell {i}: commit without outputs"
        assert "ghp_" not in src and "github_pat_" not in src
        assert len(src.splitlines()) <= MAX_CELL_LINES, f"{name} cell {i}: move logic into the package"
        assert not [n for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                                         ast.ClassDef))]
    tagged = [c for _, c, _ in code_cells(nb) if "settings" in c["metadata"].get("tags", [])]
    assert len(tagged) == 1
    assigned = {t.id for n in ast.walk(ast.parse("".join(tagged[0]["source"]))) if isinstance(n, ast.Assign)
                for tgt in n.targets for t in ([tgt] if isinstance(tgt, ast.Name) else getattr(tgt, "elts", []))}
    assert settings_vars <= assigned, f"{name}: settings cell misses {sorted(settings_vars - assigned)}"
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
                check_call(call, imported[f.id], f"{name} cell {i} {f.id}")
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
    assert checked >= min_checked


def run_notebook(name: str, overrides):
    """Run the settings cell, apply ``overrides`` over its variables, then every non-Colab cell in one namespace."""
    ns = {"__name__": "__notebook__"}
    for in_setup, c, src in code_cells(load(name)):
        if in_setup:
            continue
        exec(compile(src, f"{name}-cell", "exec"), ns)  # noqa: S102
        if "settings" in c["metadata"].get("tags", []):
            ns.update(overrides)
    return ns


def test_clean_thin_and_the_settings_cell_is_complete():
    check_static("ar_stream")


def test_every_call_matches_a_real_signature_and_every_name_is_defined_earlier():
    check_static("ar_stream")


def test_the_notebook_runs_from_the_first_attempt(tmp_path):
    ns = run_notebook("ar_stream", {"SOURCE": "synthetic", "DAYS": 40, "EXPECT_DEVICE": "cpu",
                                    "COINS": ("AAA",), "HORIZON_LEN": 8, "EVAL_HORIZON": 4, "CHANGES": TINY,
                                    "DRIVE_RUNS": str(tmp_path / "runs")})
    assert ns["data"]["train_X"].shape[1] == ns["CONTEXT_LEN"] == 32
    assert ns["data"]["train_Y"].shape[1] == ns["FUTURE_LEN"] == 8   # data carries the LONGER future
    assert ns["history"]["best_val_dir_acc"] > 0.0 and len(ns["history"]["train_loss"]) == 2
    for key in ("direction_accuracy", "persistence_accuracy", "always_up_accuracy", "trivial_gap"):
        assert key in ns["report"]
    assert len(ns["report"]["direction_accuracy_per_step"]) == ns["EVAL_HORIZON"] == 4  # free eval horizon
    assert ns["control"]["shuffled_direction_accuracy"] == pytest.approx(ns["control"]["expected"], abs=0.1)
    assert "integrity" in ns["out"] and ns["out"]["verdict"]["rows"]  # the in-place battery ran
