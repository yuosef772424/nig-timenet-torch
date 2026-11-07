"""docs/RULES.md section G (growth ratchets) and C2 (one implementation per metric), one test per rule.

``tests/ratchets.json`` holds today's count of each bad pattern (measured by tests/scan.py). A count that GROWS fails;
a count that SHRINKS also fails until the file is lowered, so a fixed pattern cannot come back. Every non-zero count
has a row in docs/bottlenecks.csv. Failure messages list the exact places.
"""

import ast
import json

from tests import scan

RATCHETS = json.loads((scan.ROOT / "tests" / "ratchets.json").read_text(encoding="utf-8"))
COUNTS = RATCHETS["counts"]
MODULE_LINES, FUNCTION_LINES = 400, 80


def _ratchet(name: str, found: list[str]) -> None:
    pinned = COUNTS[name]
    listing = "\n  ".join(found) or "(none)"
    assert len(found) <= pinned, f"{name} grew: {len(found)} > pinned {pinned} (ratchet only shrinks):\n  {listing}"
    assert len(found) >= pinned, (f"{name} shrank to {len(found)} (pinned {pinned}): good - lower it in "
                                  f"tests/ratchets.json (and docs/bottlenecks.csv) so it cannot grow back:\n"
                                  f"  {listing}")


def test_every_pinned_count_has_a_test():
    """The ratchet file and this module stay in step: a count nobody checks is a dead pin."""
    source = "".join((scan.ROOT / "tests" / f).read_text(encoding="utf-8")
                     for f in ("test_rules_growth.py", "test_hardcoded_guards.py"))
    for name in COUNTS:
        assert f'_ratchet("{name}"' in source, f"{name} is pinned in ratchets.json but no test checks it"


def test_g1_output_names_are_defined_once():
    """G1: output keys (y_high, y_{t}_nu, ...) are spelled only in model/heads.py; others derive them."""
    _ratchet("G1_output_name_literals_outside_heads", scan.output_name_literals())


def test_g2_no_branching_on_mode_names():
    """G2: behaviour comes from a registered component, not from `if x == "<mode name>"` (registered names and
    declared vocabularies: ENCODERS, LEVEL_NORMS, *_MODES, *_KINDS)."""
    _ratchet("G2_mode_name_comparisons", scan.mode_name_comparisons())


def test_h4_no_config_reads_with_a_code_default():
    """H4: trainer and jobs read config values strictly; defaults live in registries / strict dataclasses (rule B5)."""
    _ratchet("H4_config_get_defaults_in_trainer_and_jobs", scan.config_get_defaults())


def test_g3_size_budget():
    """G3: module <= 400 lines, function <= 80 lines; a module over budget needs a reason in ratchets.json."""
    over_modules = sorted(m for m, n in scan.module_lines().items() if n > MODULE_LINES)
    unexcused = [m for m in over_modules if m not in RATCHETS["G3_size_exceptions"]]
    assert not unexcused, f"over {MODULE_LINES} lines without a reason in G3_size_exceptions: {unexcused}"
    stale = [m for m in RATCHETS["G3_size_exceptions"] if m not in over_modules]
    assert not stale, f"G3_size_exceptions lists modules that are within budget now (remove them): {stale}"
    lines = scan.module_lines()
    for mod, exc in RATCHETS["G3_size_exceptions"].items():
        assert set(exc) == {"max_lines", "why"} and exc["why"], f"{mod}: an exception is {{max_lines, why}}"
        assert lines[mod] <= exc["max_lines"], (f"{mod} has {lines[mod]} lines, its G3 exception allows "
                                                f"{exc['max_lines']}: split it or justify a new cap in ratchets.json")
    _ratchet("G3_modules_over_400_lines", over_modules)
    over_functions = sorted(f"{f}: {n} lines" for f, n in scan.function_lengths().items() if n > FUNCTION_LINES)
    _ratchet("G3_functions_over_80_lines", over_functions)


def test_g4_no_dead_settings():
    """G4: every field of a *Settings / *Config dataclass is read (as `obj.field`) by code outside its class."""
    _ratchet("G4_dead_settings_fields", scan.dead_settings())


def test_g5_one_trainer_one_step():
    """G5: optimizer.step / backward / zero_grad only in trainer/, and exactly one Trainer class."""
    _ratchet("G5_optimizer_calls_outside_trainer", scan.optimizer_calls_outside_trainer())
    assert len(scan.trainer_classes()) == 1, f"G5: one trainer per project, found {scan.trainer_classes()}"


def test_g6_notebooks_are_runners():
    """G6: no `def` / `class` in a notebook code cell; a study never imports another study."""
    offenders = []
    for nb in sorted((scan.ROOT / "notebooks").glob("*.ipynb")):
        for i, cell in enumerate(json.loads(nb.read_text(encoding="utf-8"))["cells"]):
            if cell["cell_type"] != "code":
                continue
            lines = [ln for ln in "".join(cell["source"]).splitlines() if not ln.lstrip().startswith(("!", "%"))]
            try:
                tree = ast.parse("\n".join(lines))
            except SyntaxError:
                continue  # a cell that is not plain Python is caught by tests/notebooks
            offenders += [f"{nb.name} cell {i}: {type(n).__name__} {n.name}" for n in ast.walk(tree)
                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    _ratchet("G6_notebook_defs", offenders)
    studies = scan.ROOT / "studies"
    cross = []
    for path in sorted(studies.glob("*.py")) if studies.is_dir() else []:
        others = {p.stem for p in studies.glob("*.py")} - {path.stem}
        for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            mods = ([n.module] if isinstance(n, ast.ImportFrom) and n.module else
                    [a.name for a in n.names] if isinstance(n, ast.Import) else [])
            cross += [f"{path.name} imports study {m}" for m in mods if m.split(".")[-1] in others]
    assert not cross, "a study never imports another study (G6); lift shared code into the package:\n  " + (
        "\n  ".join(cross))


def test_g7_fixed_layers():
    """G7: core imports nothing from the package; stats only core; model and trainer never import workflow."""
    _ratchet("G7_layer_import_violations", scan.layer_import_violations())


def test_g8_no_orphan_file_in_a_registry_folder():
    """G8: a file in a folder that holds a Registry either registers something or is imported by another module
    (explicit registration, no auto-discovery: a file nobody imports can never be registered)."""
    assert scan.registry_folders() >= {"model", "trainer", "data_io", "eval"}
    _ratchet("G8_orphan_modules_in_registry_folders", scan.orphan_modules())


def test_c2_one_implementation_per_metric():
    """C2: metrics are defined only in stats/metrics.py (an adapter that calls `metrics.<fn>` is not a copy)."""
    _ratchet("C2_metric_definitions_outside_stats", scan.metric_definitions_outside_stats())


# --- the scans themselves must catch what they claim to catch (B3: a guard needs its own test) -------------------
def _scan_source(monkeypatch, files: dict[str, str]) -> None:
    """Make tests.scan see a fake package made of `files` ({relative path: source})."""
    fake = {rel: (src, ast.parse(src)) for rel, src in files.items()}
    monkeypatch.setattr(scan, "sources", lambda: fake)


def test_scan_finds_output_names_and_metric_copies(monkeypatch):
    _scan_source(monkeypatch, {
        "model/heads.py": 'A = "y_high_nu"\n',
        "model/other.py": 'def f(t):\n    return f"y_{t}_nu", "y_close"\n',
        "trainer/x.py": "def my_auc(a, b):\n    return 0\n",
        "trainer/ok.py": "from m import metrics\ndef auc_fn(a):\n    return metrics.auc(a, a)\n",
    })
    assert scan.output_name_literals() == ["model/other.py:2: y_{}_nu", "model/other.py:2: y_close"]
    assert scan.metric_definitions_outside_stats() == ["trainer/x.py:1: def my_auc"]


def test_scan_finds_layer_violations_and_stray_optimizer_calls(monkeypatch):
    _scan_source(monkeypatch, {
        "core/a.py": "def f():\n    from nig_timenet_torch.model import heads\n",
        "stats/b.py": "from nig_timenet_torch.model.heads import X\n",
        "model/c.py": "import nig_timenet_torch.workflow.rolling\n",
        "workflow/d.py": "from nig_timenet_torch.model.heads import X\n",
        "model/e.py": "def g(opt):\n    opt.step()\n",
        "trainer/f.py": "def g(opt):\n    opt.step()\n",
    })
    assert [v.split(": ")[1] for v in scan.layer_import_violations()] == [
        "core imports model (nig_timenet_torch.model)", "stats imports model (nig_timenet_torch.model.heads)",
        "model imports workflow (nig_timenet_torch.workflow.rolling)"]
    assert scan.optimizer_calls_outside_trainer() == ["model/e.py:2: .step()"]


def test_scan_finds_dead_settings_and_orphans(monkeypatch):
    _scan_source(monkeypatch, {
        "w/s.py": ("from dataclasses import dataclass\n@dataclass(frozen=True)\nclass RunSettings:\n    used: int = 1\n"
                   "    dead: int = 2\n"),
        "w/use.py": "def f(s):\n    return s.used\n",
        "model/reg.py": "from nig_timenet_torch.core.registry import Registry\nR = Registry('x')\nR.register('a', 1)\n",
        "model/imported.py": "X = 1\n",
        "model/orphan.py": "X = 2\n",
        "model/user.py": "from nig_timenet_torch.model.imported import X\n",
    })
    assert scan.dead_settings() == ["w/s.py: RunSettings.dead"]
    assert scan.orphan_modules() == ["model/orphan.py", "model/user.py"]  # nobody imports user.py either


def test_scan_finds_declared_vocabularies_and_ignores_method_calls(monkeypatch):
    _scan_source(monkeypatch, {
        "m/a.py": 'FOO_MODES = ("fast", "slow")\n\ndef f(x):\n    return x == "fast"\n',
        "w/s.py": ("from dataclasses import dataclass\n@dataclass\nclass RunSettings:\n    rolling: int = 1\n"),
        "w/use.py": "def f(df):\n    return df.rolling(3)\n",
    })
    assert scan.declared_vocabularies() == {"fast", "slow"}
    assert [c.split(": ")[0] for c in scan.mode_name_comparisons()] == ["m/a.py:4"]
    assert scan.dead_settings() == ["w/s.py: RunSettings.rolling"]  # a method call of that name is not a read
