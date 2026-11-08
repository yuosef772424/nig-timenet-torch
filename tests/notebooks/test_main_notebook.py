"""notebooks/main.ipynb (T3b): the interactive Arabic notebook. Beyond the shared checks of test_pipeline_notebooks
(thin, signatures, defined names):

  * the settings cell lists EVERY RunSettings field with today's default (a new field or a changed default breaks it);
  * the explanations are Arabic and every section the task names is there;
  * the Colab-only cells are tagged, everything else runs headless on the fixture end to end, with a fake Drive;
  * the notebook never pushes to GitHub, never prints the token;
  * the 'no equivalent yet' table names real ROADMAP steps.
"""

import ast
import dataclasses
import re

import pytest

from nig_timenet_torch.core.config import updated
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.workflow.settings import RunSettings
from tests.notebooks.helpers import code_cells, is_settings, load, run_notebook
from tests.scan import ROOT

ARABIC = re.compile(r"[؀-ۿ]")


@pytest.fixture(scope="module")
def nb():
    return load("main")


def _text(nb, kind):
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == kind)


def _literal(nb, name):
    cell = next(src for _, c, src in code_cells(nb) if is_settings(c))
    for node in ast.parse(cell).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not in the settings cell")


def _leaves(obj, prefix=""):
    """Dotted paths of every settings field; a section (nested dataclass) is entered, a dict value is a leaf."""
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if dataclasses.is_dataclass(v):
            yield from _leaves(v, f"{prefix}{f.name}.")
        else:
            yield f"{prefix}{f.name}"


def test_changes_lists_every_run_setting_with_its_current_default(nb):
    changes = _literal(nb, "CHANGES")
    default = RunSettings(run_dir="x")
    expected = set(_leaves(default)) - {"run_dir", "data.name"}
    assert set(changes) == expected, (sorted(set(changes) ^ expected))
    as_run = updated(default, {**changes, "data.name": default.data.name})
    assert as_run == default, "a value in CHANGES differs from the code's default: the notebook is stale"
    assert _literal(nb, "BATTERY_COST") == default.evaluation.battery.cost_per_side == 0.0008
    assert _literal(nb, "ARTIFACT") != LoaderSettings().name


def test_every_setting_in_the_settings_cell_has_an_arabic_comment(nb):
    cell = next(src for _, c, src in code_cells(nb) if is_settings(c))
    tree = ast.parse(cell)
    lines = cell.splitlines()
    missing = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            span = lines[node.lineno - 1:node.end_lineno]
            if not any(ARABIC.search(ln.split("#", 1)[1]) for ln in span if "#" in ln):
                missing.append(node.targets[0].id)
        elif isinstance(node, ast.Dict) and len(node.keys) > 3:
            for k in node.keys:
                if not ARABIC.search(lines[k.lineno - 1].partition("#")[2]):
                    missing.append(ast.literal_eval(k))
    assert not missing, f"no Arabic comment on the same line: {missing}"


def test_explanations_and_comments_are_arabic_and_detailed(nb):
    md = _text(nb, "markdown")
    assert len(ARABIC.findall(md)) > 0.3 * len(re.sub(r"\s", "", md)), "the markdown must be mostly Arabic"
    assert len(nb["cells"]) >= 34 and sum(c["cell_type"] == "markdown" for c in nb["cells"]) >= 17
    assert len(md) > 25_000, "explain every section (the TF main has ~60k characters in all)"
    for needle in ("REPO_SOURCE", "UPDATE_DRIVE_COPY", "DISCARD_LOCAL_CHANGES", "GITHUB_TOKEN", "rank IC", "T-sharp",
                   "coverage", "T-profit", "0.0008", "RESTORE_FROM", "WORK_ON", "الأهداف"):
        assert needle in md, needle
    heads = re.findall(r"^## (.+)$", md, flags=re.M)
    assert len(heads) >= 14 and all(ARABIC.search(h) for h in heads), heads


def test_colab_only_cells_are_tagged_and_the_rest_is_headless(nb):
    tagged = [src for _, c, src in code_cells(nb) if "colab-only" in c["metadata"].get("tags", [])]
    assert len(tagged) >= 4
    for src in code_cells(nb):
        text = src[2]
        if "google.colab" in text and "drive.mount" in text:
            assert "colab-only" in src[1]["metadata"].get("tags", [])
    untagged = "\n".join(s for _, c, s in code_cells(nb) if "colab-only" not in c["metadata"].get("tags", []))
    assert "google.colab" not in untagged and "boot[" not in untagged


def test_the_notebook_never_pushes_to_github_and_never_prints_the_token(nb):
    code = "\n".join(s for _, _, s in code_cells(nb))
    assert "git push" not in code and "push_to_github" not in code and "ghp_" not in code and "github_pat_" not in code
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "print":
            for arg in ast.walk(node):
                assert not (isinstance(arg, ast.Name) and arg.id in ("token", "preset")), "print(token)"


def test_the_no_equivalent_table_names_real_roadmap_steps(nb):
    md = _text(nb, "markdown")
    table = md[md.index("ما ليس هنا") if "ما ليس هنا" in md else md.index("## ١٤"):]
    roadmap = (ROOT / "docs" / "ROADMAP.md").read_text(encoding="utf-8")
    headings = set(re.findall(r"^### ([A-Za-z0-9-]+)", roadmap, flags=re.M))
    steps = set(re.findall(r"\*\*(T\d\w*|G\dr?|M\dr?|TR\d|D\d|F\d)\b", table))
    assert steps, "the table must cite ROADMAP steps in bold"
    assert steps <= headings, f"not ROADMAP steps: {sorted(steps - headings)}"
    assert "Later" in roadmap


def test_main_notebook_runs_the_fixture_end_to_end_on_a_fake_drive(tmp_path, art):
    """Settings -> paths -> restore -> data -> split -> model -> train -> push -> health -> test report -> battery ->
    summary -> push -> battery of the finished folder, through the workflow.run steps, on the fixture artifact
    (1 epoch, CPU). The run works on the 'VM disk' (WORK_ON local) so the push to the fake Drive is exercised."""
    drive, local = tmp_path / "drive", tmp_path / "local"
    out = local / "t3b" / "out"
    small = {"train.epochs": 1, "train.batch_size": 64, "train.device": "cpu", "evaluation.health_max_rows": 50}
    ns = run_notebook("main", {"ARTIFACT": str(art), "RUN_NAME": "t3b", "WORK_ON": "local", "DRIVE_RUNS": str(drive),
                               "LOCAL_RUNS": str(local), "CHANGES": small, "BATTERY_FOLDER": str(out)})
    for file in ("baseline_summary.json", "test_report.json", "health_report.md", "battery_report.txt",
                 "battery_profit.json", "battery_table_val.parquet", "battery_table_test.parquet"):
        assert (out / file).is_file(), file
        assert (drive / "t3b" / "out" / file).is_file(), f"{file} was not pushed to the fake Drive"
    assert any((drive / "t3b" / "train").iterdir()), "the checkpoint folder was not pushed"
    assert ns["summary"]["epochs_done"] == 1 and ns["report"]["n_rows"] == len(ns["run"].idx["test"])
    assert "VERDICT" in ns["render"](ns["sections"]) and ns["pushed"]["already_on_drive"] is False
    # a second session on a "new VM": the earlier run is restored from the fake Drive before anything else
    local2 = tmp_path / "local2"
    ns2 = run_notebook("main", {"ARTIFACT": str(art), "RUN_NAME": "t3b-again", "WORK_ON": "local",
                                "DRIVE_RUNS": str(drive), "LOCAL_RUNS": str(local2), "RESTORE_FROM": "t3b",
                                "CHANGES": small, "BATTERY_FOLDER": ""})
    assert ns2["restored"]["copied"] > 0 and ns2["restored"]["restored_from"] == str(drive / "t3b")
