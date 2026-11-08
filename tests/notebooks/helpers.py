"""Notebook-test helpers (HF-32): reading a notebook, its code cells, and running its non-Colab cells headless."""

import json

from tests.scan import ROOT

MAX_CELL_LINES = 22          # lines of code per cell (blank and comment lines are not counted: Arabic comments)
SETTINGS_MAX_LINES = 200     # the one settings cell lists every setting with its comment


def code_cells(nb):
    return [(i, c, "".join(c["source"])) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]


def logic_lines(src: str) -> int:
    """Lines that carry code: not blank, not a pure comment line."""
    return len([ln for ln in src.splitlines() if ln.strip() and not ln.strip().startswith("#")])


def is_settings(cell) -> bool:
    return "settings" in cell["metadata"].get("tags", [])


def load(name):
    return json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text(encoding="utf-8"))


def sections(nb):
    """(is_setup, source, cell) of every code cell: the cells between '## 1. Setup' and the next heading, and any cell
    tagged ``colab-only``, need Colab (Drive mount, GitHub, a GPU) and are skipped by the headless run."""
    out, in_setup = [], False
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "markdown":
            in_setup = src.lstrip().startswith("## 1. Setup")
        else:
            out.append((in_setup or "colab-only" in c["metadata"].get("tags", []), src, c))
    return out


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
