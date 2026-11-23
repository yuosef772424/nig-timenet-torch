"""Generate docs/INDEX.md: one page that lists every results note, audit, handoff, decision, job file group, one-off
script and standalone doc of docs/.

PURPOSE:  ``python -m tools.build_index`` writes docs/INDEX.md from the files of docs/ (first heading of each note,
  the Status line of each handoff, job files counted per entry); ``--check`` exits 1 when the file is stale.
TAGS:     tools, index, docs, results, audits, handoffs, decisions, jobs, navigation
PITFALLS: Reads only file names, first headings, Status lines, the ``entry`` key of job JSON and the first docstring
  line of a script: nothing is judged. Never edit docs/INDEX.md by hand (``--check`` and tests/test_docs_index.py
  fail); it is not part of the generated MAP (docs, not code).
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX_REL = "docs/INDEX.md"
#: the file of an audit folder whose first heading names it (first one that exists, else the first ``*.md``)
AUDIT_LEAD = ("README.md", "NOTES.md")
STATUS_PREFIX = "Status:"


def first_heading(path: Path) -> str:
    """Text of the first ``# `` line of a markdown file ("" when none)."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def status_line(path: Path) -> str:
    """The text after ``Status:`` on the first line that starts with it ("" when the file has none)."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(STATUS_PREFIX):
            return line[len(STATUS_PREFIX):].strip()
    return ""


def script_summary(path: Path) -> str:
    """First line of the module docstring of a script, else its first comment line, else ""."""
    text = path.read_text(encoding="utf-8")
    try:
        doc = ast.get_docstring(ast.parse(text))
    except SyntaxError:
        doc = None
    if doc:
        return doc.strip().splitlines()[0]
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("# ").strip()
    return ""


def _cell(text: str) -> str:
    return text.replace("|", "/").strip()


def _results(docs: Path) -> list[str]:
    out = [f"- `docs/results/{p.name}`: {_cell(first_heading(p))}" for p in sorted((docs / "results").glob("*.md"))]
    return [f"{len(out)} notes (rule C7: the failure reading is inside each).", ""] + out


def _audit_lead(folder: Path) -> Path | None:
    for name in AUDIT_LEAD:
        if (folder / name).is_file():
            return folder / name
    mds = sorted(folder.glob("*.md"))
    return mds[0] if mds else None


def _audits(docs: Path) -> list[str]:
    out = []
    for d in sorted(p for p in (docs / "audit").iterdir() if p.is_dir()):
        lead = _audit_lead(d)
        files = sum(1 for f in d.rglob("*") if f.is_file() and "__pycache__" not in f.parts)
        head = _cell(first_heading(lead)) if lead else ""
        where = f" (`{lead.name}`)" if lead else ""
        out.append(f"- `docs/audit/{d.name}/`: {head}{where}, {files} files")
    return [f"{len(out)} audit folders.", ""] + out


def _handoffs(docs: Path) -> list[str]:
    out = []
    for p in sorted((docs / "handoff").glob("*.md")):
        out.append(f"- `docs/handoff/{p.name}`: Status: {status_line(p) or 'MISSING'}")
    return [f"{len(out)} handoffs (each carries a `Status:` line: open or closed with a pointer).", ""] + out


def _decisions(docs: Path) -> list[str]:
    files = sorted((docs / "decisions").glob("D-*.md"))
    out = [f"- `docs/decisions/{p.name}`: {_cell(first_heading(p))}" for p in files]
    return [f"{len(out)} decisions (format: `docs/decisions/README.md`).", ""] + out


def _jobs(docs: Path) -> list[str]:
    entries: Counter[str] = Counter()
    for p in sorted((docs / "jobs").glob("*.json")):
        entries[str(json.loads(p.read_text(encoding="utf-8")).get("entry", "?"))] += 1
    out = [f"- entry `{e}`: {n}" for e, n in sorted(entries.items())]
    return [f"{sum(entries.values())} job files in `docs/jobs/` (a job file is queued by the planner, "
            "`docs/COLAB_OPS.md` \"Queue route\"), per entry:", ""] + out


def _scripts(docs: Path) -> list[str]:
    paths = sorted(p for p in docs.rglob("*.py") if "__pycache__" not in p.parts)
    out = [f"- `{p.relative_to(docs.parent).as_posix()}`: {_cell(script_summary(p))}" for p in paths]
    return [f"{len(out)} scripts kept under docs/ (one-off analyses next to the note or audit they belong to; "
            "not run by CI).", ""] + out


def _standalone(docs: Path) -> list[str]:
    out = [f"- `docs/{p.name}`: {_cell(first_heading(p))}" for p in sorted(docs.glob("*.md")) if p.name != "INDEX.md"]
    return ["Also `CLAUDE.md` (the reading order), `NOTES.md`, `docs/MAP.md` (code map, generated), the registers "
            "`docs/bottlenecks.csv`, `docs/friction_log.csv`, `docs/suggestions.csv`, and `docs/cards/`, "
            "`docs/hardening/`.", ""] + out


def build(root: Path = ROOT) -> str:
    docs = root / "docs"
    parts = [
        ("Standalone docs", _standalone(docs)),
        ("Results notes", _results(docs)),
        ("Audits", _audits(docs)),
        ("Handoffs", _handoffs(docs)),
        ("Decisions", _decisions(docs)),
        ("Job files", _jobs(docs)),
        ("One-off scripts under docs/", _scripts(docs)),
    ]
    lines = ["# INDEX of docs/ (generated: `python -m tools.build_index`; never edit by hand)", "",
             "Where to look. Reading order: `CLAUDE.md`. Freshness: `python -m tools.build_index --check`.", ""]
    for title, body in parts:
        lines += [f"## {title}", ""] + body + [""]
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tools.build_index", description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 when docs/INDEX.md is stale")
    args = ap.parse_args(argv)
    target = ROOT / INDEX_REL
    text = build()
    if args.check:
        if not target.is_file() or target.read_text(encoding="utf-8") != text:
            print(f"{INDEX_REL} is stale: run `python -m tools.build_index`", file=sys.stderr)
            return 1
        print(f"{INDEX_REL} is fresh")
        return 0
    target.write_text(text, encoding="utf-8")
    print(f"wrote {INDEX_REL} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
