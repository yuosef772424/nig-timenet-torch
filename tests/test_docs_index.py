"""docs/INDEX.md is generated (tools/build_index.py) and fresh; every handoff carries a Status line (hardening 2026-11b
E-06, E-07). The index is what makes results notes, audits, handoffs and decisions reachable from the entry docs."""

import re

from tests.scan import ROOT
from tools import build_index


def test_index_is_fresh():
    target = ROOT / build_index.INDEX_REL
    assert target.is_file(), "docs/INDEX.md is missing: run `python -m tools.build_index`"
    assert target.read_text(encoding="utf-8") == build_index.build(ROOT), (
        "docs/INDEX.md is stale: run `python -m tools.build_index` and commit it")


def test_every_handoff_has_a_status_line():
    status = re.compile(r"^Status: (open( \(.+\))?|closed \(.+\))\s*$")
    bad = []
    for p in sorted((ROOT / "docs" / "handoff").glob("*.md")):
        head = p.read_text(encoding="utf-8").splitlines()[:8]
        if not any(status.match(ln) for ln in head):
            bad.append(p.name)
    assert not bad, f"handoffs without `Status: open` / `Status: closed (<pointer>)` near the top: {bad}"


def test_every_listed_kind_is_in_the_index():
    text = (ROOT / build_index.INDEX_REL).read_text(encoding="utf-8")
    for pattern in ("docs/results/*.md", "docs/handoff/*.md", "docs/decisions/D-*.md", "docs/*.md"):
        for p in (ROOT).glob(pattern):
            if p.name != "INDEX.md":
                assert f"`{p.relative_to(ROOT).as_posix()}`" in text, f"{p} is not in docs/INDEX.md"
    for d in (ROOT / "docs" / "audit").iterdir():
        if d.is_dir():
            assert f"`docs/audit/{d.name}/`" in text


def test_entry_docs_link_the_index():
    for rel in ("CLAUDE.md", "docs/STATUS.md"):
        assert "docs/INDEX.md" in (ROOT / rel).read_text(encoding="utf-8"), f"{rel} does not link docs/INDEX.md"


def test_build_reads_a_small_tree(tmp_path):
    fake = tmp_path / "docs"
    for sub in ("results", "audit", "handoff", "decisions", "jobs"):
        (fake / sub).mkdir(parents=True)
    (fake / "A.md").write_text("# Title A\n", encoding="utf-8")
    (fake / "handoff" / "h.md").write_text("# H\n\nStatus: open\n", encoding="utf-8")
    text = build_index.build(tmp_path)
    assert "`docs/A.md`: Title A" in text and "Status: open" in text
