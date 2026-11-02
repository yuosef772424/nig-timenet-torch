"""Rule B6 (docs/RULES.md): a doc that states a fact about code is generated or tested. A claim found false is fixed
in the PR that finds it, and a test is added here (this file grows per finding).

Findings so far:
  * V1   (TF side) reference/README said "exactly as main.ipynb": false.
  * R0   NOTES.md said "No PyTorch code yet": false since B1. Also: RULES.md promised checks "built in R0".
  * R0   every `path` quoted in the working docs must exist (or be listed below with a reason).
"""

import re

from tests.scan import ROOT

LIVE_DOCS = ("CLAUDE.md", "docs/STATUS.md", "docs/RULES.md", "docs/ROADMAP.md", "NOTES.md", "notebooks/README.md")
PATH_IN_BACKTICKS = re.compile(r"`((?:docs|tests|src|reference|notebooks|\.github)/[^`\s]*)`")
STATE_WORDS = {"todo", "in progress", "PR open", "merged", "closed (rejected)"}

#: quoted paths that are legitimately absent today -> the reason (remove the entry the day the file exists; the
#: test below fails on a stale entry)
ABSENT_ON_PURPOSE = {
    "docs/reviews.md": "created by the first R-REVIEW (RULES F3)",
    "docs/results/baseline-001.md": "written by step T5r",
    "reference/README": "RULES.md V1 incident: that is the TF project's file, quoted as history",
    "docs/research/failure_registry.csv": "lives in the TF project (ROADMAP: Using the TF project for follow-up)",
    "docs/research/hardening/bottlenecks.csv": "lives in the TF project (ROADMAP: Using the TF project for follow-up)",
}


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _quoted_paths(text: str) -> set[str]:
    out = set()
    for m in PATH_IN_BACKTICKS.finditer(text):
        path = m.group(1).rstrip(".,:;)").split("::")[0]
        if not any(c in path for c in "<*{") and not path.endswith("D-nnn"):  # placeholders / patterns
            out.add(path)
    return out


def test_notes_does_not_claim_there_is_no_pytorch_code():
    assert "no pytorch code yet" not in _read("NOTES.md").lower()
    assert (ROOT / "src" / "nig_timenet_torch" / "model" / "builder.py").is_file(), "the claim's opposite must be true"


def test_status_is_short():
    limit = int(re.search(r"STATUS\.md`? stays under (\d+) lines", _read("CLAUDE.md")).group(1))
    assert limit == 60, "CLAUDE.md and ROADMAP R0 say 60"
    assert f"Keep this file under {limit} lines" in _read("docs/STATUS.md")
    n = len(_read("docs/STATUS.md").splitlines())
    assert n < limit, f"docs/STATUS.md has {n} lines, the limit is < {limit}: move detail to docs/ or a PR description"


def test_status_states_are_the_documented_ones():
    text = _read("docs/STATUS.md")
    assert "States: `todo` / `in progress` / `PR open` / `merged` / `closed (rejected)` / `skipped (reason)`." in text
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) >= 5 and re.match(r"^(B\d+|[RT]\d+r?)$", cells[0]):
            state = cells[3] if cells[0].startswith("B") else cells[2]
            assert state in STATE_WORDS or state.startswith("skipped ("), f"{cells[0]}: unknown state {state!r}"


def test_quoted_paths_exist():
    missing = {}
    for doc in LIVE_DOCS:
        for path in sorted(_quoted_paths(_read(doc))):
            if not (ROOT / path).exists() and path not in ABSENT_ON_PURPOSE:
                missing.setdefault(path, []).append(doc)
    assert not missing, ("docs quote paths that do not exist (rule B6): fix the doc, or add the path to "
                         f"ABSENT_ON_PURPOSE with a reason:\n{missing}")


def test_absent_on_purpose_entries_are_still_absent():
    stale = [p for p in ABSENT_ON_PURPOSE if (ROOT / p).exists()]
    assert not stale, f"these files exist now - remove them from ABSENT_ON_PURPOSE: {stale}"
    quoted = set().union(*(_quoted_paths(_read(d)) for d in LIVE_DOCS))
    unused = [p for p in ABSENT_ON_PURPOSE if p not in quoted]
    assert not unused, f"no live doc quotes these any more - remove them from ABSENT_ON_PURPOSE: {unused}"


def test_rules_do_not_promise_work_r0_has_done():
    """RULES.md used to say "(built in R0)" for every check R0 builds; each must now point at the real test."""
    rules = _read("docs/RULES.md")
    assert "built in R0" not in rules, "R0 built these checks: name the test file in the 'Enforced by' column instead"


def test_every_rule_has_an_enforcement_cell():
    """RULES.md rows `| X1 | rule | why | enforced by |` (F rows have 3 cells): the last cell is never empty."""
    rows = [ln for ln in _read("docs/RULES.md").splitlines() if re.match(r"^\| [A-G]\d+ \|", ln)]
    assert rows
    for ln in rows:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        assert cells[-1], f"{cells[0]}: empty 'Enforced by' cell"
