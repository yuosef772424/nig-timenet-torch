"""Rule B6 (docs/RULES.md): a doc that states a fact about code is generated or tested. A claim found false is fixed
in the PR that finds it, and a test is added here (this file grows per finding).

Findings so far:
  * V1   (TF side) reference/README said "exactly as main.ipynb": false.
  * R0   NOTES.md said "No PyTorch code yet": false since B1. Also: RULES.md promised checks "built in R0".
  * R0   every `path` quoted in the working docs must exist (or be listed below with a reason).
"""

import re

from tests.scan import ROOT

#: docs that quote paths as history, not as a claim about today's tree: results notes, audits (also the hardening
#: audits) and handoffs describe a past session's files, some since moved or deleted
#: Also historical records, excluded one by one: the port record (quotes the TF project's tree and files renamed since),
#: the owner decision records (immutable: a record is never edited after the merge), the append-only review log and
#: the files of the finished hardening campaign 2026-11 (its audit, plan and branch notes).
HISTORICAL_DIRS = ("docs/results/", "docs/audit/", "docs/handoff/", "docs/hardening/audit-", "docs/decisions/",
                   "docs/BUILD_PLAN.md", "docs/reviews.md", "docs/hardening/AUDIT-2026-11.md",
                   "docs/hardening/PLAN.md", "docs/hardening/BRANCH_AR_STREAM.md")
EXTRA_DOCS = ("CLAUDE.md", "NOTES.md", "notebooks/README.md")


def _live_docs() -> tuple[str, ...]:
    """Every doc under docs/ except the historical folders, plus the root docs (E-23: a hand-kept list of six)."""
    found = {p.relative_to(ROOT).as_posix() for p in (ROOT / "docs").rglob("*.md")}
    return tuple(sorted(rel for rel in found if not rel.startswith(HISTORICAL_DIRS)) + list(EXTRA_DOCS))


LIVE_DOCS = _live_docs()
PATH_IN_BACKTICKS = re.compile(r"`((?:docs|tests|tools|src|reference|notebooks|\.github)/[^`\s]*)`")

#: quoted paths that are legitimately absent today -> the reason (remove the entry the day the file exists; the
#: test below fails on a stale entry)
ABSENT_ON_PURPOSE = {
    "reference/README": "RULES.md V1 incident: that is the TF project's file, quoted as history",
    "docs/research/failure_registry.csv": "lives in the TF project (ROADMAP: Using the TF project for follow-up)",
    "docs/research/hardening/bottlenecks.csv": "lives in the TF project (ROADMAP: Using the TF project for follow-up)",
    "tools/recovered/": "deleted by HB8 (E-29); PLAN-2026-11b names it in the HB8 row (remove this entry when that "
                        "plan moves to the archive)",
    "docs/research/plan/lock/holdout.json": "the TF project's sealed-coin lock, quoted by the data contract",
    "docs/research/templates/": "the TF project's results templates, quoted by MODEL_DEV_PROTOCOL as the model",
    "tools/export_torch_artifact.py": "the TF project's export tool (the only change there, PLANNER_HANDOFF)",
    "tools/holdout_guard.py": "the TF project's holdout guard, quoted by the data contract and RULES D1",
    "tools/check_patterns.py": "the TF project's pattern check, quoted by RULES as the origin of a rule",
}


#: open PRs that may wait beyond the interval of rule F3 (F3 states only "every 5 merged PRs"; the old test used
#: 7 = 5 + 2, kept)
OPEN_PR_ALLOWANCE = 2


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _quoted_paths(text: str) -> set[str]:
    out = set()
    for m in PATH_IN_BACKTICKS.finditer(text):
        path = re.sub(r":\d+(-\d+)?$", "", m.group(1).rstrip(".,:;)").split("::")[0])
        if not any(c in path for c in "<*{") and not path.endswith("D-nnn"):  # placeholders / patterns
            out.add(path)
    return out


def test_notes_does_not_claim_there_is_no_pytorch_code():
    assert "no pytorch code yet" not in _read("NOTES.md").lower()
    assert (ROOT / "src" / "nig_timenet_torch" / "model" / "builder.py").is_file(), "the claim's opposite must be true"


def test_status_is_short():
    limit = int(re.search(r"STATUS\.md`? stays under (\d+) lines", _read("CLAUDE.md")).group(1))
    from tools.status_row import STATUS_MAX

    assert limit == STATUS_MAX, f"CLAUDE.md says {limit}, tools/status_row.py STATUS_MAX says {STATUS_MAX}"
    assert f"Keep this file under {limit} lines" in _read("docs/STATUS.md")
    n = len(_read("docs/STATUS.md").splitlines())
    assert n < limit, f"docs/STATUS.md has {n} lines, the limit is < {limit}: move detail to docs/ or a PR description"

def test_status_states_are_the_documented_ones():
    """Every row of every STATUS table (step ids are free text: PN1, MG4, ...; E-11 of the 2026-11b audit: the old
    id pattern skipped most rows), checked with the one state definition (tools.status_row.valid_state)."""
    from tools.status_row import STATES, status_rows, valid_state

    text = _read("docs/STATUS.md")
    listed = " / ".join(f"`{s}`" for s in STATES)
    assert f"States: {listed} / `skipped (reason)`." in text
    rows = status_rows(text)
    assert len(rows) >= 5, f"only {len(rows)} STATUS rows were checked: did the table layout change?"
    for _, cells in rows:
        assert valid_state(cells[2]), f"{cells[0]}: unknown state {cells[2]!r}"


def test_status_holds_open_rows_only():
    """HB7: STATUS keeps open rows (todo / in progress / PR open); a merged / closed / skipped row moves to
    docs/status_archive.md (`python -m tools.close_step --archive`)."""
    from tools.status_row import OPEN_STATES, status_rows

    for _, cells in status_rows(_read("docs/STATUS.md")):
        assert cells[2] in OPEN_STATES, (f"{cells[0]}: state {cells[2]!r} is closed: run "
                                         "`python -m tools.close_step --archive`")


def test_review_counter_is_derived_from_the_merge_history():
    """E-13: the 'PRs since last R-REVIEW' line of STATUS equals the number of PRs merged (first-parent history of
    HEAD) after the last PR the newest section of docs/reviews.md covers, and stays within the review interval of
    rule F3 plus the open PRs allowed to wait."""
    import pytest

    from tools import review_counter as rc

    stated = int(re.search(r"PRs since last R-REVIEW: (\d+)", _read("docs/STATUS.md")).group(1))
    interval = int(re.search(r"R-REVIEW every (\d+) merged PRs", _read("docs/RULES.md")).group(1))
    assert stated <= interval + OPEN_PR_ALLOWANCE, (f"{stated} PRs since the last R-REVIEW (docs/reviews.md): run it "
                                                    "before more merges")
    covered = rc.last_covered(_read("docs/reviews.md"))
    prs = rc.merged_prs(ROOT)
    if not rc.history_reaches(prs, covered):
        pytest.skip(f"the git history does not reach back past PR #{covered} (shallow clone): CI checks out "
                    "fetch-depth 0")
    derived = len({n for n in prs if n > covered})
    assert stated == derived, (f"STATUS says {stated} PRs since the last R-REVIEW, git history says {derived}: run "
                               "`python -m tools.review_counter --write`")


def test_register_ids_are_unique():
    """docs/bottlenecks.csv ids (B-nnn) are unique; B-041 was used twice once."""
    import csv
    ids = [r["id"] for r in csv.DictReader(_read("docs/bottlenecks.csv").splitlines())]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    assert not dup, f"duplicate ids in docs/bottlenecks.csv: {dup}"


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


def test_session_instructions_use_the_pr_sections_not_the_csvs():
    """Owner 2026-11-16: obstacles and suggestions go in the PR description; tools/collect_registers.py appends them to
    the CSVs. The live docs must not tell a session to edit the registers by hand, and must name both tools."""
    claude, rules = _read("CLAUDE.md"), _read("docs/RULES.md")
    assert "python -m tools.close_step" in claude and "python -m tools.collect_registers" in claude
    for text in (claude, rules, _read("docs/STATUS.md")):
        assert "append real obstacles" not in text.lower() and "appends real obstacles" not in text.lower()
    assert "appends to `docs/suggestions.csv`" not in rules
    template = _read(".github/pull_request_template.md")
    for heading in ("## Behaviour changes (rule B1)", "## Friction", "## Suggestions"):
        assert heading in template, f"the PR template lost {heading!r}"
    assert (ROOT / ".github" / "workflows" / "collect-registers.yml").is_file()


#: rule C7 (docs/PHILOSOPHY.md section 3): the headings of the failure reading, in order
FAILURE_READING = ("## What the model learned", "## Where it is right and where it fails",
                   "## The samples it failed on", "## Bias", "## Selective use", "## Why it failed",
                   "## Next hypotheses")
#: results notes written before rule C7 (2026-11-16): the list only shrinks (a note that gains the reading leaves it)
BEFORE_C7 = {"baseline-001.md", "deep-side-001.md", "e1-fixes-001.md", "gates-fixes-001.md",
             "nig-scale-default-001.md", "panel-001.md", "rolling-cpu-smoke-001.md", "staged-cpu-smoke-001.md"}
NOT_APPLICABLE = "Failure reading: not applicable ("
MIN_REASON_WORDS = 3


def test_results_notes_hold_the_failure_reading():
    """Rule C7 (owner 2026-11-16): a results note holds the seven headings of the failure reading, in order, or says in
    one line why none applies (``Failure reading: not applicable (<reason>)``: a smoke run, an arm that passed)."""
    notes = {p.name: p.read_text(encoding="utf-8") for p in (ROOT / "docs" / "results").glob("*.md")}
    assert BEFORE_C7 <= set(notes), f"remove from BEFORE_C7 (file gone): {sorted(BEFORE_C7 - set(notes))}"
    philosophy = _read("docs/PHILOSOPHY.md")
    for heading in FAILURE_READING:
        assert f"`{heading}`" in philosophy, f"docs/PHILOSOPHY.md section 3 lost the heading {heading!r}"
    for name, text in sorted(notes.items()):
        if name in BEFORE_C7:
            continue
        if NOT_APPLICABLE in text:
            reason = text.split(NOT_APPLICABLE, 1)[1].split(")", 1)[0]
            assert len(reason.split()) >= MIN_REASON_WORDS, (
                f"docs/results/{name}: '{NOT_APPLICABLE}<reason>)' needs a reason of at least {MIN_REASON_WORDS} "
                f"words, got {reason!r}")
            continue
        at = [text.find("\n" + h + "\n") for h in FAILURE_READING]
        assert min(at) >= 0 and at == sorted(at), (
            f"docs/results/{name}: the failure reading (rule C7) needs these headings in order: {FAILURE_READING}; "
            f"or the line '{NOT_APPLICABLE}<reason>)'")
        for heading, start in zip(FAILURE_READING, at, strict=True):
            body = text[start + len(heading) + 2:].split("\n## ", 1)[0]
            assert any(ln.strip() for ln in body.splitlines()), (
                f"docs/results/{name}: the section {heading!r} is empty: write at least one line (rule C7)")
