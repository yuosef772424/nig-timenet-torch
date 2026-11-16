"""Update (or create) one step's row in docs/STATUS.md.

PURPOSE:  Set a step's state, PR link and branch in its STATUS table row, or add the row, and refuse any result that is
  not under the 60-line limit; fails loudly when the row is missing (and no --title to create it).
TAGS:     tools, status, row, roadmap, table, pr, state, branch
PITFALLS: Rows are split on `|`, so a cell holding a pipe breaks the match (the row is then reported as malformed,
  never silently skipped). A new row goes before the first `todo` row of the roadmap table.
"""

from __future__ import annotations

import re

from tools._git import ToolError

STATUS_MAX = 60  # docs/STATUS.md must have fewer lines than this (tests/test_docs_claims.py::test_status_is_short)
STATES = ("todo", "in progress", "PR open", "merged", "closed (rejected)")
BRANCH_NOTE = re.compile(r"branch `[^`]*`;? ?")
STEP_CELL = re.compile(r"\s*[/,]\s*")


def valid_state(state: str) -> bool:
    return state in STATES or (state.startswith("skipped (") and state.endswith(")"))


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_row_of(line: str, step: str) -> bool:
    if not line.startswith("|"):
        return False
    first = _cells(line)[0]
    return step in STEP_CELL.split(first)


def find_rows(lines: list[str], step: str) -> list[int]:
    return [i for i, ln in enumerate(lines) if _is_row_of(ln, step)]


def pr_cell(pr: int, repo_url: str | None) -> str:
    return f"[#{pr}]({repo_url}/pull/{pr})" if repo_url else f"#{pr}"


def _with_branch(note: str, branch: str | None) -> str:
    if not branch:
        return note
    note = BRANCH_NOTE.sub("", note).strip()
    return f"branch `{branch}`; {note}" if note else f"branch `{branch}`"


def _roadmap_table_end(lines: list[str]) -> int:
    """Index where a new row goes: in the LAST table with a ``Step | Title | State`` header (the roadmap table), before
    its first ``todo`` row, else at its end."""
    heads = [i for i, ln in enumerate(lines) if ln.startswith("|") and _cells(ln)[:3] == ["Step", "Title", "State"]]
    if not heads:
        raise ToolError("docs/STATUS.md has no roadmap table (header `| Step | Title | State |`) to add a row to")
    i = heads[-1] + 2
    while i < len(lines) and lines[i].startswith("|"):
        if len(_cells(lines[i])) >= 3 and _cells(lines[i])[2] == "todo":
            return i
        i += 1
    return i


def update_status(text: str, step: str, *, branch: str | None, pr: int | None = None, state: str | None = None,
                  title: str | None = None, repo_url: str | None = None) -> str:
    """The STATUS text with ``step``'s row updated (state, PR link, branch note) or created from ``title``.
    Raises ToolError when the row is missing without a title, a state is not one of the documented ones, the row is
    malformed, or the result has ``STATUS_MAX`` lines or more."""
    if state is not None and not valid_state(state):
        raise ToolError(f"unknown state {state!r}: one of {', '.join(STATES)} or 'skipped (reason)'")
    lines = text.split("\n")
    hits = find_rows(lines, step)
    if len(hits) > 1:
        raise ToolError(f"step {step!r} has {len(hits)} rows in docs/STATUS.md; fix the file by hand first")
    if not hits:
        if not title:
            raise ToolError(f"no row for step {step!r} in docs/STATUS.md: add it, or pass --title to create it")
        row = (f"| {step} | {title} | {state or 'in progress'} | {pr_cell(pr, repo_url) if pr else ''} | "
               f"{_with_branch('', branch)} |")
        lines.insert(_roadmap_table_end(lines), row)
    else:
        cells = _cells(lines[hits[0]])
        if len(cells) != 5:
            raise ToolError(f"the row of {step!r} has {len(cells)} cells, expected 5 (Step, Title, State, PR, Note)")
        if state:
            cells[2] = state
        if pr:
            cells[3] = pr_cell(pr, repo_url)
        if cells[2] in ("in progress", "PR open"):
            cells[4] = _with_branch(cells[4], branch)
        lines[hits[0]] = "| " + " | ".join(cells) + " |"
    out = "\n".join(lines)
    n = len(out.splitlines())
    if n >= STATUS_MAX:
        raise ToolError(f"docs/STATUS.md would have {n} lines, the limit is < {STATUS_MAX} (rule B6 test): "
                        "move detail out of the file first, then run again")
    return out
