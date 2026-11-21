"""CI warning when a decision number this PR adds is also used by another branch that is not merged here.

PURPOSE:  Print a GitHub warning annotation per decision number this PR shares with an unmerged branch (two PRs that
  both took the next free D-NNN), so the clash is seen before the second PR merges; never fails the job.
TAGS:     tools, decisions, numbering, clash, ci, warning, open pull requests
PITFALLS: Needs every branch fetched (CI checkout uses fetch-depth 0, which fetches refs/remotes/origin/*); a branch
  whose head is already in HEAD is history and never counts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools import decision_draft as dd
from tools._git import ToolError, merge_base

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.decision_clash", description=__doc__.split("\n\n")[0])
    p.add_argument("--base", default="origin/main", help="PR base ref (default origin/main; its merge-base is used)")
    args = p.parse_args(argv)
    try:
        warnings = dd.open_clashes(root, dd.pr_added_decisions(root, merge_base(root, args.base)))
    except ToolError as exc:
        print(f"::warning title=decision numbers not checked::{exc}")
        return 0
    for w in warnings:
        print(f"::warning title=decision number clash::{w}")
    print(f"decision_clash: {len(warnings)} clash(es)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
