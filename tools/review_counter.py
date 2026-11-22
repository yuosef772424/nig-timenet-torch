"""The R-REVIEW counter (rule F3) derived from the merge history instead of kept by hand.

PURPOSE:  Count the PRs merged on the first-parent history of HEAD after the highest PR number the last section of
  docs/reviews.md covers ("covers #48-#73"); `--write` puts the number on the "PRs since last R-REVIEW" line of STATUS.
TAGS:     tools, review, r-review, counter, reviews, merged, prs, first-parent
PITFALLS: A merge is a "Merge pull request #N ..." commit or a squash commit ending in "(#N)"; a PR merged inside
  another merge PR (MG4 style) has no commit of its own, so it is counted only through the review's "covers" range.
  The history is complete enough when it holds a PR number at or below the covered one (a shallow clone that stops
  earlier gives a too-small count: `history_reaches` is False and the test skips).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from tools._git import ToolError, git

ROOT = Path(__file__).resolve().parent.parent
REVIEWS_REL = "docs/reviews.md"
STATUS_REL = "docs/STATUS.md"
COUNTER_LINE = re.compile(r"(PRs since last R-REVIEW: )(\d+)")
MERGE_SUBJECT = re.compile(r"^Merge pull request #(\d+)\b")
SQUASH_SUBJECT = re.compile(r"\(#(\d+)\)\s*$")


def last_covered(reviews_text: str) -> int:
    """Highest PR number named after "covers" in the heading of the LAST section of docs/reviews.md."""
    headings = [ln for ln in reviews_text.splitlines() if ln.startswith("## ")]
    if not headings:
        raise ToolError(f"{REVIEWS_REL} has no `## ` section")
    m = re.search(r"covers? (.*)", headings[-1])
    numbers = [int(n) for n in re.findall(r"#(\d+)", m.group(1))] if m else []
    if not numbers:
        raise ToolError(f"the last section heading of {REVIEWS_REL} names no covered PRs ('covers #a-#b'): "
                        f"{headings[-1]!r}")
    return max(numbers)


def merged_prs(root: Path) -> list[int]:
    """PR numbers of the merge / squash commits on the first-parent history of HEAD, newest first."""
    out = git(root, "log", "--first-parent", "--format=%s", "HEAD") or ""
    found = []
    for subject in out.splitlines():
        m = MERGE_SUBJECT.match(subject) or SQUASH_SUBJECT.search(subject)
        if m:
            found.append(int(m.group(1)))
    return found


def history_reaches(prs: list[int], covered: int) -> bool:
    """True when the history holds a PR at or below ``covered``: the walk went back past the review boundary."""
    return any(n <= covered for n in prs)


def derived_counter(root: Path = ROOT) -> int:
    covered = last_covered((root / REVIEWS_REL).read_text(encoding="utf-8"))
    return len({n for n in merged_prs(root) if n > covered})


def write_status(root: Path, n: int) -> bool:
    """Set the number on STATUS's counter line; False when it already holds ``n``."""
    path = root / STATUS_REL
    text = path.read_text(encoding="utf-8")
    if not COUNTER_LINE.search(text):
        raise ToolError(f"{STATUS_REL} has no 'PRs since last R-REVIEW: N' line")
    new = COUNTER_LINE.sub(lambda m: f"{m.group(1)}{n}", text, count=1)
    if new == text:
        return False
    path.write_text(new, encoding="utf-8")
    return True


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.review_counter", description=__doc__.split("\n\n")[0])
    p.add_argument("--write", action="store_true", help="write the number to the counter line of docs/STATUS.md")
    args = p.parse_args(argv)
    try:
        n = derived_counter(root)
        print(n)
        if args.write:
            print("STATUS updated" if write_status(root, n) else "STATUS already holds it")
    except ToolError as exc:
        print(f"review_counter: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
