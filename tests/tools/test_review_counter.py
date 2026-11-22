"""tools.review_counter: the R-REVIEW counter derived from first-parent merge commits (E-13, rule F3)."""

import subprocess

import pytest

from tools import review_counter as rc

REVIEWS = ("# reviews\n\n## 2026-11-07 first (covers #1-#3)\n\ntext #99\n\n"
           "## 2026-11-21 fifth (inside MG4 #74; covers #48-#73)\n\nx #1\n")


def test_last_covered_reads_the_last_heading_only():
    assert rc.last_covered(REVIEWS) == 73
    assert rc.last_covered("## a (covers #41-#46: H0b, T3b)\n") == 46
    with pytest.raises(rc.ToolError, match="no covered PRs"):
        rc.last_covered("## nothing here\n")


def _vcs(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.email=a@b", "-c", "user.name=t", *args], check=True,
                   capture_output=True)


def test_counts_merge_and_squash_commits_after_the_covered_pr(tmp_path):
    _vcs(tmp_path, "init", "-q", "-b", "main")
    for subject in ("start", "Merge pull request #3 from x/y", "feat: thing (#5)", "Merge pull request #6 from x/z",
                    "docs: mention PR #9 in text", "feat: other (#7)"):
        _vcs(tmp_path, "commit", "-q", "--allow-empty", "-m", subject)
    prs = rc.merged_prs(tmp_path)
    assert prs == [7, 6, 5, 3]
    assert rc.history_reaches(prs, 4) and not rc.history_reaches([7, 6], 4)
    (tmp_path / "docs").mkdir()
    (tmp_path / rc.REVIEWS_REL).write_text("## r (covers #1-#4)\n", encoding="utf-8")
    assert rc.derived_counter(tmp_path) == 3


def test_write_status_sets_the_counter_line(tmp_path):
    (tmp_path / "docs").mkdir()
    path = tmp_path / rc.STATUS_REL
    path.write_text("PRs since last R-REVIEW: 1 (note)\n", encoding="utf-8")
    assert rc.write_status(tmp_path, 4) is True
    assert path.read_text(encoding="utf-8") == "PRs since last R-REVIEW: 4 (note)\n"
    assert rc.write_status(tmp_path, 4) is False
