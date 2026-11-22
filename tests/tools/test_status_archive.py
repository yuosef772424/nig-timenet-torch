"""tools.status_row archive helpers and ``close_step --archive`` / ``--check`` for closed rows still in STATUS (HB7)."""

import pytest

from tools import close_step as cs
from tools import status_row

STATUS = """# STATUS
| Step | Title | State | PR / job | Note |
|---|---|---|---|---|
| A1 | open one | in progress | | n |
| B2 | done one | merged | [#1](u) | verbatim |
| C3 | merged in a merge | merged (MG3) | #2 | |
| D4 | dropped | skipped (not needed) | | |
| E5 | rejected | closed (rejected) | | |
| F6 | next | PR open | | |
"""
ARCHIVE = """# archive

| Step | Title | State | PR / job | Note |
|---|---|---|---|---|
| Z0 | old | merged | | |

closing words
"""


def test_open_and_closed_rows_are_told_apart():
    assert status_row.OPEN_STATES == ("todo", "in progress", "PR open")
    assert status_row.closed_rows(STATUS) == ["B2", "C3", "D4", "E5"]


def test_archive_moves_closed_rows_verbatim_to_the_end_of_the_archive_table():
    status, archive, moved = status_row.archive_closed(STATUS, ARCHIVE)
    assert moved == ["B2", "C3", "D4", "E5"]
    assert [ln.split("|")[1].strip() for ln in status.splitlines() if ln.startswith("| ")][1:] == ["A1", "F6"]
    rows = [ln for ln in archive.splitlines() if ln.startswith("| ")]
    assert rows[2:] == [ln for ln in STATUS.splitlines() if ln.startswith(("| B2", "| C3", "| D4", "| E5"))]
    assert archive.splitlines().index(rows[-1]) < archive.splitlines().index("closing words")
    assert status_row.closed_rows(status) == []


def test_archive_without_closed_rows_changes_nothing_and_needs_a_table():
    only_open = "\n".join(ln for ln in STATUS.splitlines() if not ln.startswith(("| B2", "| C3", "| D4", "| E5")))
    assert status_row.archive_closed(only_open, "no table")[2] == []
    with pytest.raises(status_row.ToolError, match="no table"):
        status_row.archive_closed(STATUS, "no table here")


def test_cli_archive_and_check_report_closed_rows(tmp_path, capsys):
    (tmp_path / "docs").mkdir()
    (tmp_path / cs.STATUS_REL).write_text(STATUS, encoding="utf-8")
    (tmp_path / cs.ARCHIVE_REL).write_text(ARCHIVE, encoding="utf-8")
    assert cs.main(["--archive"], root=tmp_path) == 0
    assert "archived 4 row(s)" in capsys.readouterr().out
    assert cs.main(["--archive"], root=tmp_path) == 0
    assert "archived 0 row(s)" in capsys.readouterr().out
    text = (tmp_path / cs.STATUS_REL).read_text(encoding="utf-8")
    assert "B2" not in text and "A1" in text
    assert "| B2 | done one | merged |" in (tmp_path / cs.ARCHIVE_REL).read_text(encoding="utf-8")
