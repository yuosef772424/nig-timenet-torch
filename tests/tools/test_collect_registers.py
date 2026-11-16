"""tools/collect_registers.py: the PR-section parser (good rows, bad rows, a missing section), the CSV column mapping,
idempotence, the JSON input path and a mocked GitHub API (no network). Rule F1 / F5."""

import csv
import io
import json

import pytest

from tools import collect_registers as cr

FRICTION_HEADER = "date,step,what_slowed_or_broke,where,minutes_lost,root_cause,fix_idea\n"
SUGGESTION_HEADER = "date,step,kind,where,what,minutes_lost,proposed_fix,status\n"

GOOD = """## What was built
text

## Friction
<!-- an example in a comment is ignored
- step: X
-->
- step: TL1
  what_slowed_or_broke: the thing, with a comma
  where: src/a.py:12
  minutes_lost: 15
  root_cause: because
  fix_idea: do this
- step: TL1
  date: 2026-11-01
  what_slowed_or_broke: second
  where: docs/x.md
  minutes_lost: 0
  root_cause: r
  fix_idea: f

## Suggestions
- step: TL1
  kind: suggestion
  where: tools/x.py
  what: add a thing
  minutes_lost: 5
  proposed_fix: add it
- step: TL1
  kind: stale
  where: docs/y.md:3
  what: out of date
  minutes_lost: 1
  proposed_fix: fixed here
  status: fixed

## Checks
done
"""


def pr(number, body, merged_at="2026-11-17T10:00:00Z"):
    return {"number": number, "body": body, "merged_at": merged_at}


@pytest.fixture
def root(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / cr.FRICTION_REL).write_text(FRICTION_HEADER + "2026-11-02,P00,old row,x,5,c,f\n", encoding="utf-8")
    (tmp_path / cr.SUGGESTIONS_REL).write_text(SUGGESTION_HEADER, encoding="utf-8")  # header only, no final row
    (tmp_path / cr.STATE_REL).write_text(json.dumps({"collected": {}, "since": "2026-11-16T00:01:00Z"}),
                                         encoding="utf-8")
    return tmp_path


def rows(root, rel):
    return list(csv.DictReader((root / rel).read_text(encoding="utf-8").splitlines()))


# ----------------------------------------------------------------------------------------------------- parser
def test_good_rows_map_one_to_one_to_the_csv_columns():
    got = cr.parse_body(GOOD, "2026-11-17T10:00:00Z")
    assert got.errors == []
    f, s = got.rows["Friction"], got.rows["Suggestions"]
    assert [tuple(r) for r in f] == [cr.FRICTION_COLS] * 2 and [tuple(r) for r in s] == [cr.SUGGESTION_COLS] * 2
    assert f[0] == {"date": "2026-11-17", "step": "TL1", "what_slowed_or_broke": "the thing, with a comma",
                    "where": "src/a.py:12", "minutes_lost": "15", "root_cause": "because", "fix_idea": "do this"}
    assert f[1]["date"] == "2026-11-01"  # given
    assert s[0]["status"] == "open" and s[1]["status"] == "fixed"  # default and given


def test_none_rows_are_explicit_and_valid():
    body = "## Friction\n- none: nothing slowed me down\n\n## Suggestions\n- none\n"
    got = cr.parse_body(body, "2026-11-17T10:00:00Z")
    assert got.errors == [] and got.rows == {"Friction": [], "Suggestions": []}


@pytest.mark.parametrize("body, needle", [
    ("## Suggestions\n- none\n", "`## Friction` section is missing"),
    ("## Friction\n- none\n", "`## Suggestions` section is missing"),
    ("## Friction\n\n## Suggestions\n- none\n", "Friction: the section is empty"),
    ("## Friction\n- TODO: replace this line\n## Suggestions\n- none\n", "cannot read the line"),
    ("## Friction\n- none\n## Suggestions\n- kind: suggestion\n  what: w\n", "field 'where' is missing"),
    ("## Friction\n- step: a\n  what_slowed_or_broke: w\n  where: p\n  minutes_lost: many\n  root_cause: r\n"
     "  fix_idea: f\n## Suggestions\n- none\n", "minutes_lost must be a whole number"),
    ("## Friction\n- none\n## Suggestions\n- step: a\n  kind: idea\n  where: p\n  what: w\n  minutes_lost: 1\n"
     "  proposed_fix: f\n", "kind must be one of"),
    ("## Friction\n- none\n## Suggestions\n- step: a\n  colour: red\n", "unknown field 'colour'"),
    ("## Friction\n- none\n- step: a\n## Suggestions\n- none\n", "`- none` together with rows"),
    ("## Friction\n- step: a\n  step: b\n## Suggestions\n- none\n", "repeats the field 'step'"),
    ("## Friction\n  minutes_lost: 3\n## Suggestions\n- none\n", "before any `- ` bullet"),
    ("## Friction\n- none\n## Suggestions\n- step: a\n  date: yesterday\n", "date must be YYYY-MM-DD"),
])
def test_bad_rows_are_reported_not_guessed(body, needle):
    errors = cr.parse_body(body, "2026-11-17T10:00:00Z").errors
    assert any(needle in e for e in errors), errors


def test_a_missing_or_empty_body_reports_both_sections():
    assert len(cr.parse_body("", "2026-11-17T10:00:00Z").errors) == 2
    assert len(cr.parse_body(None, "2026-11-17T10:00:00Z").errors) == 2


def test_crlf_bodies_from_the_github_editor_parse():
    got = cr.parse_body(GOOD.replace("\n", "\r\n"), "2026-11-17T10:00:00Z")
    assert got.errors == [] and len(got.rows["Friction"]) == 2


def test_the_template_itself_does_not_validate_until_it_is_filled_in():
    from pathlib import Path

    text = (Path(cr.ROOT) / ".github" / "pull_request_template.md").read_text(encoding="utf-8")
    errors = cr.parse_body(text, "2026-11-17T10:00:00Z").errors
    assert len(errors) == 2 and all("cannot read the line" in e for e in errors)  # the TODO placeholder lines
    assert cr.parse_body(GOOD, "2026-11-17T10:00:00Z").errors == []
    for col in cr.FRICTION_COLS[1:] + cr.SUGGESTION_COLS[1:]:
        assert col in text, f"the template does not document the column {col!r}"


# ---------------------------------------------------------------------------------------------- collect / CSV
def test_collect_appends_rows_in_column_order_and_is_idempotent(root):
    res = cr.collect(root, [pr(59, GOOD)])
    assert res.collected == {59: (2, 2)} and not res.bad
    f, s = rows(root, cr.FRICTION_REL), rows(root, cr.SUGGESTIONS_REL)
    assert len(f) == 3 and f[0]["step"] == "P00" and f[1]["what_slowed_or_broke"] == "the thing, with a comma"
    assert len(s) == 2 and s[1]["status"] == "fixed" and s[0]["proposed_fix"] == "add it"  # header-only file handled
    before = (root / cr.FRICTION_REL).read_bytes(), (root / cr.SUGGESTIONS_REL).read_bytes()
    again = cr.collect(root, [pr(59, GOOD)])
    assert again.skipped == [59] and not again.collected
    assert ((root / cr.FRICTION_REL).read_bytes(), (root / cr.SUGGESTIONS_REL).read_bytes()) == before
    state = json.loads((root / cr.STATE_REL).read_text(encoding="utf-8"))
    assert state["collected"]["59"] == {"merged_at": "2026-11-17T10:00:00Z", "friction": 2, "suggestions": 2}


def test_a_bad_pr_writes_nothing_of_its_own_but_good_prs_still_go_in(root):
    res = cr.collect(root, [pr(60, "## Friction\n- none\n"), pr(61, GOOD, "2026-11-18T10:00:00Z")])
    assert list(res.bad) == [60] and res.collected == {61: (2, 2)}
    assert "60" not in json.loads((root / cr.STATE_REL).read_text(encoding="utf-8"))["collected"]
    assert len(rows(root, cr.FRICTION_REL)) == 3


def test_old_unmerged_and_skipped_prs(root):
    res = cr.collect(root, [pr(5, "", "2026-11-01T00:00:00Z"), pr(6, GOOD, None), pr(7, "nope")], skip={7: "no body"})
    assert not res.bad and not res.collected
    assert json.loads((root / cr.STATE_REL).read_text(encoding="utf-8"))["collected"]["7"]["skipped"] == "no body"
    assert len(rows(root, cr.FRICTION_REL)) == 1


def test_dry_run_changes_nothing(root):
    before = (root / cr.STATE_REL).read_bytes()
    assert cr.collect(root, [pr(59, GOOD)], dry_run=True).collected == {59: (2, 2)}
    assert (root / cr.STATE_REL).read_bytes() == before and len(rows(root, cr.FRICTION_REL)) == 1


def test_a_changed_csv_header_stops_the_run(root):
    (root / cr.FRICTION_REL).write_text("date,step\n", encoding="utf-8")
    with pytest.raises(cr.ToolError, match="differs from the collector's columns"):
        cr.collect(root, [pr(59, GOOD)])


def test_real_registers_have_the_collectors_columns():
    for rel, cols in ((cr.FRICTION_REL, cr.FRICTION_COLS), (cr.SUGGESTIONS_REL, cr.SUGGESTION_COLS)):
        header = (cr.ROOT / rel).read_text(encoding="utf-8").splitlines()[0]
        assert tuple(next(csv.reader([header]))) == cols


# ------------------------------------------------------------------------------------------------------ input
def test_cli_from_json_then_again(root, tmp_path, capsys):
    src = tmp_path / "prs.json"
    src.write_text(json.dumps([pr(59, GOOD), pr(60, "")]), encoding="utf-8")
    assert cr.main(["--from-json", str(src)], root=root) == 1  # PR 60 is bad
    out = capsys.readouterr()
    assert "PR #59: 2 friction row(s), 2 suggestion row(s)" in out.out and "PR #60 NOT collected" in out.err
    assert cr.main(["--from-json", str(src), "--skip", "60:empty body"], root=root) == 0
    assert "already collected: #59" in capsys.readouterr().out
    assert len(rows(root, cr.FRICTION_REL)) == 3


def test_json_shape_and_skip_syntax_are_validated(root, tmp_path, capsys):
    src = tmp_path / "bad.json"
    src.write_text('{"number": 1}', encoding="utf-8")
    assert cr.main(["--from-json", str(src)], root=root) == 1
    src.write_text("[]", encoding="utf-8")
    assert cr.main(["--from-json", str(src), "--skip", "x"], root=root) == 1
    assert "write N:reason" in capsys.readouterr().err


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_api_path_is_mocked_and_pages_until_old(root, capsys):
    seen = []
    pages = [
        [{"number": 59, "body": GOOD, "merged_at": "2026-11-17T10:00:00Z", "updated_at": "2026-11-17T10:00:00Z"},
         {"number": 58, "body": "closed, not merged", "merged_at": None, "updated_at": "2026-11-17T09:00:00Z"}],
        [{"number": 40, "body": "old", "merged_at": "2026-11-01T00:00:00Z", "updated_at": "2026-11-01T00:00:00Z"}],
        [{"number": 1, "body": "never read", "merged_at": "2026-10-01T00:00:00Z", "updated_at": "2026-10-01"}],
    ]

    def fake_urlopen(req, timeout):
        seen.append((req.full_url, req.headers["Authorization"]))
        return FakeResponse(json.dumps(pages[len(seen) - 1]).encode())

    env = {"GITHUB_TOKEN": "tkn"}
    assert cr.main(["--repo", "o/r"], root=root, urlopen=fake_urlopen, environ=env) == 0
    assert len(seen) == 2 and "repos/o/r/pulls?state=closed" in seen[0][0] and seen[0][1] == "Bearer tkn"
    assert len(rows(root, cr.FRICTION_REL)) == 3
    assert "PR #59" in capsys.readouterr().out


def test_no_input_is_an_error_not_a_guess(root, capsys):
    assert cr.main([], root=root, environ={}) == 1
    assert "no input" in capsys.readouterr().err
