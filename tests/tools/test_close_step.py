"""tools/close_step.py on a temporary git repository (no torch, no network): old values after decisions, numbering and
clash warnings, in-place updates, the B1 section, the STATUS row and every --check failure (rules B1, B6)."""

import json
import subprocess

import pytest

from tests import defaults_pin as dp
from tools import close_step as cs
from tools import decision_draft as dd
from tools import status_row

STATUS = """# STATUS
> Keep this file under 60 lines.

| Step | Title | State | PR / job | Note |
|---|---|---|---|---|
| ST1 | Staged | in progress | | first |
| T9 | Later | todo | | |
"""


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.email=a@b", "-c", "user.name=t", *args], check=True,
                   capture_output=True)


def _pin(root, defaults):
    p = root / dd.PIN_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"fingerprint": dp.fingerprint(defaults), "defaults": defaults}), encoding="utf-8")


def _decision(root, name, *lines):
    (root / dd.DECISIONS_REL / name).write_text("# D\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


class Fake:
    """Hooks with a mutable ``current`` defaults map and a log of pytest calls."""

    def __init__(self):
        self.current = {"m.f(x)": "3", "m.g(y)": "'a'"}
        self.map = "map v1"
        self.failing: set[str] = set()
        self.calls: list[tuple[str, str | None]] = []

    def pytest(self, root, target, env_var):
        self.calls.append((target, env_var))
        if env_var:
            self.failing.discard(target)  # the regeneration fixed it
        return 1 if target in self.failing else 0

    def hooks(self):
        return cs.Hooks(defaults=lambda: dict(self.current), map_text=lambda: self.map, pytest=self.pytest,
                        today=lambda: "2026-11-16")


@pytest.fixture
def repo(tmp_path):
    (tmp_path / dd.DECISIONS_REL).mkdir(parents=True)
    _decision(tmp_path, "D-001-a.md", "- `m.f(x)`: `1` -> `2`")
    _decision(tmp_path, "D-002-b.md", "- `m.f(x)`: `2` -> `3`")
    _pin(tmp_path, {"m.f(x)": "1", "m.g(y)": "'a'"})
    (tmp_path / "docs" / "STATUS.md").write_text(STATUS, encoding="utf-8")
    (tmp_path / "docs" / "MAP.md").write_text("map v1", encoding="utf-8")
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "base")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(tmp_path, "checkout", "-qb", "my-branch")
    return tmp_path


@pytest.fixture
def fake():
    return Fake()


def run(repo, fake, *argv):
    return cs.main(["--step", "ST1", *argv], root=repo, hooks=fake.hooks())


def only_new_decision(repo):
    files = sorted(p for p in (repo / dd.DECISIONS_REL).glob("D-*.md") if p.name > "D-002-b.md")
    assert len(files) == 1, files
    return files[0]


# ------------------------------------------------------------------------------------- old value after decisions
def test_old_value_is_the_one_in_force_after_the_decisions_not_the_raw_pin(repo, fake, capsys):
    fake.current["m.f(x)"] = "4"  # the pin file says 1, D-001 and D-002 moved it to 3
    assert run(repo, fake) == 0
    new = only_new_decision(repo)
    assert new.name == "D-003-st1-defaults.md"
    assert dp.parse_decision(new) == [("m.f(x)", "3", "4")]
    assert "TODO: why" in new.read_text(encoding="utf-8")
    assert dp.apply_decisions(dd.load_pin(repo), repo / dd.DECISIONS_REL) == {"m.f(x)": "4", "m.g(y)": "'a'"}


def test_new_default_is_pinned_and_documented_as_new(repo, fake):
    fake.current["m.h(z)"] = "7"
    assert run(repo, fake) == 0
    assert dp.parse_decision(only_new_decision(repo)) == [("m.h(z)", dp.ADDED, "7")]
    assert dd.load_pin(repo)["m.h(z)"] == "7"  # the pin file got the key (python -m tests.defaults_pin --add)
    assert dd.load_pin(repo)["m.f(x)"] == "1"  # an existing value is never rewritten


def test_removed_default_is_documented(repo, fake):
    del fake.current["m.g(y)"]
    assert run(repo, fake) == 0
    assert dp.parse_decision(only_new_decision(repo)) == [("m.g(y)", "'a'", "<removed>")]


def test_no_changed_default_writes_no_decision(repo, fake, capsys):
    assert run(repo, fake) == 0
    assert not [p for p in (repo / dd.DECISIONS_REL).glob("D-*.md") if p.name > "D-002-b.md"]
    assert "No pinned default is new, changed or removed" in capsys.readouterr().out


def test_update_in_place_keeps_the_prose_and_the_old_value(repo, fake):
    fake.current["m.f(x)"] = "4"
    run(repo, fake)
    new = only_new_decision(repo)
    new.write_text(new.read_text(encoding="utf-8").replace("TODO: why", "because"), encoding="utf-8")
    fake.current["m.f(x)"] = "5"
    fake.current["m.h(z)"] = "1"
    assert run(repo, fake) == 0
    assert only_new_decision(repo) == new  # no D-004
    assert dp.parse_decision(new) == [("m.f(x)", "3", "5"), ("m.h(z)", dp.ADDED, "1")]
    assert "because" in new.read_text(encoding="utf-8")


def test_decision_flag_must_name_a_record_of_this_pr(repo, fake, capsys):
    fake.current["m.f(x)"] = "4"
    assert run(repo, fake, "--decision", "D-002") == 1
    assert "not added by this PR" in capsys.readouterr().err
    assert run(repo, fake, "--decision", "D-009") == 1


def test_two_records_in_the_pr_need_the_flag(repo, fake, capsys):
    _decision(repo, "D-003-x.md", "- `m.g(y)`: `'a'` -> `'b'`")
    _decision(repo, "D-004-y.md", "- `m.k(q)`: `<new>` -> `1`")
    fake.current |= {"m.g(y)": "'b'", "m.k(q)": "1", "m.f(x)": "9"}
    assert run(repo, fake) == 1
    assert "pass --decision" in capsys.readouterr().err
    assert run(repo, fake, "--decision", "D-004") == 0
    # D-003's key m.g(y) is not repeated; the old value of m.f(x) is the one after D-001 and D-002, not the pin's
    assert dp.parse_decision(repo / dd.DECISIONS_REL / "D-004-y.md") == [("m.f(x)", "3", "9"), ("m.k(q)", "<new>", "1")]


def test_a_later_record_keeps_its_line_for_a_default_an_earlier_record_of_the_pr_also_changed(repo, fake):
    """MG4: D-039 and D-040 (two merged PRs) change one default in turn; updating D-040 must not drop its line."""
    _decision(repo, "D-003-x.md", "- `m.f(x)`: `3` -> `4`")
    _decision(repo, "D-004-y.md", "- `m.f(x)`: `4` -> `5`", "- `m.k(q)`: `<new>` -> `1`")
    fake.current |= {"m.f(x)": "5", "m.k(q)": "1"}
    assert run(repo, fake, "--decision", "D-004") == 0
    assert dp.parse_decision(repo / dd.DECISIONS_REL / "D-004-y.md") == [("m.f(x)", "4", "5"), ("m.k(q)", "<new>", "1")]
    assert check(repo, fake) == []


def test_merge_mode_keeps_every_record_and_lists_them_all(repo, fake, capsys):
    _decision(repo, "D-003-x.md", "- `m.g(y)`: `'a'` -> `'b'`")
    _decision(repo, "D-004-y.md", "- `m.k(q)`: `<new>` -> `1`")
    fake.current |= {"m.g(y)": "'b'", "m.k(q)": "1"}
    before = {p.name: p.read_text(encoding="utf-8") for p in (repo / dd.DECISIONS_REL).glob("D-*.md")}
    assert run(repo, fake, "--merge", "--state", "PR open") == 0
    after = {p.name: p.read_text(encoding="utf-8") for p in (repo / dd.DECISIONS_REL).glob("D-*.md")}
    assert after == before
    section = (repo / cs.OUT_REL).read_text(encoding="utf-8")
    assert "| `m.g(y)` | `'a'` | `'b'` | D-003 |" in section and "| `m.k(q)` | (new default) | `1` | D-004 |" in section
    assert "| ST1 | Staged | PR open |" in (repo / "docs" / "STATUS.md").read_text(encoding="utf-8")


def test_a_number_used_by_a_branch_not_merged_here_is_reported_and_a_merged_one_is_not(repo, fake):
    _git(repo, "checkout", "-q", "-b", "other", "main")
    _decision(repo, "D-003-other-work.md", "- `m.z(a)`: `<new>` -> `1`")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "other")
    _git(repo, "update-ref", "refs/remotes/origin/other", "HEAD")
    _git(repo, "checkout", "-q", "my-branch")
    _decision(repo, "D-003-mine.md", "- `m.f(x)`: `3` -> `4`")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "mine")
    warnings = cs.check_warnings(repo, "origin/main")
    assert len(warnings) == 1 and "D-003 (D-003-mine.md)" in warnings[0] and "origin/other" in warnings[0]
    assert "D-004" in warnings[0]
    _git(repo, "merge", "-q", "--no-edit", "other")   # once the other branch is merged here it is history
    assert cs.check_warnings(repo, "origin/main") == []


# -------------------------------------------------------------------------------------- numbering and clashes
def test_number_taken_on_another_branch_warns(repo, fake, capsys):
    _git(repo, "checkout", "-q", "-b", "other", "main")
    _decision(repo, "D-003-other-work.md", "- `m.z(a)`: `<new>` -> `1`")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "other")
    _git(repo, "update-ref", "refs/remotes/origin/other", "HEAD")
    _git(repo, "checkout", "-q", "my-branch")
    (repo / dd.DECISIONS_REL / "D-003-other-work.md").unlink(missing_ok=True)
    fake.current["m.f(x)"] = "4"
    assert run(repo, fake) == 0
    err = capsys.readouterr().err
    assert "D-003 is also taken" in err and "origin/other" in err and "D-004" in err
    assert only_new_decision(repo).name.startswith("D-003-")  # the number is kept; the owner decides


def test_the_same_file_on_a_remote_branch_is_not_a_clash(repo, fake, capsys):
    fake.current["m.f(x)"] = "4"
    run(repo, fake)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "pr")
    _git(repo, "update-ref", "refs/remotes/origin/my-branch", "HEAD")  # a pushed copy of this very branch
    assert dd.clash_warnings(3, "D-003-st1-defaults.md", dd.remote_decision_files(repo)) == []


def test_number_override_and_gap_free_next(repo, fake):
    assert dd.next_number(repo) == 3 and dd.next_number(repo, 9) == 9
    fake.current["m.f(x)"] = "4"
    assert run(repo, fake, "--slug", "Bad Slug") == 1
    assert run(repo, fake, "--number", "5", "--slug", "my-slug") == 0
    assert only_new_decision(repo).name == "D-005-my-slug.md"


# ------------------------------------------------------------------------------------------------ B1 section
def test_b1_section_lists_each_changed_default_with_its_decision(repo, fake, capsys):
    fake.current |= {"m.f(x)": "4", "m.h(z)": "7"}
    run(repo, fake)
    section = (repo / cs.OUT_REL).read_text(encoding="utf-8")
    assert capsys.readouterr().out.endswith(section + "\n")
    assert "## Behaviour changes (rule B1)" in section
    assert "| `m.f(x)` | `3` | `4` | D-003 |" in section
    assert "| `m.h(z)` | (new default) | `7` | D-003 |" in section
    assert cs.FREE_TEXT_MARK in section  # the free-text line is left to the session


def test_b1_section_names_regenerated_pinned_outputs(repo, fake):
    (repo / "tests" / "pins" / "model_assembly.json").write_text("{}", encoding="utf-8")
    run(repo, fake)
    assert "`tests/pins/model_assembly.json`" in (repo / cs.OUT_REL).read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------------- map and pins
def test_map_and_stale_pytest_pins_are_regenerated_only_when_stale(repo, fake):
    fake.map = "map v2"
    fake.failing = {"tests/model/test_assembly_pins.py"}
    assert run(repo, fake) == 0
    assert (repo / "docs" / "MAP.md").read_text(encoding="utf-8") == "map v2"
    regen = [c for c in fake.calls if c[1] is not None]
    assert regen == [("tests/model/test_assembly_pins.py", "UPDATE_MODEL_PINS")]  # only the stale pin, with its env var


def test_a_pin_test_that_fails_after_regeneration_stops_the_run(repo, fake, capsys):
    class Stuck(Fake):
        def pytest(self, root, target, env_var):
            return 1 if target.endswith("test_baseline_pin.py") else 0

    assert cs.main(["--step", "ST1"], root=repo, hooks=Stuck().hooks()) == 1
    assert "still fails after regenerating" in capsys.readouterr().err


# --------------------------------------------------------------------------------------------------- STATUS
def test_close_updates_the_status_row_from_git_and_the_flags(repo, fake):
    run(repo, fake, "--pr", "59", "--state", "PR open", "--repo", "https://github.com/o/r.git")
    row = next(ln for ln in (repo / "docs" / "STATUS.md").read_text(encoding="utf-8").splitlines() if "ST1" in ln)
    assert "| PR open |" in row and "[#59](https://github.com/o/r/pull/59)" in row
    assert "branch `my-branch`; first" in row


def test_missing_status_row_fails_loudly_and_changes_nothing(repo, fake, capsys):
    fake.current["m.f(x)"] = "4"
    assert cs.main(["--step", "NOPE"], root=repo, hooks=fake.hooks()) == 1
    assert "no row for step 'NOPE'" in capsys.readouterr().err
    assert not [p for p in (repo / dd.DECISIONS_REL).glob("D-*.md") if p.name > "D-002-b.md"]


def test_new_row_from_title_goes_before_the_first_todo_row():
    lines = status_row.update_status(STATUS, "TL1", branch="b", title="New tools", state="in progress").splitlines()
    new = "| TL1 | New tools | in progress |  | branch `b` |"
    assert new in lines and lines.index(new) == lines.index("| T9 | Later | todo | | |") - 1


def test_status_sixty_line_guard_and_state_names():
    long = STATUS + "\n".join(f"line {i}" for i in range(52)) + "\n"  # 7 + 52 = 59 lines
    assert len(long.splitlines()) == 59
    status_row.update_status(long, "ST1", branch=None, state="PR open")  # same length: fine
    with pytest.raises(status_row.ToolError, match="limit is < 60"):
        status_row.update_status(long, "TL1", branch=None, title="x")  # a new row makes 60
    with pytest.raises(status_row.ToolError, match="unknown state"):
        status_row.update_status(STATUS, "ST1", branch=None, state="done")
    status_row.update_status(STATUS, "ST1", branch=None, state="skipped (not needed)")


def test_joined_step_ids_and_branch_note_are_replaced():
    text = STATUS.replace("| ST1 |", "| RW1 / ST1 |").replace("first", "branch `old`; first")
    out = status_row.update_status(text, "ST1", branch="new")
    assert "branch `new`; first" in out and "old" not in out


# -------------------------------------------------------------------------------------------------- --check
def check(repo, fake, **kw):
    return cs.check(repo, "origin/main", fake.hooks(), **kw)


def snapshot(repo):
    return {p.relative_to(repo).as_posix(): p.read_bytes()
            for p in repo.rglob("*") if p.is_file() and ".git" not in p.parts}


def test_check_is_clean_on_a_fresh_pr(repo, fake):
    assert check(repo, fake, step="ST1") == []
    assert cs.main(["--check", "--step", "ST1"], root=repo, hooks=fake.hooks()) == 0


def test_check_fails_on_each_stale_case_and_writes_nothing(repo, fake, capsys):
    before = snapshot(repo)
    fake.map = "map v2"
    assert any("MAP.md is stale" in p for p in check(repo, fake))
    fake.map = "map v1"
    fake.failing = {"tests/jobs/test_baseline_pin.py"}
    assert any("baseline_fixture.json is stale" in p for p in check(repo, fake))
    assert check(repo, fake, skip_pin_tests=True) == []
    fake.failing = set()
    fake.current["m.h(z)"] = "7"  # a default the pin file lacks
    assert any("lacks 1 default" in p for p in check(repo, fake))
    fake.current.pop("m.h(z)")
    fake.current["m.f(x)"] = "4"  # changed, no decision line
    problems = check(repo, fake)
    assert any("missing or wrong for `m.f(x)`" in p for p in problems)
    assert any("no decision line in this PR for `m.f(x)`" in p for p in problems)
    assert any("no row for step 'NOPE'" in p for p in check(repo, fake, step="NOPE"))
    assert cs.main(["--check"], root=repo, hooks=fake.hooks()) == 1
    assert "STALE:" in capsys.readouterr().err
    assert snapshot(repo) == before


def test_check_fails_on_a_wrong_old_value_and_a_wrong_new_value(repo, fake):
    fake.current["m.f(x)"] = "4"
    _decision(repo, "D-003-x.md", "- `m.f(x)`: `1` -> `4`")  # 1 is not in force: D-002 made it 3
    assert any("decision line is wrong" in p for p in check(repo, fake))
    _decision(repo, "D-003-x.md", "- `m.f(x)`: `3` -> `5`")  # the code has 4
    assert any("missing or wrong for `m.f(x)`" in p for p in check(repo, fake))
    _decision(repo, "D-003-x.md", "- `m.f(x)`: `3` -> `4`")
    assert check(repo, fake) == []


def test_check_fails_while_a_todo_is_left_then_passes(repo, fake):
    fake.current["m.f(x)"] = "4"
    run(repo, fake)
    problems = check(repo, fake)
    assert problems and all("TODO" in p for p in problems)
    new = only_new_decision(repo)
    new.write_text(new.read_text(encoding="utf-8").replace("TODO", "Done"), encoding="utf-8")
    assert check(repo, fake) == []


def test_check_fails_on_a_status_file_that_is_too_long(repo, fake):
    (repo / "docs" / "STATUS.md").write_text(STATUS + "x\n" * 60, encoding="utf-8")
    assert any("limit is < 60" in p for p in check(repo, fake))


def test_pin_diff_fails_on_a_todo_in_a_decision_the_pr_adds(repo, monkeypatch, capsys):
    from tests import pin_diff

    monkeypatch.setattr(dp, "ROOT", repo)
    monkeypatch.setattr(dp, "PIN_FILE", repo / dd.PIN_REL)
    _decision(repo, "D-003-x.md", "- `m.k(q)`: `<new>` -> `1`", "TODO: why")
    p = repo / dd.PIN_REL
    data = json.loads(p.read_text(encoding="utf-8"))
    data["defaults"]["m.k(q)"] = "1"
    _pin(repo, data["defaults"])
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "pr")
    assert pin_diff.main("origin/main") == 1
    assert "TODO left: docs/decisions/D-003-x.md" in capsys.readouterr().out


def test_decision_clash_cli_warns_in_github_format_and_never_fails(repo, fake, capsys):
    from tools import decision_clash

    _git(repo, "checkout", "-q", "-b", "other", "main")
    _decision(repo, "D-003-other-work.md", "- `m.z(a)`: `<new>` -> `1`")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "other")
    _git(repo, "update-ref", "refs/remotes/origin/other", "HEAD")
    _git(repo, "checkout", "-q", "my-branch")
    _decision(repo, "D-003-mine.md", "- `m.f(x)`: `3` -> `4`")
    assert decision_clash.main([], root=repo) == 0
    assert capsys.readouterr().out.startswith("::warning title=decision number clash::decision number D-003")
    assert decision_clash.main(["--base", "no-such-ref"], root=repo) == 0
    assert "not checked" in capsys.readouterr().out
