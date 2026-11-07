"""Rule B7: module cards (PURPOSE / TAGS / PITFALLS) and the generated navigation map docs/MAP.md.

The allowlist ``tests/cards_allowlist.txt`` holds the card-less modules of the day: it only shrinks, a module a PR
changes may not be on it. Base of the PR: env ``CARD_BASE``, else the merge-base with ``origin/main``; with neither,
the two git tests skip locally and FAIL when ``CARD_REQUIRE_BASE=1`` (CI).
"""

import os
import subprocess

import pytest

from tests import build_map
from tests.cards import (
    ALLOWLIST_REL,
    PKG_REL,
    ROOT,
    base_allowlist,
    base_text,
    card_of_file,
    card_problems,
    changed_modules,
    module_files,
    parse_card,
    provenance_dropped,
    read_allowlist,
    resolve_base,
)

PKG_DIR = ROOT / PKG_REL
GOOD = '''"""Title

PURPOSE:  What it is for,
          wrapped on a second line.
TAGS:     one, two, three
PITFALLS: none
SOURCE:   upstream x @ abc
"""'''


def _allow():
    return read_allowlist(ROOT / ALLOWLIST_REL)


def _base_or_skip():
    base = resolve_base()
    if base is None:
        if os.environ.get("CARD_REQUIRE_BASE") == "1":
            pytest.fail("CARD_REQUIRE_BASE=1 but no base: set CARD_BASE or fetch origin/main (fetch-depth: 0)")
        pytest.skip("no base branch (CARD_BASE / origin/main): run in CI")
    return base


# ------------------------------------------------------------------------------------------ the repository
def test_cards_are_valid():
    allow = set(_allow())
    bad = {m: card_problems(card_of_file(PKG_DIR / m)) for m in module_files() if m not in allow}
    bad = {m: p for m, p in bad.items() if p}
    assert not bad, f"modules without a valid card (add PURPOSE / TAGS / PITFALLS, rule B7): {bad}"


def test_allowlist_is_current():
    files = set(module_files())
    allow = _allow()
    assert allow == sorted(allow) and len(allow) == len(set(allow)), "allowlist must be sorted, no duplicates"
    missing = [m for m in allow if m not in files]
    carded = [m for m in allow if m in files and not card_problems(card_of_file(PKG_DIR / m))]
    assert not missing, f"allowlist entries that are not modules (remove them): {missing}"
    assert not carded, f"modules that now have a card: remove them from {ALLOWLIST_REL}: {carded}"


def test_allowlist_never_grows():
    base = _base_or_skip()
    old = base_allowlist(ROOT, base)
    if old is None:
        pytest.skip("the base has no allowlist yet (the PR that introduces it)")
    assert set(_allow()) <= set(old), f"modules added to the allowlist: {sorted(set(_allow()) - set(old))}"


def test_touched_modules_have_cards():
    base = _base_or_skip()
    allow = set(_allow())
    touched = changed_modules(ROOT, base)
    bad = [m for m in touched if m in allow]
    assert not bad, f"changed modules need a card (rule B7); write it and remove them from the allowlist: {bad}"
    dropped = [m for m in touched if provenance_dropped(base_text(ROOT, base, f"{PKG_REL}/{m}"),
                                                       (PKG_DIR / m).read_text(encoding="utf-8"))]
    assert not dropped, f"provenance (SOURCE:) line dropped: {dropped}"


def test_map_is_fresh():
    path = ROOT / build_map.MAP_REL
    assert path.exists() and path.read_text(encoding="utf-8") == build_map.build(), (
        f"{build_map.MAP_REL} is stale: run `python -m tests.build_map` and commit it")


def test_map_lists_every_registry_and_module():
    text = (ROOT / build_map.MAP_REL).read_text(encoding="utf-8")
    assert len(build_map.registry_definitions(PKG_DIR)) == 10
    assert all(f"`{m}`" in text for m in module_files())


# ------------------------------------------------------------------------------------------ the parser
def test_parse_valid_card_joins_continuation_lines():
    f = parse_card(GOOD.strip('"'))
    assert f["PURPOSE"] == "What it is for, wrapped on a second line." and f["SOURCE"] == "upstream x @ abc"
    assert card_problems(f) == []


def test_parse_card_on_first_line():
    assert card_problems(parse_card("PURPOSE: a\nTAGS: a, b, c\nPITFALLS: none\n\nprose")) == []


@pytest.mark.parametrize("drop", ["PURPOSE", "TAGS", "PITFALLS"])
def test_missing_field_is_invalid(drop):
    lines = [ln for ln in GOOD.strip('"').splitlines() if not ln.startswith(drop)]
    assert any(drop in p for p in card_problems(parse_card("\n".join(lines))))


def test_long_purpose_and_few_tags_are_invalid():
    long = "PURPOSE: " + "x" * 241 + "\nTAGS: a, b\nPITFALLS: none"
    problems = card_problems(parse_card(long))
    assert any("PURPOSE has 241" in p for p in problems) and any("TAGS has 2" in p for p in problems)


def test_provenance_dropped_scan():
    old = '"""T\n\nSOURCE: upstream @ abc\n"""\n'
    assert provenance_dropped(old, '"""T\n\nPURPOSE: x\n"""\n')
    assert not provenance_dropped(old, old) and not provenance_dropped(None, "x")
    assert provenance_dropped('"""T\nsource: crypto-signal-prediction @ abc\n"""', '"""T"""')


# ------------------------------------------------------------------------- fake git histories (rule E3: fast)
def _git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.email=a@b", "-c", "user.name=t", *args], check=True,
                   capture_output=True)


@pytest.fixture
def repo(tmp_path):
    pkg = tmp_path / PKG_REL
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "old.py").write_text('"""Old module, no card."""\n')
    (pkg / "carded.py").write_text(GOOD + "\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / ALLOWLIST_REL).write_text("old.py\n")
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "base")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(tmp_path, "checkout", "-qb", "pr")
    return tmp_path


def test_base_from_env_then_merge_base_then_none(repo, tmp_path_factory):
    assert resolve_base(repo, {"CARD_BASE": "HEAD"}) == "HEAD"
    assert resolve_base(repo, {"CARD_BASE": "deadbeef"}) == resolve_base(repo, {})  # unknown sha -> merge-base
    assert resolve_base(repo, {"CARD_BASE": ""}) == resolve_base(repo, {})
    assert resolve_base(repo, {}) is not None
    assert resolve_base(tmp_path_factory.mktemp("nogit"), {}) is None


def test_untouched_pr_changes_nothing(repo):
    assert changed_modules(repo, resolve_base(repo, {})) == []


def test_touch_check_fires_for_a_changed_allowlisted_module(repo):
    pkg = repo / PKG_REL
    (pkg / "old.py").write_text('"""Old module, no card."""\nX = 1\n')
    _git(repo, "commit", "-qam", "edit")
    touched = changed_modules(repo, resolve_base(repo, {}))
    assert touched == ["old.py"]
    assert set(touched) & set(read_allowlist(repo / ALLOWLIST_REL))  # the test above would fail on this


def test_touch_check_sees_uncommitted_and_new_files_not_init(repo):
    pkg = repo / PKG_REL
    (pkg / "carded.py").write_text(GOOD + "\nY = 2\n")
    (pkg / "new.py").write_text('"""New."""\n')
    (pkg / "__init__.py").write_text("Z = 3\n")
    assert changed_modules(repo, resolve_base(repo, {})) == ["carded.py", "new.py"]


def test_base_allowlist_and_provenance_from_history(repo):
    base = resolve_base(repo, {})
    assert base_allowlist(repo, base) == ["old.py"]
    assert base_allowlist(repo, "HEAD~0") == ["old.py"]
    (repo / ALLOWLIST_REL).write_text("old.py\nnew.py\n")
    assert set(read_allowlist(repo / ALLOWLIST_REL)) - set(base_allowlist(repo, base)) == {"new.py"}
    assert "SOURCE:" in base_text(repo, base, f"{PKG_REL}/carded.py")
    assert base_text(repo, base, f"{PKG_REL}/absent.py") is None


def test_base_without_allowlist_is_none(repo):
    (repo / ALLOWLIST_REL).unlink()
    _git(repo, "commit", "-qam", "rm")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    assert base_allowlist(repo, "HEAD") is None
