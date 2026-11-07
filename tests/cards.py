"""Module cards (rule B7): parser, validity, allowlist and git helpers shared by tests/test_cards.py and the map
builder tests/build_map.py. Nothing here is a test.

Card = fields at the start of a line in a module docstring (the docstring's first line may be a title):
    PURPOSE:  <= 240 characters (joined), TAGS: >= 3 comma-separated words, PITFALLS: text or "none",
    SOURCE:   provenance of a ported module (kept when the base version had one).
A field continues on the following indented lines (so cards respect the 120-column ruff limit, no E501 ignores).
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG_REL = "src/nig_timenet_torch"
ALLOWLIST_REL = "tests/cards_allowlist.txt"
FIELDS = ("PURPOSE", "TAGS", "PITFALLS", "SOURCE")
FIELD_RE = re.compile(r"^(PURPOSE|TAGS|PITFALLS|SOURCE)[ \t]*:[ \t]*(.*)$")
PURPOSE_MAX = 240
TAGS_MIN = 3
TITLE_MAX = 100
#: a provenance line outside the SOURCE field (older ports wrote "source: ..."): losing it is an error
PROVENANCE_RE = re.compile(r"^(SOURCE|source)[ \t]*:", re.M)


def parse_card(docstring: str | None) -> dict[str, str]:
    """{field: joined text} of the card fields found in ``docstring`` (``{}`` for None). Continuation lines are
    the indented non-empty lines right after a field line; a blank line or a new field ends the field."""
    fields: dict[str, str] = {}
    current = None
    for line in (docstring or "").splitlines():
        m = FIELD_RE.match(line)
        if m:
            current = m.group(1)
            fields[current] = m.group(2).strip()
        elif current and line.strip() and line[0] in " \t":
            fields[current] = (fields[current] + " " + line.strip()).strip()
        else:
            current = None
    return fields


def title_of(docstring: str | None) -> str:
    """First docstring line when it is not a card field, else ""."""
    if not docstring:
        return ""
    first = docstring.strip().splitlines()[0].strip()
    return "" if FIELD_RE.match(first) else first


def card_problems(fields: dict[str, str]) -> list[str]:
    """Why ``fields`` is not a valid card (empty list = valid)."""
    out = [f"{f} missing or empty" for f in ("PURPOSE", "TAGS", "PITFALLS") if not fields.get(f)]
    if len(fields.get("PURPOSE", "")) > PURPOSE_MAX:
        out.append(f"PURPOSE has {len(fields['PURPOSE'])} characters (max {PURPOSE_MAX})")
    tags = [t for t in (x.strip() for x in fields.get("TAGS", "").split(",")) if t]
    if fields.get("TAGS") and len(tags) < TAGS_MIN:
        out.append(f"TAGS has {len(tags)} items (min {TAGS_MIN})")
    return out


def module_docstring(text: str) -> str | None:
    return ast.get_docstring(ast.parse(text))


def module_files(pkg_dir: Path | None = None) -> list[str]:
    """Paths (relative to the package, posix) of every non-``__init__`` module, sorted."""
    pkg = pkg_dir or ROOT / PKG_REL
    return sorted(p.relative_to(pkg).as_posix() for p in pkg.rglob("*.py") if p.name != "__init__.py")


def card_of_file(path: Path) -> dict[str, str]:
    return parse_card(module_docstring(path.read_text(encoding="utf-8")))


def read_allowlist(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


# ------------------------------------------------------------------------------------------------------- git
def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def resolve_base(root: Path = ROOT, env: dict | None = None) -> str | None:
    """Base revision of the PR: env ``CARD_BASE`` (CI, when it names a known commit), else
    ``git merge-base HEAD origin/main``, else None."""
    env = os.environ if env is None else env
    given = (env.get("CARD_BASE") or "").strip()
    if given and _git(root, "rev-parse", "--verify", "-q", f"{given}^{{commit}}").returncode == 0:
        return given  # an unknown rev (a force-pushed `before` sha) falls through to the merge-base
    r = _git(root, "merge-base", "HEAD", "origin/main")
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def changed_modules(root: Path, base: str, pkg_rel: str = PKG_REL) -> list[str]:
    """Non-``__init__`` modules (relative to the package) added / modified / renamed since ``base``, including
    uncommitted and untracked ones."""
    names: set[str] = set()
    for args in (("diff", "--name-only", "--diff-filter=AMR", f"{base}...HEAD", "--", pkg_rel),
                 ("diff", "--name-only", "--diff-filter=AMR", "HEAD", "--", pkg_rel),
                 ("ls-files", "--others", "--exclude-standard", "--", pkg_rel)):
        r = _git(root, *args)
        if r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
        names |= {ln.strip() for ln in r.stdout.splitlines() if ln.strip()}
    prefix = pkg_rel + "/"
    return sorted(n[len(prefix):] for n in names if n.startswith(prefix) and n.endswith(".py")
                  and not n.endswith("__init__.py"))


def base_text(root: Path, base: str, rel_path: str) -> str | None:
    """Content of ``rel_path`` (repo-relative) at ``base``; None when it did not exist there."""
    r = _git(root, "show", f"{base}:{rel_path}")
    return r.stdout if r.returncode == 0 else None


def base_allowlist(root: Path, base: str) -> list[str] | None:
    """The allowlist at ``base``; None when the base has none (the PR that introduces the list)."""
    text = base_text(root, base, ALLOWLIST_REL)
    return None if text is None else [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]


def provenance_dropped(old_text: str | None, new_text: str) -> bool:
    """True when the base version of a module had a provenance line and the new one has none."""
    if old_text is None:
        return False
    return bool(PROVENANCE_RE.search(old_text)) and not PROVENANCE_RE.search(new_text)
