"""Draft or update the decision record of a PR from the defaults that differ from the base (rule B1).

PURPOSE:  For every default new, changed or removed since the PR base, compute the line `key: old -> new` (old = value
  in force after the existing decisions), number the record, warn on clashes, write or update the decision file.
TAGS:     tools, decisions, defaults, pin, base, number, clash, draft, rule, b1
PITFALLS: "old" is the pin file with the decisions applied in order, minus the record being drafted; never the raw pin
  value. A drafted record carries `TODO` placeholders that tests/pin_diff.py and `close_step --check` reject.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from tests import defaults_pin as dp
from tools._git import ToolError, git

PIN_REL = "tests/pins/defaults.json"
DECISIONS_REL = "docs/decisions"
REMOVED = "<removed>"
Line = tuple[str, str, str]  # key, old, new


def load_pin(root: Path) -> dict[str, str]:
    import json

    path = root / PIN_REL
    return json.loads(path.read_text(encoding="utf-8"))["defaults"] if path.exists() else {}


def decisions_dir(root: Path) -> Path:
    return root / DECISIONS_REL


def decision_number(name: str) -> int:
    m = dp.DECISION_NAME.match(name)
    if not m:
        raise ToolError(f"{name!r} is not a decision file name (D-nnn-short-name.md)")
    return int(m.group(1))


def base_effective(root: Path, mb: str) -> dict[str, str]:
    """The defaults in force at the base commit: its pin file with its decisions applied."""
    import json

    pin_text = git(root, "show", f"{mb}:{PIN_REL}", check=False)
    pin = json.loads(pin_text)["defaults"] if pin_text else {}
    names = (git(root, "ls-tree", "--name-only", mb, f"{DECISIONS_REL}/", check=False) or "").split()
    with tempfile.TemporaryDirectory() as tmp:
        for rel in names:
            name = rel.rsplit("/", 1)[-1]
            if dp.DECISION_NAME.match(name):
                (Path(tmp) / name).write_text(git(root, "show", f"{mb}:{rel}") or "", encoding="utf-8")
        return dp.apply_decisions(pin, Path(tmp))


def pr_added_decisions(root: Path, mb: str) -> list[Path]:
    """Decision files this PR adds (committed, staged or untracked), sorted by name."""
    names = set((git(root, "diff", "--name-only", "--diff-filter=A", mb, "--", DECISIONS_REL) or "").split())
    names |= set((git(root, "ls-files", "--others", "--exclude-standard", "--", DECISIONS_REL) or "").split())
    return sorted(root / n for n in names if dp.DECISION_NAME.match(n.rsplit("/", 1)[-1]))


def pick_target(root: Path, added: list[Path], decision: str | None) -> Path | None:
    """The record to update in place: the one named by ``--decision``, else the only one this PR adds, else None."""
    if decision:
        num = int(re.sub(r"^D-", "", decision))
        folder = decisions_dir(root)
        hits = [p for p in dp.decision_files(folder) if decision_number(p.name) == num]
        if not hits:
            raise ToolError(f"no decision file D-{num:03d} in {DECISIONS_REL}")
        if hits[0] not in added:
            raise ToolError(f"{hits[0].name} is not added by this PR: a decision on the base is history, "
                            "write a new record instead")
        return hits[0]
    if len(added) > 1:
        raise ToolError("this PR adds several decision records (" + ", ".join(p.name for p in added)
                        + "): pass --decision D-NNN to say which one the tool updates")
    return added[0] if added else None


def draft_lines(current: dict[str, str], base_eff: dict[str, str], other_eff: dict[str, str],
                other_pr_keys: set[str]) -> list[Line]:
    """One ``(key, old, new)`` per default that differs from the base and that the other records of the PR do not
    already bring to its current value (a merge PR can hold two records that change the same default in turn: the
    later one keeps its line, old = the value after the earlier one). ``old`` is the value in force without the
    record being drafted (``other_eff``), ``<new>`` for a default that did not exist at the base, ``<removed>`` as
    ``new`` for one that is gone."""
    out: list[Line] = []
    for key in sorted(set(current) | set(base_eff) | set(other_eff)):
        if key in other_pr_keys and current.get(key) == other_eff.get(key):
            continue  # another record of the PR already brings this default to its current value
        cur = current.get(key)
        if key not in base_eff:
            if cur is not None:
                out.append((key, dp.ADDED, cur))
            continue
        old = other_eff.get(key)
        if cur != old:
            out.append((key, old if old is not None else dp.ADDED, cur if cur is not None else REMOVED))
    return out


def render_line(line: Line) -> str:
    key, old, new = line
    return f"- `{key}`: `{old}` -> `{new}`"


def keys_of(paths: list[Path]) -> set[str]:
    return {key for p in paths for key, _, _ in dp.parse_decision(p)}


def draft_text(number: int, title: str, today: str, lines: list[Line]) -> str:
    body = "\n".join(render_line(ln) for ln in lines)
    return (f"# D-{number:03d} — {title}\n\n"
            f"Date: {today}\n"
            "Owner decision: TODO: cite the owner's instruction (step prompt or message) that decided this\n"
            "Why: TODO: why (the incident or measurement that justifies each line below)\n\n"
            f"## What changes behaviour\n- TODO: why, in words, per line below\n\n{body}\n")


def rewrite_lines(text: str, lines: list[Line]) -> str:
    """``text`` with its decision lines replaced by ``lines`` (prose kept); the block sits where the first old line was,
    or at the end when the file has none."""
    out: list[str] = []
    placed = False
    block = [render_line(ln) for ln in lines]
    for raw in text.split("\n"):
        if dp.DECISION_LINE.match(raw):
            if not placed:
                out += block
                placed = True
            continue
        out.append(raw)
    if not placed:
        while out and out[-1] == "":
            out.pop()
        out += ["", *block, ""]
    return "\n".join(out)


# --------------------------------------------------------------------------------------------------- numbering
def local_numbers(root: Path) -> list[int]:
    return [decision_number(p.name) for p in dp.decision_files(decisions_dir(root))]


def remote_decision_files(root: Path) -> dict[str, list[str]]:
    """{remote branch: decision file names} over every ``origin/*`` ref (git ls-tree: no API, no network)."""
    refs = (git(root, "for-each-ref", "--format=%(refname:short)", "refs/remotes/origin", check=False) or "").split()
    out: dict[str, list[str]] = {}
    for ref in refs:
        if ref.endswith("/HEAD") or ref == "origin":
            continue
        names = (git(root, "ls-tree", "--name-only", ref, f"{DECISIONS_REL}/", check=False) or "").split()
        out[ref] = [n.rsplit("/", 1)[-1] for n in names if dp.DECISION_NAME.match(n.rsplit("/", 1)[-1])]
    return out


@dataclass
class Numbering:
    number: int
    warnings: list[str] = field(default_factory=list)


def clash_warnings(number: int, name: str, remote: dict[str, list[str]]) -> list[str]:
    """Warnings for a record ``name`` with ``number`` when another branch has a DIFFERENT file with that number."""
    taken = {ref: [n for n in names if decision_number(n) == number and n != name] for ref, names in remote.items()}
    taken = {ref: names for ref, names in taken.items() if names}
    if not taken:
        return []
    highest = max([number] + [decision_number(n) for names in remote.values() for n in names])
    where = "; ".join(f"{ref}: {', '.join(names)}" for ref, names in sorted(taken.items()))
    return [f"WARNING: decision number D-{number:03d} is also taken on another branch ({where}). "
            f"First number free everywhere: D-{highest + 1:03d}; renumber with --number {highest + 1} (the file "
            "and the heading), or accept that the second PR to merge renumbers."]


def open_clashes(root: Path, added: list[Path], remote: dict[str, list[str]] | None = None) -> list[str]:
    """Warnings for each decision this PR adds whose NUMBER another branch uses for a different file, counting only
    branches whose work is not already in HEAD (a merged or included branch is history, not a clash). Two open PRs
    that both took the next free number are found here before the second one merges."""
    remote = remote if remote is not None else remote_decision_files(root)
    out = []
    for path in added:
        num, name = decision_number(path.name), path.name
        hits = sorted((ref, other) for ref, names in remote.items() for other in names
                      if decision_number(other) == num and other != name and not _in_head(root, ref))
        if hits:
            free = max([num] + [decision_number(n) for names in remote.values() for n in names]
                       + local_numbers(root)) + 1
            where = "; ".join(f"{ref}: {other}" for ref, other in hits)
            out.append(f"decision number D-{num:03d} ({name}) is also used on a branch not merged here ({where}); "
                       f"the PR that merges second renumbers (first number free everywhere: D-{free:03d})")
    return out


def _in_head(root: Path, ref: str) -> bool:
    return git(root, "merge-base", "--is-ancestor", ref, "HEAD", check=False) is not None


def next_number(root: Path, override: int | None = None) -> int:
    return override if override is not None else max(local_numbers(root) or [0]) + 1
