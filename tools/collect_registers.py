"""Collect friction and suggestion rows from merged PR descriptions into docs/friction_log.csv and docs/suggestions.csv.

PURPOSE:  Parse the `## Friction` and `## Suggestions` sections of merged PR bodies (format: the PR template), validate
  every row and append a good PR's rows to the two CSV registers once; a bad or missing section is reported.
TAGS:     tools, registers, friction, suggestions, csv, pull, request, collector, idempotent
PITFALLS: Idempotence lives in docs/registers_collected.json (PR number -> counts), not in the CSVs (their columns are
  unchanged on purpose); a PR is collected all-or-nothing; PRs merged before the state file's `since` are ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from tools._git import ToolError

ROOT = Path(__file__).resolve().parent.parent
STATE_REL = "docs/registers_collected.json"
FRICTION_REL = "docs/friction_log.csv"
SUGGESTIONS_REL = "docs/suggestions.csv"
#: CSV columns in file order; the PR sections use exactly these names as `field: value` keys
FRICTION_COLS = ("date", "step", "what_slowed_or_broke", "where", "minutes_lost", "root_cause", "fix_idea")
SUGGESTION_COLS = ("date", "step", "kind", "where", "what", "minutes_lost", "proposed_fix", "status")
#: a field the session may leave out: its default (``date`` = the PR's merge date)
OPTIONAL = {"date": None, "status": "open"}
KINDS = ("suggestion", "unclear", "stale")
SECTIONS = {"Friction": FRICTION_COLS, "Suggestions": SUGGESTION_COLS}
HEADING = re.compile(r"^##\s+(.*?)\s*$")
FIELD = re.compile(r"^(?P<bullet>-\s+)?(?P<key>[a-z_]+):[ \t]*(?P<value>.*)$")
NONE_ROW = re.compile(r"^-\s+none\b(?::[ \t]*(?P<why>.*))?$", re.I)
COMMENT = re.compile(r"<!--.*?-->", re.S)


@dataclass
class Parsed:
    """The rows of one PR: ``rows[section]`` is None when the section is missing or has errors."""

    rows: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _section_bodies(body: str) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    current = None
    for line in COMMENT.sub("", body or "").splitlines():
        m = HEADING.match(line)
        if m:
            current = m.group(1) if m.group(1) in SECTIONS else None
            if current:
                out.setdefault(current, [])
        elif current:
            out[current].append(line.rstrip())
    return {k: "\n".join(v) for k, v in out.items()}


def _records(section: str, text: str, errors: list[str]) -> list[dict[str, str]]:
    """Records of one section: a `- key: value` line starts one, indented `key: value` lines add fields; `- none` (with
    an optional reason) is an explicit empty section."""
    records: list[dict[str, str]] = []
    explicit_none = False
    seen_errors = len(errors)
    for raw in text.splitlines():
        if not raw.strip():
            continue
        if NONE_ROW.match(raw.strip()) and not raw.startswith((" ", "\t")):
            explicit_none = True
            continue
        m = FIELD.match(raw.strip())
        if not m or (m["bullet"] is None and not raw.startswith((" ", "\t"))):
            errors.append(f"{section}: cannot read the line {raw.strip()[:80]!r} (expected `- field: value`)")
            continue
        if m["bullet"]:
            records.append({})
        elif not records:
            errors.append(f"{section}: field {m['key']!r} before any `- ` bullet")
            continue
        if m["key"] in records[-1]:
            errors.append(f"{section}: row {len(records)} repeats the field {m['key']!r}")
        records[-1][m["key"]] = m["value"].strip()
    if explicit_none and records:
        errors.append(f"{section}: `- none` together with rows")
    if not records and not explicit_none and len(errors) == seen_errors:
        errors.append(f"{section}: the section is empty (write rows, or `- none: <why>`)")
    return records


def _validate(section: str, i: int, rec: dict[str, str], date: str) -> tuple[dict[str, str], list[str]]:
    cols = SECTIONS[section]
    errs = [f"{section} row {i}: unknown field {k!r} (columns: {', '.join(cols)})" for k in rec if k not in cols]
    row = {c: rec.get(c, "").strip() for c in cols}
    for c in cols:
        if not row[c]:
            if c in OPTIONAL:
                row[c] = OPTIONAL[c] or date
            else:
                errs.append(f"{section} row {i}: field {c!r} is missing or empty")
    if row["minutes_lost"] and not re.fullmatch(r"\d+", row["minutes_lost"]):
        errs.append(f"{section} row {i}: minutes_lost must be a whole number, got {row['minutes_lost']!r}")
    if section == "Suggestions" and row["kind"] and row["kind"] not in KINDS:
        errs.append(f"{section} row {i}: kind must be one of {', '.join(KINDS)}, got {row['kind']!r}")
    if row["date"] and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["date"]):
        errs.append(f"{section} row {i}: date must be YYYY-MM-DD, got {row['date']!r}")
    return row, errs


def parse_body(body: str, merged_at: str) -> Parsed:
    """Rows of both sections, validated; any problem goes to ``errors`` and the PR is not collected."""
    out = Parsed()
    bodies = _section_bodies(body)
    date = (merged_at or "")[:10]
    for section in SECTIONS:
        if section not in bodies:
            out.errors.append(f"the `## {section}` section is missing")
            continue
        rows = []
        for i, rec in enumerate(_records(section, bodies[section], out.errors), start=1):
            row, errs = _validate(section, i, rec, date)
            out.errors += errs
            rows.append(row)
        out.rows[section] = rows
    return out


# ----------------------------------------------------------------------------------------------- state and CSV
def load_state(root: Path) -> dict:
    path = root / STATE_REL
    if not path.exists():
        raise ToolError(f"{STATE_REL} is missing (it records `since` and the collected PRs)")
    return json.loads(path.read_text(encoding="utf-8"))


def _append(path: Path, cols: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    header = next(csv.reader([text.splitlines()[0]]))
    if tuple(header) != cols:
        raise ToolError(f"{path.name}: header {header} differs from the collector's columns {list(cols)}")
    with path.open("a", encoding="utf-8", newline="") as fh:
        if not text.endswith("\n"):
            fh.write("\n")
        w = csv.writer(fh, lineterminator="\n")
        for r in rows:
            w.writerow([r[c] for c in cols])


@dataclass
class Result:
    collected: dict[int, tuple[int, int]] = field(default_factory=dict)
    skipped: list[int] = field(default_factory=list)  # already collected
    bad: dict[int, list[str]] = field(default_factory=dict)


def collect(root: Path, prs: list[dict], *, dry_run: bool = False, skip: dict[int, str] | None = None) -> Result:
    """Append the rows of every merged PR not in the state file, once. ``prs`` = ``[{number, body, merged_at}]``."""
    state = load_state(root)
    done = state.setdefault("collected", {})
    res = Result()
    friction: list[dict[str, str]] = []
    suggestions: list[dict[str, str]] = []
    for pr in sorted((p for p in prs if p.get("merged_at")), key=lambda p: (p["merged_at"], p["number"])):
        n = int(pr["number"])
        if str(n) in done:
            res.skipped.append(n)
            continue
        if pr["merged_at"] <= state.get("since", ""):
            continue
        if skip and n in skip:
            done[str(n)] = {"merged_at": pr["merged_at"], "skipped": skip[n]}
            continue
        parsed = parse_body(pr.get("body") or "", pr["merged_at"])
        if parsed.errors:
            res.bad[n] = parsed.errors
            continue
        f_rows, s_rows = parsed.rows["Friction"], parsed.rows["Suggestions"]
        friction += f_rows
        suggestions += s_rows
        done[str(n)] = {"merged_at": pr["merged_at"], "friction": len(f_rows), "suggestions": len(s_rows)}
        res.collected[n] = (len(f_rows), len(s_rows))
    if not dry_run and (friction or suggestions or skip):
        _append(root / FRICTION_REL, FRICTION_COLS, friction)
        _append(root / SUGGESTIONS_REL, SUGGESTION_COLS, suggestions)
        (root / STATE_REL).write_text(json.dumps(state, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return res


# ------------------------------------------------------------------------------------------------------ input
def prs_from_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(p, dict) and {"number", "body", "merged_at"} <= p.keys()
                                             for p in data):
        raise ToolError(f"{path}: expected a JSON list of {{number, body, merged_at}} objects")
    return data


def prs_from_api(repo: str, token: str, since: str,
                 urlopen: Callable = urllib.request.urlopen, max_pages: int = 10) -> list[dict]:
    """Merged PRs updated since ``since`` from the GitHub REST API (most recently updated first, paged)."""
    out: list[dict] = []
    for page in range(1, max_pages + 1):
        url = (f"https://api.github.com/repos/{repo}/pulls?state=closed&sort=updated&direction=desc"
               f"&per_page=100&page={page}")
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}",
                                                   "Accept": "application/vnd.github+json",
                                                   "X-GitHub-Api-Version": "2022-11-28"})
        with urlopen(req, timeout=30) as resp:
            batch = json.loads(resp.read().decode("utf-8"))
        if not batch:
            break
        out += [{"number": p["number"], "body": p.get("body") or "", "merged_at": p["merged_at"]}
                for p in batch if p.get("merged_at")]
        if all((p.get("updated_at") or "") <= since for p in batch):
            break
    return out


# --------------------------------------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tools.collect_registers", description=__doc__.split("\n\n")[0])
    p.add_argument("--from-json", type=Path, help="JSON list of {number, body, merged_at} (no network)")
    p.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"), help="owner/name for the API (default "
                   "$GITHUB_REPOSITORY); the API needs $GITHUB_TOKEN")
    p.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    p.add_argument("--skip", action="append", default=[], metavar="N:REASON",
                   help="mark PR N as collected without rows (a body that cannot be fixed); repeatable")
    return p


def main(argv: list[str] | None = None, root: Path = ROOT, urlopen: Callable = urllib.request.urlopen,
         environ: dict | None = None) -> int:
    args = build_parser().parse_args(argv)
    environ = os.environ if environ is None else environ
    try:
        if args.from_json:
            prs = prs_from_json(args.from_json)
        else:
            token = environ.get("GITHUB_TOKEN")
            if not token or not args.repo:
                raise ToolError("no input: pass --from-json FILE, or set GITHUB_TOKEN and --repo / GITHUB_REPOSITORY")
            prs = prs_from_api(args.repo, token, load_state(root).get("since", ""), urlopen)
        skip = {}
        for item in args.skip:
            num, _, why = item.partition(":")
            if not num.isdigit() or not why.strip():
                raise ToolError(f"--skip {item!r}: write N:reason")
            skip[int(num)] = why.strip()
        res = collect(root, prs, dry_run=args.dry_run, skip=skip)
    except ToolError as exc:
        print(f"collect_registers: {exc}", file=sys.stderr)
        return 1
    for n, (f, s) in sorted(res.collected.items()):
        print(f"PR #{n}: {f} friction row(s), {s} suggestion row(s)" + (" (dry run)" if args.dry_run else ""))
    if res.skipped:
        print(f"already collected: {', '.join('#' + str(n) for n in res.skipped)}")
    for n, errs in sorted(res.bad.items()):
        print(f"PR #{n} NOT collected:", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
    return 1 if res.bad else 0


if __name__ == "__main__":
    sys.exit(main())
