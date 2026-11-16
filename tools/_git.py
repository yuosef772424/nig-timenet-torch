"""Thin git wrapper shared by the tools in this folder.

PURPOSE:  One place that runs ``git -C <root> ...`` and returns text, so every tool fails the same way and the tests
  can point it at a temporary repository.
TAGS:     tools, git, subprocess, helper
PITFALLS: ``git`` raises ToolError with git's stderr on a non-zero exit unless ``check=False`` (then returns None).
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class ToolError(Exception):
    """A user-facing failure: the CLI prints it and exits 1 (never a traceback for an expected problem)."""


def git(root: Path, *args: str, check: bool = True) -> str | None:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if r.returncode != 0:
        if check:
            raise ToolError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
        return None
    return r.stdout


def merge_base(root: Path, base: str) -> str:
    """The commit the PR forked from ``base`` (a branch or ref); ``base`` itself when git finds none."""
    out = git(root, "merge-base", "HEAD", base, check=False)
    if out and out.strip():
        return out.strip()
    if git(root, "rev-parse", "--verify", "-q", f"{base}^{{commit}}", check=False) is None:
        raise ToolError(f"base {base!r} is not a known revision (git fetch origin, or pass --base)")
    return base
