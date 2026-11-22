"""CI check: the PR description holds valid `## Friction` and `## Suggestions` sections.

PURPOSE:  Parse a PR description with the collector's own parser (tools.collect_registers.parse_body) and fail with the
  list of problems, so a bad section is fixed before the merge and not found by the weekly collection.
TAGS:     tools, ci, pull, request, friction, suggestions, sections, validate
PITFALLS: The body comes in the environment variable PR_BODY (never interpolated into a shell command: a PR description
  is untrusted text); a missing or empty variable is an error. Only meant for pull_request events.
"""

from __future__ import annotations

import os
import sys

from tools.collect_registers import parse_body


def problems(body: str) -> list[str]:
    """The collector's parse errors for ``body`` (empty = both sections valid)."""
    if not (body or "").strip():
        return ["the PR description is empty: it needs `## Friction` and `## Suggestions` (see the PR template)"]
    return parse_body(body, "").errors


def main(env: dict[str, str] | None = None) -> int:
    body = (os.environ if env is None else env).get("PR_BODY", "")
    errs = problems(body)
    for e in errs:
        print(f"PR description: {e}", file=sys.stderr)
    print("check_pr_sections: " + (f"{len(errs)} problem(s)" if errs else "ok"))
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
