"""CI check for rule B1 (HF-31): every key added, changed or removed in ``tests/pins/defaults.json`` versus the PR base
appears on a line of a decision file that the PR adds. Without it a PR could edit the pin file instead of writing a
decision. Run in CI as ``python -m tests.pin_diff origin/<base branch>``; exit code 1 lists the undocumented keys.
"""

import json
import subprocess
import sys

from tests import defaults_pin as dp


def undocumented(base_pin: dict[str, str], new_pin: dict[str, str], documented: set[str]) -> list[str]:
    """Keys whose pinned value differs between base and new (added, changed or removed) and are not documented."""
    touched = {k for k in base_pin.keys() | new_pin.keys() if base_pin.get(k) != new_pin.get(k)}
    return sorted(touched - documented)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=dp.ROOT).stdout


def main(base: str) -> int:
    pin_path = dp.PIN_FILE.relative_to(dp.ROOT).as_posix()
    try:
        base_pin = json.loads(_git("show", f"{base}:{pin_path}"))["defaults"]
    except subprocess.CalledProcessError:
        base_pin = {}  # the pin file is new in this PR
    added = _git("diff", "--name-only", "--diff-filter=A", f"{base}...HEAD", "--", "docs/decisions").split()
    documented = {key for rel in added if rel.endswith(".md") and dp.DECISION_NAME.match(rel.rsplit("/", 1)[-1])
                  for key, _, _ in dp.parse_decision(dp.ROOT / rel)}
    missing = undocumented(base_pin, dp.load_pin(), documented)
    for key in missing:
        print(f"pin key without a decision line in this PR: {key}")
    if missing:
        print(f"{len(missing)} key(s): write docs/decisions/D-nnn-*.md in this PR (rule B1)")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
