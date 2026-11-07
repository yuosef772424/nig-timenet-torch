"""Ratchet-count helper shared by the growth and hardcoded-name guards (HF-32)."""

import json

from tests import scan

RATCHETS = json.loads((scan.ROOT / "tests" / "ratchets.json").read_text(encoding="utf-8"))
COUNTS = RATCHETS["counts"]


def ratchet(name: str, found: list[str]) -> None:
    pinned = COUNTS[name]
    listing = "\n  ".join(found) or "(none)"
    assert len(found) <= pinned, f"{name} grew: {len(found)} > pinned {pinned} (ratchet only shrinks):\n  {listing}"
    assert len(found) >= pinned, (f"{name} shrank to {len(found)} (pinned {pinned}): good - lower it in "
                                  f"tests/ratchets.json (and docs/bottlenecks.csv) so it cannot grow back:\n"
                                  f"  {listing}")
