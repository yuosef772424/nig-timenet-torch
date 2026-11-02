"""Helpers to read the dict literal of docs/DATA_ARTIFACT_CONTRACT.md (shared by the data-contract tests)."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
DATASET_DIR = FIXTURES / "tiny_dataset.dataset"


def contract() -> dict:
    text = (ROOT / "docs" / "DATA_ARTIFACT_CONTRACT.md").read_text(encoding="utf-8")
    block = re.search(r"```python\n(CONTRACT = .*?)\n```", text, re.S).group(1)
    return ast.literal_eval(block.split("=", 1)[1])


def ndim(meaning: str) -> int:
    """Rank declared at the start of a shape_meaning, e.g. '(N, T=16, F=18): ...' -> 3, '(): ...' -> 0."""
    inside = re.match(r"\(([^)]*)\)", meaning).group(1).strip().rstrip(",")
    return 0 if not inside else len(inside.split(","))
