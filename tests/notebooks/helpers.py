"""Notebook-test helpers (HF-32)."""

MAX_CELL_LINES = 22


def code_cells(nb):
    return [(i, c, "".join(c["source"])) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
