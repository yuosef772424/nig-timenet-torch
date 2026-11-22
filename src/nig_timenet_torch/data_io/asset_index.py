"""PURPOSE: The coin of each artifact row, read from ``asset_bounds`` only: the asset position (``asset_ids``)
    and the coin name (``row_coins``) of any rows ``idx``, with no full-dataset array.
TAGS: asset bounds, coin of each row, asset ids, row coins, coin channel, searchsorted, artifact rows
PITFALLS: ``asset_ids`` is the position in ``dataset["asset_bounds"]`` AS STORED (not sorted by start); a row outside
    every bound gets -1 and the name None (the artifact tiles its rows, so this only guards a hand-made dataset);
    the cost is O(len(idx) log coins), so call it with the rows you need, not with the whole dataset.
"""

import numpy as np

__all__ = ["asset_ids", "row_coins"]


def asset_ids(dataset: dict, idx) -> np.ndarray:
    """The asset index of each row ``idx`` (its position in ``asset_bounds``), -1 for a row in no bound."""
    bounds = dataset["asset_bounds"]
    idx = np.asarray(idx, dtype=np.int64)
    if not bounds:
        return np.full(idx.shape, -1, dtype=np.int64)
    start = np.array([b["start"] for b in bounds], dtype=np.int64)
    end = np.array([b["end"] for b in bounds], dtype=np.int64)
    order = np.argsort(start, kind="stable")
    pos = np.searchsorted(start[order], idx, side="right") - 1
    hit = order[np.maximum(pos, 0)]
    return np.where((pos >= 0) & (idx < end[hit]), hit, -1).astype(np.int64)


def row_coins(dataset: dict, idx) -> np.ndarray:
    """The coin name of each row ``idx`` (object array; None for a row in no bound)."""
    names = np.empty(len(dataset["asset_bounds"]) + 1, dtype=object)   # the last slot (index -1) stays None
    names[:-1] = [b["name"] for b in dataset["asset_bounds"]]
    return names[asset_ids(dataset, idx)]
