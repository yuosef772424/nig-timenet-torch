"""PURPOSE: Batches and evaluation chunks that hold WHOLE groups (timestamps), for a model whose rows interact inside
    a batch (model/cross_coin.py); the group id of a row is the last input channel at the last bar.
TAGS: group batches, time groups, cross section, batch order, evaluation chunks, cross coin, group channel
PITFALLS: a batch is one or more whole groups and has AT LEAST ``batch_size`` rows unless the data ends (so batches
    are uneven and a group larger than ``batch_size`` is one batch); the shuffle permutes GROUPS with torch's global
    RNG (reseeded per epoch by the trainer). ``group_chunks`` needs rows that are contiguous per group (time-major
    order: workflow ``SplitSettings.cross_section``); a group split over two chunks would see half its peers.
"""

import torch

__all__ = ["row_groups", "group_batches", "group_chunks"]


def row_groups(x: torch.Tensor) -> torch.Tensor:
    """The group id (int64) of every row of ``x`` (n, T, F + 1): its last channel at the last bar."""
    return x[:, -1, -1].round().to(torch.int64)


def group_batches(gid: torch.Tensor, batch_size: int, shuffle: bool = True) -> list[torch.Tensor]:
    """Index tensors over the rows: each batch = whole groups, groups in a shuffled order, closed as soon as it holds
    ``batch_size`` rows."""
    uniq, inv = torch.unique(gid, return_inverse=True)
    order = torch.argsort(inv, stable=True)
    counts = torch.bincount(inv, minlength=len(uniq))
    rows_of = order.split(counts.tolist())
    batches, cur, n = [], [], 0
    for g in (torch.randperm(len(uniq)) if shuffle else torch.arange(len(uniq))).tolist():
        cur.append(rows_of[g])
        n += len(rows_of[g])
        if n >= batch_size:
            batches.append(torch.cat(cur))
            cur, n = [], 0
    if cur:
        batches.append(torch.cat(cur))
    return batches


def group_chunks(gid: torch.Tensor, size: int) -> list[slice]:
    """Consecutive slices of about ``size`` rows that end at a group border (rows contiguous per group)."""
    n = len(gid)
    border = torch.ones(n + 1, dtype=torch.bool)
    border[1:n] = gid[1:] != gid[:-1]
    ends = torch.nonzero(border[1:]).flatten() + 1                  # row counts at which a group ends
    out, start = [], 0
    while start < n:
        i = int(torch.searchsorted(ends, torch.tensor(start + size)))
        stop = int(ends[min(i, len(ends) - 1)])
        out.append(slice(start, stop))
        start = stop
    return out
