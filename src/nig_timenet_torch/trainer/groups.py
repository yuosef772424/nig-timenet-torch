"""PURPOSE: Batches and evaluation chunks that hold WHOLE groups (timestamps), for a model whose rows interact inside
    a batch (model/cross_coin.py), and the BATCH_ORDERS registry ("rows" | "time_blocks") of the flat model.
TAGS: group batches, time groups, cross section, batch order, time blocks, evaluation chunks, cross coin, group channel
PITFALLS: a batch is one or more whole groups and has AT LEAST ``batch_size`` rows unless the data ends (so batches
    are uneven and a group larger than ``batch_size`` is one batch); the shuffle permutes GROUPS with torch's global
    RNG (reseeded per epoch by the trainer). ``group_chunks`` needs rows that are contiguous per group (time-major
    order: workflow ``SplitSettings.cross_section``); a group split over two chunks would see half its peers.
    ``BATCH_ORDERS`` (flat model, ``TrainSettings.batch_order``, an experiment arm, off by default): "rows" = random
    rows per batch (today); "time_blocks" = each batch is EXACTLY ONE block of ``time_block_bars`` consecutive
    timestamps (all coins of them; blocks are never merged, ``batch_size`` is NOT used for training batches in this
    mode, so batches are uneven), rows inside in time order, the ORDER OF THE BATCHES shuffled each epoch.
    For this model the order inside a batch does not change the gradient (the loss is a mean over rows); what
    changes is the COMPOSITION: one market moment per batch gives correlated, noisier gradients, and the BatchNorm
    layers (``level_norm: batch``) then see per-period statistics in training but moving statistics in eval.
    No extra input channel (unlike the cross-section path); evaluation is unchanged.
"""

import torch

from nig_timenet_torch.core.registry import Contract, Registry

__all__ = ["row_groups", "group_batches", "group_chunks", "row_batches", "block_batches",
           "time_block_batches", "BATCH_ORDERS"]


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


def row_batches(n: int, batch_size: int, shuffle: bool = True) -> list[torch.Tensor]:
    """Mini-batch index tensors over n rows; the shuffle draws from torch's global RNG, so it derives from the
    seed_everything(seed) call made before the Trainer is built."""
    order = torch.randperm(n) if shuffle else torch.arange(n)
    return list(order.split(batch_size))


def block_batches(gid: torch.Tensor, shuffle: bool = True) -> list[torch.Tensor]:
    """One batch per group (never merged up to a batch size): the row indices of each group, the groups in a shuffled
    order (torch's global RNG) or ascending id."""
    uniq, inv = torch.unique(gid, return_inverse=True)
    order = torch.argsort(inv, stable=True)
    rows_of = order.split(torch.bincount(inv, minlength=len(uniq)).tolist())
    return [rows_of[g] for g in (torch.randperm(len(uniq)) if shuffle else torch.arange(len(uniq))).tolist()]


def time_block_batches(times: torch.Tensor, bars: int, shuffle: bool = True) -> list[torch.Tensor]:
    """Batches that are EXACTLY one block each: all rows of ``bars`` consecutive timestamp ids (``times`` = the id of
    every row, consecutive integers; ``bars`` = 1: one moment), the rows inside sorted by time (stable), the order
    of the blocks shuffled."""
    times = torch.as_tensor(times, dtype=torch.int64)
    return [ix[torch.argsort(times[ix], stable=True)] for ix in block_batches(times // bars, shuffle)]


def _rows_order(n: int, batch_size: int, times, bars: int) -> list[torch.Tensor]:
    return row_batches(n, batch_size)


def _blocks_order(n: int, batch_size: int, times, bars: int) -> list[torch.Tensor]:
    if times is None or len(times) != n:
        raise ValueError("batch_order with time blocks needs the timestamp id of every training row "
                         f"(Trainer.row_times); got {None if times is None else len(times)} for {n} rows")
    return time_block_batches(times, bars)


#: ``fn(n_rows, batch_size, row_times, time_block_bars) -> index tensors``; ``needs_times``: reads ``row_times``
BATCH_ORDERS = Registry("batch_order", Contract("batch_order", required_meta=("fn", "needs_times")))
BATCH_ORDERS.register("rows", _rows_order, version=1, fn=_rows_order, needs_times=False)
BATCH_ORDERS.register("time_blocks", _blocks_order, version=1, fn=_blocks_order, needs_times=True)
