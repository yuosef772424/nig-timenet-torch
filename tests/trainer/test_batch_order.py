"""trainer/groups.py BATCH_ORDERS: "rows" is today's batching; "time_blocks" = exactly one block of consecutive
timestamps per batch, rows in time order, batch order shuffled per epoch."""

import pytest
import torch

from nig_timenet_torch.trainer.groups import BATCH_ORDERS, block_batches, row_batches, time_block_batches


def _times():
    g = torch.Generator().manual_seed(0)
    t = torch.arange(12).repeat_interleave(torch.tensor([3, 5, 2, 4, 3, 3, 5, 2, 4, 3, 3, 3]))
    return t[torch.randperm(len(t), generator=g)]            # rows in no particular order


def test_block_batches_one_batch_per_group_and_shuffled():
    gid = torch.tensor([5, 5, 7, 9, 9, 9])
    plain = block_batches(gid, shuffle=False)
    assert [b.tolist() for b in plain] == [[0, 1], [2], [3, 4, 5]]
    torch.manual_seed(1)
    shuf = block_batches(gid)
    assert sorted(b.tolist() for b in shuf) == [[0, 1], [2], [3, 4, 5]]


@pytest.mark.parametrize("bars", [1, 3])
def test_time_blocks_hold_exactly_whole_blocks_in_time_order(bars):
    t = _times()
    torch.manual_seed(0)
    batches = BATCH_ORDERS.entry("time_blocks").meta["fn"](len(t), 4, t, bars)
    assert sorted(torch.cat(batches).tolist()) == list(range(len(t)))
    for ix in batches:
        tb = t[ix]
        assert (tb[1:] >= tb[:-1]).all()                                     # time order inside the batch
        assert len(torch.unique(tb // bars)) == 1                            # exactly one block
        assert len(ix) == int((t // bars == (tb[0] // bars)).sum())          # the block is whole


def test_time_blocks_epochs_differ_and_the_seed_reproduces():
    t = _times()

    def epoch(seed):
        torch.manual_seed(seed)
        return [b.tolist() for b in time_block_batches(t, 1)]
    assert epoch(1) == epoch(1) and epoch(1) != epoch(2)


def test_rows_order_is_todays_batches_exactly():
    fn = BATCH_ORDERS.entry("rows").meta["fn"]
    torch.manual_seed(3)
    a = fn(103, 16, None, 1)
    torch.manual_seed(3)
    b = list(torch.randperm(103).split(16))
    assert len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    torch.manual_seed(3)
    assert all(torch.equal(x, y) for x, y in zip(row_batches(103, 16), b, strict=True))


def test_time_blocks_without_times_raises_and_unknown_name_raises():
    with pytest.raises(ValueError, match="timestamp id"):
        BATCH_ORDERS.entry("time_blocks").meta["fn"](5, 2, None, 1)
    with pytest.raises(KeyError):
        BATCH_ORDERS.entry("nope")
