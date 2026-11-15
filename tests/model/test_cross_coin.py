"""model/cross_coin.py and its plumbing: the cross-coin block is the identity at init, depends on a row's own
timestamp only, the builder option keeps the default model's keys, and the group helpers of the trainer hold whole
timestamps (trainer/groups.py)."""

import numpy as np
import pytest
import torch

from nig_timenet_torch.model.builder import build_nig_timenet_v2
from nig_timenet_torch.model.cross_coin import CrossCoin, group_grid, split_group
from nig_timenet_torch.trainer.groups import group_batches, group_chunks, row_groups

KW = dict(d_model=16, num_layers=1, num_heads=4, num_kv_heads=2, head_hidden=16, dropout=0.0, encoder="gru")
GID = torch.tensor([5, 5, 5, 7, 7, 9, 5, 7, 7, 7])


def _open(c: CrossCoin) -> CrossCoin:
    with torch.no_grad():
        c.gate.fill_(1.0)
    return c


def test_identity_at_init_then_only_the_own_timestamp_matters():
    torch.manual_seed(0)
    c, h = CrossCoin(16, 1, 4, 0.0).eval(), torch.randn(10, 16)
    assert torch.equal(c(h, GID), h) and c.last_attention_rows == 9          # the lone row of group 9 has no peer
    out = _open(c)(h, GID)
    perm = torch.randperm(10)
    torch.testing.assert_close(c(h[perm], GID[perm]), out[perm], atol=1e-5, rtol=0)  # row order does not matter
    moved = h.clone()
    moved[0] += torch.randn(16)                                               # a coin of timestamp 5 changes
    out2 = c(moved, GID)
    assert (out2[1] - out[1]).abs().max() > 1e-3                              # its peers move
    assert torch.equal(out2[GID != 5], out[GID != 5])                         # other timestamps do not
    torch.testing.assert_close(c(h, None), c(h, torch.arange(10)))            # no group channel = every row alone


def test_group_grid_positions_are_unique_inside_a_group():
    inv, pos, n_groups, width = group_grid(GID)
    assert (n_groups, width) == (3, 5)
    assert len({(int(a), int(b)) for a, b in zip(inv, pos, strict=True)}) == len(GID)


def test_split_group_reads_the_last_channel():
    x = torch.randn(4, 8, 3)
    gid = torch.tensor([3.0, 3.0, 8.0, 12345678.0])
    xg = torch.cat([x, gid[:, None, None].expand(-1, 8, 1)], dim=-1)
    feats, got = split_group(xg, 3)
    assert torch.equal(feats, x) and got.tolist() == [3, 3, 8, 12345678]
    assert split_group(x, 3)[1] is None
    with pytest.raises(ValueError, match="group channel"):
        split_group(torch.randn(4, 8, 5), 3)


def test_builder_option_default_keeps_the_keys_and_the_model_starts_as_the_per_coin_model():
    torch.manual_seed(0)
    base = build_nig_timenet_v2(32, 3, **KW)
    torch.manual_seed(0)
    cross = build_nig_timenet_v2(32, 3, **KW, cross_coin_layers=1)
    extra = set(cross.state_dict()) - set(base.state_dict())
    assert extra and all(k.startswith("cross_coin.") for k in extra)
    cross.load_state_dict(base.state_dict(), strict=False)
    x = torch.randn(6, 32, 3)
    xg = torch.cat([x, torch.tensor([1.0, 1, 1, 2, 2, 2])[:, None, None].expand(-1, 32, 1)], dim=-1)
    base.eval(), cross.eval()
    with torch.no_grad():
        want, got, plain = base(x), cross(xg), base(xg)
    for k in want:
        torch.testing.assert_close(got[k], want[k])       # gate = 0: the same function
        torch.testing.assert_close(plain[k], want[k])     # a model without the block ignores the channel
    _open(cross.cross_coin)
    with torch.no_grad():
        assert not torch.allclose(cross(xg)["y_close"], want["y_close"])
    with pytest.raises(NotImplementedError, match="cross_coin_layers"):
        build_nig_timenet_v2(32, 3, **KW, cross_coin_layers=1, nig_head="scalar_joint")


def test_group_batches_and_chunks_hold_whole_groups():
    torch.manual_seed(0)
    gid = torch.tensor(np.repeat(np.arange(40), 7))                     # 40 timestamps x 7 coins, time-major
    batches = group_batches(gid[torch.randperm(len(gid))], 20)          # any row order
    rows = torch.cat(batches)
    assert sorted(rows.tolist()) == list(range(len(gid)))               # every row exactly once
    assert all(len(b) >= 20 for b in batches[:-1])
    for c in group_chunks(gid, 20):
        ids = gid[c]
        assert len(ids) >= 20 or c.stop == len(gid)
        assert not set(ids.tolist()) & (set(gid[:c.start].tolist()) | set(gid[c.stop:].tolist()))
    x = torch.cat([torch.randn(len(gid), 4, 3), gid[:, None, None].expand(-1, 4, 1).float()], dim=-1)
    assert torch.equal(row_groups(x), gid)
