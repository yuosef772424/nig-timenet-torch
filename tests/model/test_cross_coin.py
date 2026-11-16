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


@pytest.mark.parametrize("kind", ["attention", "mean"])
def test_every_kind_is_the_identity_at_init_and_local_to_the_timestamp(kind):
    torch.manual_seed(0)
    c, h = CrossCoin(16, 1, 4, 0.0, kind=kind).eval(), torch.randn(10, 16)
    assert torch.equal(c(h, GID), h)
    out = _open(c)(h, GID)
    moved = h.clone()
    moved[0] += torch.randn(16)
    out2 = c(moved, GID)
    assert (out2[1] - out[1]).abs().max() > 1e-4 and torch.equal(out2[GID != 5], out[GID != 5])
    with pytest.raises(ValueError, match="cross_coin_kind"):
        CrossCoin(16, kind="median")


def test_relative_inputs_are_the_differences_between_the_coins_of_a_timestamp():
    from nig_timenet_torch.model.cross_coin import RelativeInputs

    torch.manual_seed(0)
    x, gid = torch.randn(6, 5, 3), torch.tensor([4, 4, 4, 9, 9, 2])
    out = RelativeInputs()(x, gid)
    assert out.shape == (6, 5, 12)
    own, rel, z, market = out.split(3, dim=-1)
    assert torch.equal(own, x)
    torch.testing.assert_close(market[:3], x[:3].mean(0, keepdim=True).expand(3, -1, -1))
    torch.testing.assert_close(rel[:3], x[:3] - x[:3].mean(0, keepdim=True))
    torch.testing.assert_close(rel[:3].sum(0), torch.zeros(5, 3), atol=1e-5, rtol=0)  # differences sum to zero
    assert rel[5].abs().max() == 0 and z[5].abs().max() == 0                        # a coin alone has no difference
    alone = RelativeInputs()(x, None)
    assert torch.equal(alone[..., 9:], x) and alone[..., 3:9].abs().max() == 0


def test_cross_input_model_reads_its_peers_and_keeps_the_contract():
    torch.manual_seed(0)
    m = build_nig_timenet_v2(32, 3, **KW, cross_input="relative").eval()
    base = build_nig_timenet_v2(32, 3, **KW).eval()
    assert m.instance_norm.gamma.shape == (12,) and base.instance_norm.gamma.shape == (3,)
    x = torch.randn(6, 32, 3)
    xg = torch.cat([x, torch.tensor([1.0, 1, 1, 2, 2, 2])[:, None, None].expand(-1, 32, 1)], dim=-1)
    with torch.no_grad():
        out = m(xg)
        moved = xg.clone()
        moved[0, :, :3] += torch.randn(32, 3)                # coin 0 of timestamp 1 changes
        out2 = m(moved)
    assert set(out) == set(base(x)) and all(torch.isfinite(v).all() for v in out.values())
    assert (out2["y_close"][1] - out["y_close"][1]).abs() > 1e-6          # its peer's prediction moves
    torch.testing.assert_close(out2["y_close"][3:], out["y_close"][3:])   # the other timestamp does not
    assert m._gid is None
    with pytest.raises(ValueError, match="cross_input"):
        build_nig_timenet_v2(32, 3, **KW, cross_input="ranks")


def test_cross_center_makes_the_close_mean_relative_to_its_timestamp_and_is_off_by_default():
    """D-032: with the switch the close mean of every timestamp sums to 0 (the high / low means and a row without the
    timestamp channel are untouched); without it the model is the same graph as before."""
    from nig_timenet_torch.model.architectures import ANTI_MEMORIZATION_CONFIG, ModelSettings, build_model
    from nig_timenet_torch.model.cross_coin import center_groups
    from nig_timenet_torch.model.heads import head_spec

    def model(**over):
        torch.manual_seed(0)
        return build_model(ModelSettings("nig_timenet_v2", {**ANTI_MEMORIZATION_CONFIG, "encoder": "gru", **over}),
                           16, 3).eval()

    off, on = model(), model(cross_center=True)
    assert not off.cross_center and [k for k, _ in off.named_parameters()] == [k for k, _ in on.named_parameters()]
    x = 0.02 * torch.randn(12, 16, 3)
    gid = torch.tensor([5.0] * 4 + [9.0] * 8)
    xg = torch.cat([x, gid[:, None, None].expand(-1, 16, 1)], dim=-1)
    mu = {t: head_spec("nig_regression", t).output_keys["mu"] for t in ("close", "high")}
    with torch.no_grad():
        a, b, alone = off(xg), on(xg), on(x)
    assert torch.allclose(b[mu["close"]], center_groups(a[mu["close"]], gid.long()), atol=1e-6)
    assert abs(float(b[mu["close"]][:4].sum())) < 1e-5 and abs(float(b[mu["close"]][4:].sum())) < 1e-5
    assert torch.equal(b[mu["high"]], a[mu["high"]])
    assert torch.allclose(alone[mu["close"]], off(x)[mu["close"]])          # no timestamp channel: nothing to centre
    v = torch.arange(6.0).reshape(6, 1)
    assert torch.allclose(center_groups(v, torch.tensor([1, 1, 1, 4, 4, 4])).flatten(),
                          torch.tensor([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]))
