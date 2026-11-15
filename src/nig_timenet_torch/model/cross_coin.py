"""PURPOSE: Cross-coin block of NigTimeNetV2 (``cross_coin_layers``): the rows of one batch that share a timestamp
    attend to each other between the trunk and the heads, as a residual that is the identity at initialisation.
TAGS: cross coin, panel, cross asset, multi coin, attention across coins, time group, group channel, residual, zero init
PITFALLS: the group of a row (its timestamp id) arrives as ONE EXTRA INPUT CHANNEL, constant over the window
    (``split_group``; float32 holds ids up to 2^24 exactly). An input without that channel = every row alone (the block
    then only sees the row itself). The block is exact only when a batch holds WHOLE timestamps: training batches and
    evaluation chunks must be cut at group borders (trainer/groups.py). A row's output depends on the other rows of
    its timestamp, by design. Improves on model/panel.py (the TF port, parked): same NIG / class heads and output
    contract as the per-coin model, no (day, pos) grid from the caller, starts as the per-coin model exactly.
"""

import torch
from torch import nn

__all__ = ["CrossCoinBlock", "CrossCoin", "split_group", "group_grid"]


def split_group(x: torch.Tensor, n_features: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(features (B, T, F), group id (B,) int64 or None): the channel after the ``n_features`` feature channels, read
    at the last bar, is the row's group id; an input with exactly ``n_features`` channels has no groups."""
    if x.shape[-1] == n_features:
        return x, None
    if x.shape[-1] != n_features + 1:
        raise ValueError(f"expected {n_features} feature channels (+ 1 group channel), got {x.shape[-1]}")
    return x[..., :n_features], x[:, -1, n_features].round().to(torch.int64)


def group_grid(gid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """(group index, position inside the group, number of groups, largest group) of every row."""
    _, inv = torch.unique(gid, return_inverse=True)
    counts = torch.bincount(inv)
    order = torch.argsort(inv, stable=True)
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.empty_like(inv)
    pos[order] = torch.arange(len(inv), device=inv.device) - starts[inv[order]]
    return inv, pos, int(counts.numel()), int(counts.max())


class CrossCoinBlock(nn.Module):
    """Pre-norm block over the coins of one timestamp: g = g + attn(norm(g)); g = g + ffn(norm(g)). Padding is never
    a key. ``peer_dropout``: in training each real coin is hidden as a KEY with this probability (a coin always sees
    itself), so the block cannot lean on one particular peer or on a fixed universe size."""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1, peer_dropout: float = 0.0):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))
        self.drop = nn.Dropout(dropout)
        self.peer_dropout = float(peer_dropout)

    def forward(self, g: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """g (G, N, d); real (G, N) bool."""
        n = g.shape[1]
        blocked = (~real)[:, None, :].expand(-1, n, -1)                       # (G, query, key): True = not a key
        if self.training and self.peer_dropout > 0:
            hide = torch.rand(real.shape, device=g.device) < self.peer_dropout
            blocked = blocked | hide[:, None, :]
        eye = torch.eye(n, dtype=torch.bool, device=g.device)[None]
        blocked = blocked & ~eye                                              # every row keeps itself as a key
        mask = blocked.repeat_interleave(self.attn.num_heads, dim=0)
        z = self.norm1(g)
        g = g + self.drop(self.attn(z, z, z, attn_mask=mask, need_weights=False)[0])
        return g + self.drop(self.ffn(self.norm2(g)))


class CrossCoin(nn.Module):
    """h (B, d) -> h + gate * (blocks(h grouped by timestamp) - h): ``gate`` (d,) starts at 0, so the model starts as
    the per-coin model and the cross-coin path has to earn its weight. ``last_attention_rows`` = how many rows had at
    least one peer in the last forward (a diagnostic: 0 means the batches do not hold cross-sections)."""

    def __init__(self, d_model: int, layers: int = 1, num_heads: int = 4, dropout: float = 0.1,
                 peer_dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList(CrossCoinBlock(d_model, num_heads, dropout, peer_dropout) for _ in range(layers))
        self.out_norm = nn.LayerNorm(d_model)
        self.gate = nn.Parameter(torch.zeros(d_model))
        self.last_attention_rows = 0

    def forward(self, h: torch.Tensor, gid: torch.Tensor | None) -> torch.Tensor:
        if gid is None:
            gid = torch.arange(len(h), device=h.device)
        inv, pos, n_groups, width = group_grid(gid)
        self.last_attention_rows = int((torch.bincount(inv)[inv] > 1).sum())
        grid = h.new_zeros(n_groups, width, h.shape[-1])
        real = torch.zeros(n_groups, width, dtype=torch.bool, device=h.device)
        grid[inv, pos], real[inv, pos] = h, True
        for block in self.blocks:
            grid = block(grid, real)
        return h + self.gate * (self.out_norm(grid[inv, pos]) - h)
