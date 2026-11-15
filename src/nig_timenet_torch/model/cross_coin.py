"""PURPOSE: The multi-coin parts of NigTimeNetV2: ``cross_input`` (each coin's window relative to its timestamp's
    market, bar by bar) and ``cross_coin_layers`` (rows of one timestamp exchange information after the trunk).
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

__all__ = ["CrossCoinBlock", "CrossSectionPool", "CROSS_COIN_KINDS", "CrossCoin", "RelativeInputs",
           "CROSS_INPUTS", "split_group", "group_grid"]


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


class RelativeInputs(nn.Module):
    """(B, T, F) -> (B, T, 4F), bar by bar inside each timestamp group: [own, own - group mean, (own - group mean) /
    group sd, group mean]. Windows of one timestamp cover the same bars, so the group mean at bar t is the market at
    that bar: the encoder then reads how each coin moved AGAINST the others over the whole window (the differences
    between coins), next to the market itself. A row without peers gets zeros in the two relative blocks. No
    weights."""

    width_mult = 4

    def forward(self, x: torch.Tensor, gid: torch.Tensor | None) -> torch.Tensor:
        if gid is None:
            return torch.cat([x, torch.zeros_like(x), torch.zeros_like(x), x], dim=-1)
        inv, _, n_groups, _ = group_grid(gid)
        n = torch.bincount(inv, minlength=n_groups).to(x.dtype)[:, None, None]
        mean = x.new_zeros(n_groups, *x.shape[1:]).index_add_(0, inv, x) / n
        dev = x - mean[inv]
        var = x.new_zeros(n_groups, *x.shape[1:]).index_add_(0, inv, dev * dev) / n
        return torch.cat([x, dev, dev / (var[inv] + 1e-6).sqrt(), mean[inv]], dim=-1)


#: cross_input -> module class (None = the per-coin input as it is)
CROSS_INPUTS = {"none": None, "relative": RelativeInputs}


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


class CrossSectionPool(nn.Module):
    """The cheapest cross-coin step (kind "mean"): every coin gets [own, timestamp mean, own - mean, timestamp sd] of
    the trunk through one hidden layer. No attention: what a coin is RELATIVE to its timestamp is given, not learned
    (ridge on the simple features gains 0.04 rank IC on ``low`` from exactly this: TRANSFORMER_HEAD_RESULTS.md
    section 5)."""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1, peer_dropout: float = 0.0):
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(4 * d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(2 * d_model, d_model))

    def forward(self, g: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        w = real[..., None].to(g.dtype)
        n = w.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (g * w).sum(dim=1, keepdim=True) / n
        sd = (((g - mean) ** 2 * w).sum(dim=1, keepdim=True) / n + 1e-6).sqrt()
        return self.ffn(torch.cat([g, mean.expand_as(g), g - mean, sd.expand_as(g)], dim=-1))


#: cross_coin_kind -> block class (d_model, num_heads, dropout, peer_dropout)
CROSS_COIN_KINDS = {"attention": CrossCoinBlock, "mean": CrossSectionPool}


class CrossCoin(nn.Module):
    """h (B, d) -> h + gate * (blocks(h grouped by timestamp) - h): ``gate`` (d,) starts at 0, so the model starts as
    the per-coin model and the cross-coin path has to earn its weight. ``last_attention_rows`` = how many rows had at
    least one peer in the last forward (a diagnostic: 0 means the batches do not hold cross-sections)."""

    def __init__(self, d_model: int, layers: int = 1, num_heads: int = 4, dropout: float = 0.1,
                 peer_dropout: float = 0.0, kind: str = "attention"):
        super().__init__()
        if kind not in CROSS_COIN_KINDS:
            raise ValueError(f"cross_coin_kind must be one of {tuple(CROSS_COIN_KINDS)}, not {kind!r}")
        self.blocks = nn.ModuleList(CROSS_COIN_KINDS[kind](d_model, num_heads, dropout, peer_dropout)
                                    for _ in range(layers))
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
