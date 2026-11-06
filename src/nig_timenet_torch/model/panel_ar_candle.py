"""The PANEL version of the autoregressive candle transformer: many coins, one shared model, one rollout.

A batch is a grid of aligned windows: ``x`` (B, N, C, F) - B window positions, N coins on the SAME
timestamps, C context candles. The trunk is the single-coin ``ArCandleTransformer`` recipe (input
embedding + learned positions + causal temporal blocks), plus a per-coin embedding and position-wise
CROSS-COIN attention (``CrossAssetBlock`` over the coin axis at every timestamp, so a coin may look at
the other coins at time t but never at any coin's future). Heads predict every coin's next candle
(mu, log_sigma, dir_logit), so the rollout is panel-wide: each step appends every coin's own predicted
candle, like N text models generating in lockstep with a shared vocabulary.
"""

import torch
from torch import nn

from nig_timenet_torch.model.ar_candle_transformer import ENCODERS
from nig_timenet_torch.model.layers import TransformerBlock
from nig_timenet_torch.model.panel import CrossAssetBlock

__all__ = ["PanelArCandleTransformer", "build_panel_ar_candle", "rollout"]


class PanelArCandleTransformer(nn.Module):
    """Causal temporal attention per coin + position-wise cross-coin attention; ``context_len`` and
    ``horizon_len`` are free parameters exactly as in the single-coin model."""

    def __init__(self, n_features: int, n_coins: int, context_len: int = 32, horizon_len: int = 16,
                 d_model: int = 64, num_layers: int = 2, num_heads: int = 4, num_kv_heads: int = 2,
                 cross_layers: int = 1, max_rel_pos: int = 64, dropout: float = 0.1,
                 attn_dropout: float = 0.0, encoder: str = "transformer"):
        super().__init__()
        if n_coins < 1:
            raise ValueError(f"n_coins must be >= 1, got {n_coins}")
        if encoder not in ENCODERS:
            raise ValueError(f"unknown encoder {encoder!r}; known: {sorted(ENCODERS)}")
        self.encoder_kind = encoder
        self.context_len, self.horizon_len, self.n_coins = int(context_len), int(horizon_len), int(n_coins)
        if encoder == "transformer":
            self.inp = nn.Linear(n_features, d_model)
            self.pos = nn.Parameter(torch.zeros(1, 1, context_len + horizon_len, d_model))
            self.blocks = nn.ModuleList([
                TransformerBlock(d_model, num_heads, num_kv_heads, max_rel_pos, causal=True,
                                 dropout=dropout, attn_dropout=attn_dropout) for _ in range(num_layers)])
        else:
            self.rnn = nn.GRU(n_features, d_model, num_layers=num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0)
        self.coin = nn.Parameter(torch.zeros(1, n_coins, 1, d_model))
        self.cross = nn.ModuleList([
            CrossAssetBlock(d_model=d_model, num_heads=num_heads, dropout=dropout,
                            attn_dropout=attn_dropout) for _ in range(cross_layers)])
        self.mu = nn.Linear(d_model, n_features)
        self.log_sigma = nn.Linear(d_model, n_features)
        self.dir = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """x (B, N, T, F) -> per-coin next-candle predictions; ``mask`` (B, N) True = a real coin (a
        padded slot never acts as an attention key and its outputs are zeroed)."""
        b, n, t, _ = x.shape
        if self.encoder_kind == "gru":
            flat = x.reshape(b * n, t, -1)
            h, _ = self.rnn(flat)  # (B*N, T, d): causal by construction, no positions needed
            h = h.reshape(b, n, t, -1) + self.coin[:, :n]
        else:
            if t > self.pos.shape[2]:
                pad = self.pos[:, :, -1:].expand(1, 1, t - self.pos.shape[2], self.pos.shape[3])
                pos = torch.cat([self.pos, pad], dim=2)  # a rollout longer than the trained horizon
            else:
                pos = self.pos
            h = self.inp(x) + pos[:, :, :t] + self.coin[:, :n]
            h = h.reshape(b * n, t, -1)
            for block in self.blocks:
                h = block(h)
            h = h.reshape(b, n, t, -1)
        if mask is None:
            mask = torch.ones(b, n, dtype=torch.bool, device=x.device)
        key_mask = mask[:, None, :].expand(b, t, n).reshape(b * t, n)
        h = h.permute(0, 2, 1, 3).reshape(b * t, n, -1)
        for block in self.cross:
            h = block(h, key_mask)
        h = h.reshape(b, t, n, -1).permute(0, 2, 1, 3)
        keep = mask[:, :, None, None].to(h.dtype)  # a padded slot's outputs are zeroed, bias included
        return {"mu": self.mu(h) * keep, "log_sigma": self.log_sigma(h) * keep,
                "dir_logit": self.dir(h).squeeze(-1) * mask[:, :, None].to(h.dtype), "mask": mask}


def build_panel_ar_candle(n_features: int, n_coins: int, context_len: int = 32, horizon_len: int = 16,
                          **config) -> PanelArCandleTransformer:
    """The panel model from plain kwargs (the notebook passes its settings dict through)."""
    return PanelArCandleTransformer(n_features=n_features, n_coins=n_coins, context_len=context_len,
                                    horizon_len=horizon_len, **config)


def rollout(model: PanelArCandleTransformer, context: torch.Tensor, horizon_len: int | None = None,
            mask: torch.Tensor | None = None) -> dict:
    """Panel-wide autoregressive generation: each step appends EVERY coin's own predicted candle
    (detached), like N text models generating in lockstep. Returns (B, N, H, ·) stacks."""
    h = model.horizon_len if horizon_len is None else int(horizon_len)
    if h < 1:
        raise ValueError(f"horizon_len must be >= 1, got {h}")
    seq = context.detach()
    mus, log_sigmas, dir_logits = [], [], []
    for _ in range(h):
        with torch.no_grad():
            out = model(seq, mask)
        mus.append(out["mu"][:, :, -1])
        log_sigmas.append(out["log_sigma"][:, :, -1])
        dir_logits.append(out["dir_logit"][:, :, -1])
        seq = torch.cat([seq, out["mu"][:, :, -1].detach()[:, :, None]], dim=2)
    return {"mu": torch.stack(mus, dim=2), "log_sigma": torch.stack(log_sigmas, dim=2),
            "dir_logit": torch.stack(dir_logits, dim=2)}
