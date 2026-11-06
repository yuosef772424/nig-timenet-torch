"""The autoregressive candle transformer: a small decoder-only GPT over candles.

Every position t predicts the features of candle t+1 (``mu``), the uncertainty of that prediction
(``log_sigma``) and the direction of the next close move (``dir_logit``). With a causal mask the
prediction at t is a function of candles <= t only, so generating works exactly like text: roll out
the model's OWN predicted candle as the next input, horizon_len times, to forecast a whole stretch of
future candles candle by candle. ``context_len`` and ``horizon_len`` are parameters — any pair works.
"""

import torch
from torch import nn

from nig_timenet_torch.model.layers import TransformerBlock

__all__ = ["ENCODERS", "ArCandleTransformer", "rollout"]

#: the pluggable temporal trunks: the causal GPT-style transformer (default) or the GRU stack
ENCODERS = ("transformer", "gru")


class ArCandleTransformer(nn.Module):
    """A pluggable temporal encoder + three heads. ``encoder`` picks the trunk:

    * ``"transformer"`` (default): linear input embedding + learned absolute positions + causal
      TransformerBlocks (relative attention, GQA) — the GPT-style trunk;
    * ``"gru"``: stacked GRU layers (causal by construction) — the ANTI_MEMORIZATION preset's trunk on
      the TF side, and the cheap option for small data.

    ``forward(x)`` with x (B, T, F) returns ``{"mu": (B, T, F), "log_sigma": (B, T, F),
    "dir_logit": (B, T)}`` — each position's prediction of the NEXT candle (language-model style).
    """

    def __init__(self, n_features: int, context_len: int = 32, horizon_len: int = 16, d_model: int = 64,
                 num_layers: int = 2, num_heads: int = 4, num_kv_heads: int = 2, max_rel_pos: int = 64,
                 dropout: float = 0.1, attn_dropout: float = 0.0, encoder: str = "transformer"):
        super().__init__()
        if context_len < 1 or horizon_len < 1:
            raise ValueError(f"context_len and horizon_len must be >= 1, got {context_len}/{horizon_len}")
        if encoder not in ENCODERS:
            raise ValueError(f"unknown encoder {encoder!r}; known: {sorted(ENCODERS)}")
        self.encoder_kind = encoder
        self.context_len, self.horizon_len = int(context_len), int(horizon_len)
        if encoder == "transformer":
            self.inp = nn.Linear(n_features, d_model)
            self.pos = nn.Parameter(torch.zeros(1, context_len + horizon_len, d_model))
            self.blocks = nn.ModuleList([
                TransformerBlock(d_model, num_heads, num_kv_heads, max_rel_pos, causal=True,
                                 dropout=dropout, attn_dropout=attn_dropout) for _ in range(num_layers)])
        else:
            self.rnn = nn.GRU(n_features, d_model, num_layers=num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0)
        self.mu = nn.Linear(d_model, n_features)
        self.log_sigma = nn.Linear(d_model, n_features)
        self.dir = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        t = x.shape[1]
        if self.encoder_kind == "gru":
            h, _ = self.rnn(x)  # (B, T, d): causal by construction, no positions needed
        else:
            pos = self.pos
            if t > pos.shape[1]:
                # a rollout longer than the trained horizon (a free eval horizon): extend the position
                # table by repeating its last entry - the relative-attention blocks stay exact
                pad = pos[:, -1:].expand(1, t - pos.shape[1], pos.shape[2])
                pos = torch.cat([pos, pad], dim=1)
            h = self.inp(x) + pos[:, :t]
            for block in self.blocks:
                h = block(h)
        return {"mu": self.mu(h), "log_sigma": self.log_sigma(h),
                "dir_logit": self.dir(h).squeeze(-1)}


def rollout(model: ArCandleTransformer, context: torch.Tensor, horizon_len: int | None = None) -> dict:
    """Autoregressive generation, exactly like a text model: predict the next candle, append it (detached,
    gradient-free) as the next input, repeat. Returns ``{"mu", "log_sigma", "dir_logit"}`` stacked over
    the horizon steps (B, H, F) / (B, H) plus ``inputs`` (B, H, F): the candles actually fed back."""
    h = model.horizon_len if horizon_len is None else int(horizon_len)
    if h < 1:
        raise ValueError(f"horizon_len must be >= 1, got {h}")
    seq = context.detach()
    mus, log_sigmas, dir_logits, fed = [], [], [], []
    for _ in range(h):
        with torch.no_grad():
            out = model(seq)
        mu_t, ls_t, dl_t = out["mu"][:, -1], out["log_sigma"][:, -1], out["dir_logit"][:, -1]
        mus.append(mu_t)
        log_sigmas.append(ls_t)
        dir_logits.append(dl_t)
        nxt = mu_t.detach()
        fed.append(nxt)
        seq = torch.cat([seq, nxt[:, None]], dim=1)
    return {"mu": torch.stack(mus, dim=1), "log_sigma": torch.stack(log_sigmas, dim=1),
            "dir_logit": torch.stack(dir_logits, dim=1), "inputs": torch.stack(fed, dim=1)}
