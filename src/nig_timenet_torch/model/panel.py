"""Cross-asset panel model (source: crypto-signal-prediction cross_asset/model.py @ aeafc564), BUILD_PLAN.md P43.

A shared per-coin temporal encoder (the NIG-TimeNet v2 trunk up to ``trunk_drop``) -> masked attention across the
coins of the same day -> per-coin heads. Inputs: ``{"x": (M, T, F) or {timeframe: (M, T_tf, F_tf)}, "day": (M,),
"pos": (M,)}``; the M rows are scattered to a (days, coins, d) grid by (day, pos), a key mask hides padding, one
day never sees another (each day is its own batch row), then the rows are gathered back. Outputs ``{"logit": (M, 3),
"mu": (M, 3)}`` in ``targets`` order: logit = "beats the day median" (sigmoid at export), mu = scaled relative
return. ``cross_attention=False`` (variant B) keeps the per-coin FFN and drops only the attention.

Torch module names follow the Keras layer names so ``load_oracle_weights`` maps Keras variable paths
('panel_model/cross_block_1/mha/query/kernel') onto parameters. The MultiHeadAttention projections keep the Keras
einsum layouts (query/key/value kernel (d, heads, key_dim), output kernel (heads, key_dim, d)): weights copy exactly.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from nig_timenet_torch.model.builder import build_nig_timenet_v2

__all__ = ["CrossAssetBlock", "KerasMultiHeadAttention", "PanelModel", "build_panel_model", "panel_encoder",
           "TARGETS"]

TARGETS = ("high", "low", "close")
_MASK_ADDER = -1e9  # Keras Softmax(mask): adds (1 - mask) * large negative before softmax


class _EinsumDense(nn.Module):
    def __init__(self, kernel_shape: tuple[int, ...], bias_shape: tuple[int, ...]):
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(kernel_shape))
        self.bias = nn.Parameter(torch.zeros(bias_shape))
        nn.init.xavier_uniform_(self.kernel.view(kernel_shape[0], -1) if len(kernel_shape) == 3 else self.kernel)


class KerasMultiHeadAttention(nn.Module):
    """keras.layers.MultiHeadAttention(num_heads, key_dim) self-attention with a boolean (B, Tq, Tk) mask."""

    def __init__(self, d_model: int, num_heads: int, key_dim: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads, self.key_dim = num_heads, key_dim
        self.query = _EinsumDense((d_model, num_heads, key_dim), (num_heads, key_dim))
        self.key = _EinsumDense((d_model, num_heads, key_dim), (num_heads, key_dim))
        self.value = _EinsumDense((d_model, num_heads, key_dim), (num_heads, key_dim))
        self.attention_output = _EinsumDense((num_heads, key_dim, d_model), (d_model,))
        nn.init.xavier_uniform_(self.attention_output.kernel.view(-1, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q = torch.einsum("btd,dhk->bthk", x, self.query.kernel) + self.query.bias
        k = torch.einsum("btd,dhk->bthk", x, self.key.kernel) + self.key.bias
        v = torch.einsum("btd,dhk->bthk", x, self.value.kernel) + self.value.bias
        q = q * (1.0 / math.sqrt(self.key_dim))
        scores = torch.einsum("bshk,bthk->bhts", k, q)  # Keras: einsum(key, query) -> (B, h, Tq, Tk)
        scores = scores + (1.0 - mask[:, None].to(scores.dtype)) * _MASK_ADDER
        attn = self.drop(torch.softmax(scores, dim=-1))
        ctx = torch.einsum("bhts,bshk->bthk", attn, v)
        return torch.einsum("bthk,hkd->btd", ctx, self.attention_output.kernel) + self.attention_output.bias


class CrossAssetBlock(nn.Module):
    """Pre-norm Transformer block over the coin axis: x + MHA(LN(x)), then x + FFN(LN(x)). attention=False drops
    the attention half."""

    def __init__(self, d_model=64, num_heads=4, ff_mult=2, dropout=0.25, attn_dropout=0.1, attention=True):
        super().__init__()
        self.attention = attention
        if attention:
            self.ln_attn = nn.LayerNorm(d_model, eps=1e-5)
            self.mha = KerasMultiHeadAttention(d_model, num_heads, d_model // num_heads, attn_dropout)
            self.drop1 = nn.Dropout(dropout)
        self.ln_ffn = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn_in = nn.Linear(d_model, ff_mult * d_model)
        self.ffn_out = nn.Linear(ff_mult * d_model, d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d); key_mask: (B, N) bool, True = a real coin. Every query (padding too) sees real keys only."""
        if self.attention:
            am = key_mask[:, None, :].expand(-1, x.shape[1], -1)
            x = x + self.drop1(self.mha(self.ln_attn(x), am))
        return x + self.drop2(self.ffn_out(F.gelu(self.ffn_in(self.ln_ffn(x)))))


class PanelModel(nn.Module):
    def __init__(self, encoder: nn.Module, enc_dim: int, d_model=64, n_cross_layers=1, num_heads=4,
                 cross_attention=True, dropout=0.25, attn_dropout=0.1, ff_mult=2, class_head_hidden=32,
                 head_hidden=64, targets=TARGETS):
        super().__init__()
        self.encoder, self.targets, self.d_model = encoder, tuple(targets), d_model
        self.cross_attention, self.n_cross_layers = cross_attention, n_cross_layers
        if enc_dim != d_model:
            self.enc_proj = nn.Linear(enc_dim, d_model)
        for i in range(n_cross_layers):
            setattr(self, f"cross_block_{i + 1}", CrossAssetBlock(d_model, num_heads, ff_mult, dropout, attn_dropout,
                                                                  cross_attention))
        self.panel_out_norm = nn.LayerNorm(d_model, eps=1e-5)
        for t in self.targets:
            setattr(self, f"cls_fc_{t}", nn.Linear(d_model, class_head_hidden))
            setattr(self, f"cls_logit_{t}", nn.Linear(class_head_hidden, 1))
            setattr(self, f"reg_fc_{t}", nn.Linear(d_model, head_hidden))
            setattr(self, f"reg_mu_{t}", nn.Linear(head_hidden, 1))
        self.head_drop = nn.Dropout(dropout)

    def encode(self, x) -> torch.Tensor:
        enc = self.encoder
        return enc.trunk(x) if hasattr(enc, "trunk") else enc(x)

    def forward(self, inputs: dict) -> dict[str, torch.Tensor]:
        day = torch.as_tensor(inputs["day"]).to(torch.int64)
        pos = torch.as_tensor(inputs["pos"]).to(torch.int64)
        h = self.encode(inputs["x"])
        if hasattr(self, "enc_proj"):
            h = self.enc_proj(h)
        if self.n_cross_layers:
            n_days, n_pos = int(day.max()) + 1, int(pos.max()) + 1
            # tf.scatter_nd sums duplicates; index_put_ with accumulate=True does the same
            grid = h.new_zeros((n_days, n_pos, self.d_model)).index_put((day, pos), h, accumulate=True)
            key_mask = h.new_zeros((n_days, n_pos)).index_put((day, pos), torch.ones_like(h[:, 0]),
                                                              accumulate=True) > 0
            for i in range(self.n_cross_layers):
                grid = getattr(self, f"cross_block_{i + 1}")(grid, key_mask)
            h = grid[day, pos]
        h = self.panel_out_norm(h)
        logit = torch.cat([getattr(self, f"cls_logit_{t}")(self.head_drop(F.relu(getattr(self, f"cls_fc_{t}")(h))))
                           for t in self.targets], dim=1)
        mu = torch.cat([getattr(self, f"reg_mu_{t}")(self.head_drop(F.silu(getattr(self, f"reg_fc_{t}")(h))))
                        for t in self.targets], dim=1)
        return {"logit": logit, "mu": mu}

    @torch.no_grad()
    def load_oracle_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """Keys: 'encoder.<attr path>' (passed to the encoder's own loader) and Keras variable paths
        'panel_model/<layer>/.../<var>' for the panel's layers. Raises on any unmatched or missing parameter."""
        enc = {k[len("encoder."):]: v for k, v in weights.items() if k.startswith("encoder.")}
        if enc:
            self.encoder.load_oracle_weights(enc)
        params = {n: p for n, p in self.named_parameters() if not n.startswith("encoder.")}
        assigned = set()
        for key, value in weights.items():
            if key.startswith("encoder."):
                continue
            parts = key.split("/")[1:]  # drop the model name
            leaf = parts[-1]
            dense = leaf == "kernel" and value.ndim == 2
            leaf = {"gamma": "weight", "beta": "bias"}.get(leaf, leaf)
            if dense:
                leaf, value = "weight", value.t()
            name = ".".join(parts[:-1] + [leaf])
            if name not in params:
                raise KeyError(f"panel weight {key!r} has no torch parameter {name!r}")
            if value.shape != params[name].shape:
                raise ValueError(f"{name}: shape {tuple(value.shape)} != {tuple(params[name].shape)}")
            params[name].copy_(value)
            assigned.add(name)
        missing = sorted(set(params) - assigned)
        if missing:
            raise KeyError(f"panel parameters not assigned by the oracle weights: {missing}")


def panel_encoder(seq_len, n_features, **model_kwargs) -> nn.Module:
    """The shared per-coin encoder: build_nig_timenet_v2 with no heads (upstream extract_encoder cuts the model at
    trunk_drop, so the heads' weights are unused and absent)."""
    return build_nig_timenet_v2(seq_len, n_features, **{**model_kwargs, "price_targets": (), "head_types": {}})


def build_panel_model(encoder: nn.Module, **cfg) -> PanelModel:
    """PanelModel over ``encoder``; its output width is read from ``encoder.trunk_dim`` (NIG-TimeNet trunk) or
    ``cfg['enc_dim']``."""
    cfg = dict(cfg)
    enc_dim = cfg.pop("enc_dim", None) or encoder.trunk_dim
    return PanelModel(encoder, enc_dim, **cfg)
