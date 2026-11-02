"""Encoder layers of NIG-TimeNet v2, ported from crypto-signal-prediction model/*.py @ aeafc564 (BUILD_PLAN.md
P08-P13). Tensor layout is the TF one throughout: (batch, time, features), channels last.
"""

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["InstanceNorm", "SymLog", "CausalMultiScaleDecomp", "keras_dense_weight_to_torch",
           "keras_conv1d_weight_to_torch", "num_patches", "PatchEmbedding",
           "RMSNorm", "RelativeGQAttention"]


class InstanceNorm(nn.Module):
    """RevIN-style per-window, per-feature normalisation (source: model/input_norm.py). Returns (z, stats) with
    stats = [mean, log std] per feature (B, 2F), so the volatility level is not lost.

    mean / population variance over the time axis (TF reduce_variance, i.e. mean((x - mean)^2)), std = sqrt(var + eps),
    z = (x - mean) / std, optional soft clip ``clip * tanh(z / clip)``, optional affine ``z * gamma + beta``.
    Only stats_mode="full" is implemented (the oracle's); "symlog"/"none" come with P33b.
    """

    STATS_MODES = ("full", "symlog", "none")

    def __init__(self, n_features: int, eps: float = 1e-4, affine: bool = True,
                 clip: float | None = None, stats_mode: str = "full", stats_clip: float = 8.0):
        super().__init__()
        if stats_mode not in self.STATS_MODES:
            raise ValueError(f"stats_mode must be one of {self.STATS_MODES}, not {stats_mode!r}")
        if stats_mode != "full":
            raise NotImplementedError(f"stats_mode={stats_mode!r} is not ported yet (BUILD_PLAN.md P08: P33b)")
        self.eps, self.affine, self.clip = eps, affine, clip
        self.stats_mode, self.stats_clip = stats_mode, stats_clip
        if affine:
            self.gamma = nn.Parameter(torch.ones(n_features))
            self.beta = nn.Parameter(torch.zeros(n_features))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).square().mean(dim=1, keepdim=True)
        std = torch.sqrt(var + self.eps)
        z = (x - mean) / std
        if self.clip:
            z = self.clip * torch.tanh(z / self.clip)
        if self.affine:
            z = z * self.gamma + self.beta
        stats = torch.cat([mean.squeeze(1), torch.log(std.squeeze(1))], dim=-1)
        return z, stats


class SymLog(nn.Module):
    """sign(x) * log1p(|x|), then clipped to [-clip, clip] (source: model/input_norm.py): passes a feature's raw LEVEL
    on a bounded scale (~identity for |x| < 1)."""

    def __init__(self, clip: float | None = 8.0):
        super().__init__()
        self.clip = clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.sign(x) * torch.log1p(torch.abs(x))
        return torch.clamp(y, -self.clip, self.clip) if self.clip else y


def keras_dense_weight_to_torch(kernel: torch.Tensor) -> torch.Tensor:
    """Keras Dense kernel (in, out) -> torch.nn.Linear weight (out, in)."""
    return kernel.t().contiguous()


class CausalMultiScaleDecomp(nn.Module):
    """Causal multi-scale trend/seasonal split (source: model/decomposition.py).

    For each kernel size k: left-pad by repeating the FIRST time step k-1 times, then a stride-1 average pool of
    size k (past-only moving average). A softmax gate over [x_last, mean |x_t - x_{t-1}|] (Dense 2F -> K, name
    ``gate``) mixes the K trends: trend = einsum("bktf,bk->btf"); seasonal = x - trend (so trend + seasonal == x).
    """

    def __init__(self, n_features: int, kernel_sizes: tuple[int, ...] = (3, 5, 9, 17)):
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.gate = nn.Linear(2 * n_features, len(self.kernel_sizes))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xc = x.transpose(1, 2)  # (B, F, T) for 1-D pooling over time
        trends = []
        for k in self.kernel_sizes:
            padded = F.pad(xc, (k - 1, 0), mode="replicate")  # repeat x[:, :1] (TF tf.repeat edge), not zero/reflect
            trends.append(F.avg_pool1d(padded, kernel_size=k, stride=1).transpose(1, 2))
        trends = torch.stack(trends, dim=1)  # (B, K, T, F)
        rough = (x[:, 1:, :] - x[:, :-1, :]).abs().mean(dim=1)
        g = torch.cat([x[:, -1, :], rough], dim=-1)
        w = torch.softmax(self.gate(g), dim=-1)
        trend = torch.einsum("bktf,bk->btf", trends, w)
        return trend, x - trend


def keras_conv1d_weight_to_torch(kernel: torch.Tensor) -> torch.Tensor:
    """Keras Conv1D kernel (kernel_size, in, out) -> torch.nn.Conv1d weight (out, in, kernel_size)."""
    return kernel.permute(2, 1, 0).contiguous()


def num_patches(seq_len: int, patch_len: int, stride: int) -> tuple[int, int]:
    """(n_tokens, left_pad) so that the last patch ends exactly at the last time step (source: model/patches.py)."""
    pad = (stride - (seq_len - patch_len) % stride) % stride
    return (seq_len + pad - patch_len) // stride + 1, pad


class PatchEmbedding(nn.Module):
    """Overlapping patches -> d_model tokens + learned absolute position embedding (source: model/patches.py).

    Left-pads by repeating the first step (``num_patches`` pad), Conv1d(in, d_model, patch_len, stride, valid), then
    ``+ pos`` of shape (1, n_tokens, d_model). ``pos`` depends on seq_len, which the BUILD_PLAN P10 interface does not
    take: pass ``seq_len=`` to build eagerly (needed before creating an optimizer / loading weights), otherwise it is
    built on the first forward like Keras' build() (see docs/friction_log.csv, P10).
    """

    def __init__(self, in_features: int, d_model: int, patch_len: int = 4, stride: int = 2,
                 *, seq_len: int | None = None):
        super().__init__()
        self.d_model, self.patch_len, self.stride = d_model, patch_len, stride
        self.proj = nn.Conv1d(in_features, d_model, kernel_size=patch_len, stride=stride, padding=0)
        self.seq_len: int | None = None
        self.pos: nn.Parameter | None = None
        if seq_len is not None:
            self.build(seq_len)

    def build(self, seq_len: int) -> None:
        if seq_len < self.patch_len:
            raise ValueError("seq_len must be >= patch_len")
        if self.pos is not None:
            if seq_len != self.seq_len:
                raise ValueError(f"PatchEmbedding was built for seq_len={self.seq_len}, got {seq_len}")
            return
        self.seq_len = seq_len
        self.n_tokens, self.pad = num_patches(seq_len, self.patch_len, self.stride)
        self.pos = nn.Parameter(torch.randn(1, self.n_tokens, self.d_model, device=self.proj.weight.device) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.build(x.shape[1])
        if self.pad > 0:
            x = torch.cat([x[:, :1, :].expand(-1, self.pad, -1), x], dim=1)
        return self.proj(x.transpose(1, 2)).transpose(1, 2) + self.pos


class RMSNorm(nn.Module):
    """``scale * x * rsqrt(mean(x^2, -1) + eps)`` (source: model/transformer.py RMSNorm)."""

    def __init__(self, d_model: int, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = x.square().mean(dim=-1, keepdim=True)
        return self.scale * x * torch.rsqrt(ms + self.epsilon)


class RelativeGQAttention(nn.Module):
    """Grouped-query self-attention with a learned relative-position bias (source: model/transformer.py).

    wq/wk/wv/wo have no bias; k, v are repeated num_heads // num_kv_heads times along the head axis; scores =
    q k^T / sqrt(d_h) + rel_bias[clip(i - j, -max_rel_pos, max_rel_pos) + max_rel_pos]; optional causal (i >= j) /
    window (|i - j| <= window) mask fills with finfo.min; softmax; dropout; ctx = attn v; wo(ctx).
    After each forward, ``last_scores_pre_softmax`` / ``last_attn_weights`` hold the (B, H, T, T) tensors.
    """

    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int | None = None, max_rel_pos: int = 16,
                 causal: bool = False, window: int | None = None, dropout: float = 0.0):
        super().__init__()
        num_kv_heads = num_kv_heads or num_heads
        if d_model % num_heads or num_heads % num_kv_heads:
            raise ValueError("d_model % num_heads and num_heads % num_kv_heads must both be 0")
        self.d_model, self.h, self.hkv = d_model, num_heads, num_kv_heads
        self.dh = d_model // num_heads
        self.max_rel_pos, self.causal, self.window = max_rel_pos, causal, window
        self.wq = nn.Linear(d_model, self.h * self.dh, bias=False)
        self.wk = nn.Linear(d_model, self.hkv * self.dh, bias=False)
        self.wv = nn.Linear(d_model, self.hkv * self.dh, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.rel_bias = nn.Parameter(torch.zeros(2 * max_rel_pos + 1, num_heads))
        self.drop = nn.Dropout(dropout)
        self.last_scores_pre_softmax: torch.Tensor | None = None
        self.last_attn_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.wq(x).reshape(b, t, self.h, self.dh)
        k = self.wk(x).reshape(b, t, self.hkv, self.dh)
        v = self.wv(x).reshape(b, t, self.hkv, self.dh)
        if self.hkv != self.h:
            rep = self.h // self.hkv
            k, v = k.repeat_interleave(rep, dim=2), v.repeat_interleave(rep, dim=2)  # == tf.repeat (not tile)
        q, k, v = (a.transpose(1, 2) for a in (q, k, v))  # (B, H, T, dh)

        scores = torch.matmul(q, k.transpose(-1, -2)) * (float(self.dh) ** -0.5)
        pos = torch.arange(t, device=x.device)
        rel = pos[:, None] - pos[None, :]
        idx = rel.clamp(-self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
        scores = scores + self.rel_bias[idx].permute(2, 0, 1)[None].to(scores.dtype)

        if self.causal or self.window is not None:
            allowed = torch.ones_like(rel, dtype=torch.bool)
            if self.causal:
                allowed = allowed & (rel >= 0)
            if self.window is not None:
                allowed = allowed & (rel.abs() <= self.window)
            scores = scores.masked_fill(~allowed[None, None], torch.finfo(scores.dtype).min)

        self.last_scores_pre_softmax = scores
        attn = torch.softmax(scores, dim=-1)
        self.last_attn_weights = attn
        ctx = torch.matmul(self.drop(attn), v).transpose(1, 2).reshape(b, t, self.d_model)
        return self.wo(ctx)
