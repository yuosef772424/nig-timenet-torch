"""Encoder layers of NIG-TimeNet v2, ported from crypto-signal-prediction model/*.py @ aeafc564 (BUILD_PLAN.md
P08-P13). Tensor layout is the TF one throughout: (batch, time, features), channels last.
"""

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["InstanceNorm", "SymLog", "CausalMultiScaleDecomp", "keras_dense_weight_to_torch",
           "keras_conv1d_weight_to_torch", "num_patches", "PatchEmbedding",
           "RMSNorm", "RelativeGQAttention", "SwiGLU", "TransformerBlock",
           "LastToken", "AttentionPool", "FiLM", "GatedFusion",
           "BatchNormChannelsLast", "SpatialDropout1D", "GaussianNoise", "KerasGRU"]


class InstanceNorm(nn.Module):
    """RevIN-style per-window, per-feature normalisation (source: model/input_norm.py). Returns (z, stats) with
    stats = [mean, log std] per feature (B, 2F), so the volatility level is not lost.

    mean / population variance over the time axis (TF reduce_variance, i.e. mean((x - mean)^2)), std = sqrt(var + eps),
    z = (x - mean) / std, optional soft clip ``clip * tanh(z / clip)``, optional affine ``z * gamma + beta``.
    stats_mode: "full" = [mean, log std]; "symlog" = [sign(mean) log1p|mean|, clip(log std, +-stats_clip)];
    "none" = stats returned as None (P33b).
    """

    STATS_MODES = ("full", "symlog", "none")

    def __init__(self, n_features: int, eps: float = 1e-4, affine: bool = True,
                 clip: float | None = None, stats_mode: str = "full", stats_clip: float = 8.0):
        super().__init__()
        if stats_mode not in self.STATS_MODES:
            raise ValueError(f"stats_mode must be one of {self.STATS_MODES}, not {stats_mode!r}")
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
        if self.stats_mode == "none":
            return z, None
        m, log_s = mean.squeeze(1), torch.log(std.squeeze(1))
        if self.stats_mode == "symlog":
            m = torch.sign(m) * torch.log1p(torch.abs(m))
            log_s = torch.clamp(log_s, -self.stats_clip, self.stats_clip)
        return z, torch.cat([m, log_s], dim=-1)


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


class SwiGLU(nn.Module):
    """``w2(silu(w1 x) * w3 x)``, no biases, no inner dropout (source: model/transformer.py SwiGLU)."""

    def __init__(self, d_model: int, dff: int | None = None):
        super().__init__()
        self.dff = dff or int(round(8 * d_model / 3 / 8) * 8)
        self.w1 = nn.Linear(d_model, self.dff, bias=False)
        self.w3 = nn.Linear(d_model, self.dff, bias=False)
        self.w2 = nn.Linear(self.dff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Pre-norm block (source: model/transformer.py TransformerBlock):
    x = x + drop1(attn(norm1(x))); x = x + drop2(ffn(norm2(x))). Dropout is identity in eval()."""

    def __init__(self, d_model, num_heads, num_kv_heads=None, max_rel_pos=16, causal=False, window=None,
                 dropout=0.1, attn_dropout=0.0):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d_model), RMSNorm(d_model)
        self.attn = RelativeGQAttention(d_model, num_heads, num_kv_heads, max_rel_pos, causal, window, attn_dropout)
        self.ffn = SwiGLU(d_model)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


class LastToken(nn.Module):
    """x[:, -1, :] (source: model/readout.py)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -1, :]


class AttentionPool(nn.Module):
    """Learned-query attention pooling over all tokens (source: model/readout.py):
    w = softmax(einsum("btd,d->bt", x, query) / sqrt(d)); out = einsum("bt,btd->bd", w, x)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = torch.tensor(float(x.shape[-1]), dtype=x.dtype, device=x.device)
        w = torch.softmax(torch.einsum("btd,d->bt", x, self.query) / torch.sqrt(d), dim=-1)
        return torch.einsum("bt,btd->bd", w, x)


class FiLM(nn.Module):
    """x * (1 + gamma(e)) + beta(e) per token, gamma/beta kernels zero-initialised (identity at start)
    (source: model/readout.py). Not wired into the builder until P34 (level_film is off in both oracles)."""

    def __init__(self, d_model: int, e_dim: int | None = None):
        super().__init__()
        e_dim = e_dim or d_model
        self.to_gamma, self.to_beta = nn.Linear(e_dim, d_model), nn.Linear(e_dim, d_model)
        for lin in (self.to_gamma, self.to_beta):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + self.to_gamma(e)[:, None, :]) + self.to_beta(e)[:, None, :]


class GatedFusion(nn.Module):
    """g = sigmoid(W [a; b]); g * a + (1 - g) * b (source: model/readout.py). Not wired until P34 (two_tower off)."""

    def __init__(self, d_model: int, b_dim: int | None = None):
        super().__init__()
        self.gate = nn.Linear(d_model + (b_dim or d_model), d_model)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([a, b], dim=-1)))
        return g * a + (1.0 - g) * b


class BatchNormChannelsLast(nn.Module):
    """Keras BatchNormalization(axis=-1) on (B, F) or (B, T, F): attribute names match Keras (gamma, beta, and the
    moving_mean / moving_variance BUFFERS), epsilon 1e-3, momentum 0.99 (torch momentum 0.01). Training uses the
    batch statistics over every axis but the last and updates the moving averages (population-unbiased variance,
    as Keras); eval uses the moving averages."""

    def __init__(self, n_features: int, epsilon: float = 1e-3, momentum: float = 0.99):
        super().__init__()
        self.epsilon, self.momentum = epsilon, momentum
        self.gamma = nn.Parameter(torch.ones(n_features))
        self.beta = nn.Parameter(torch.zeros(n_features))
        self.register_buffer("moving_mean", torch.zeros(n_features))
        self.register_buffer("moving_variance", torch.ones(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            dims = tuple(range(x.ndim - 1))
            mean, var = x.mean(dim=dims), x.var(dim=dims, unbiased=False)
            with torch.no_grad():
                self.moving_mean.mul_(self.momentum).add_((1 - self.momentum) * mean)
                self.moving_variance.mul_(self.momentum).add_((1 - self.momentum) * var)
        else:
            mean, var = self.moving_mean, self.moving_variance
        return (x - mean) * torch.rsqrt(var + self.epsilon) * self.gamma + self.beta


class SpatialDropout1D(nn.Module):
    """Keras SpatialDropout1D on (B, T, F): drops whole feature channels per sample in training; identity in eval."""

    def __init__(self, rate: float):
        super().__init__()
        self.drop = nn.Dropout1d(rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x.transpose(1, 2)).transpose(1, 2)


class GaussianNoise(nn.Module):
    """Keras GaussianNoise: x + N(0, stddev^2) in training; identity in eval."""

    def __init__(self, stddev: float):
        super().__init__()
        self.stddev = stddev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.stddev if self.training else x


class KerasGRU(nn.Module):
    """Keras GRU(units, return_sequences=True) (reset_after=True, sigmoid/tanh, zero initial state) on torch.nn.GRU.
    Both use n = tanh(W_n x + b_in + r * (U_n h + b_hn)) and h' = z * h + (1 - z) * n; only the gate order differs
    (Keras z|r|h, torch r|z|n) — ``keras_weight_map`` converts the Keras cell weights."""

    def __init__(self, in_features: int, units: int):
        super().__init__()
        self.units = units
        self.cell = nn.GRU(in_features, units, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cell(x)[0]

    def _reorder(self, w: torch.Tensor) -> torch.Tensor:  # last axis Keras [z, r, h] -> torch [r, z, n]
        z, r, h = w.split(self.units, dim=-1)
        return torch.cat([r, z, h], dim=-1)

    def keras_weight_map(self, leaf: str, value: torch.Tensor) -> dict[str, torch.Tensor]:
        """Keras cell weight (``kernel`` (in, 3u), ``recurrent_kernel`` (u, 3u), ``bias`` (2, 3u)) -> torch
        parameter names relative to this module."""
        if leaf == "kernel":
            return {"cell.weight_ih_l0": self._reorder(value).t().contiguous()}
        if leaf == "recurrent_kernel":
            return {"cell.weight_hh_l0": self._reorder(value).t().contiguous()}
        if leaf == "bias":
            return {"cell.bias_ih_l0": self._reorder(value[0]), "cell.bias_hh_l0": self._reorder(value[1])}
        raise KeyError(f"unknown Keras GRU weight {leaf!r}")
