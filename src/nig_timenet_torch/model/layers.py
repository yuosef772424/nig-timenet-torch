"""Encoder layers of NIG-TimeNet v2, ported from crypto-signal-prediction model/*.py @ aeafc564 (BUILD_PLAN.md
P08-P13). Tensor layout is the TF one throughout: (batch, time, features), channels last.
"""

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["InstanceNorm", "SymLog", "CausalMultiScaleDecomp", "keras_dense_weight_to_torch"]


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
