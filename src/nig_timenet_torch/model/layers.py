"""Encoder layers of NIG-TimeNet v2, ported from crypto-signal-prediction model/*.py @ aeafc564 (BUILD_PLAN.md
P08-P13). Tensor layout is the TF one throughout: (batch, time, features), channels last.
"""

import torch
from torch import nn

__all__ = ["InstanceNorm", "SymLog"]


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
