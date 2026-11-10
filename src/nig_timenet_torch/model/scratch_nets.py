"""PURPOSE: Two models built from scratch (no project layer) to find where memorization starts and learning breaks:
    conv1d_net (dilated 1D convolutions + a dense head) and no_dense_net (no Linear layer anywhere: convolutions only).
TAGS: conv1d, dilated convolution, no dense, fully convolutional, global pooling, new architecture, from scratch
PITFALLS: both squash the inputs with sign(x) log1p(|x|) (an outlier bar stays bounded, no learned state); no_dense_net
    has no 1x1 convolution either (every kernel spans 3 steps) and reads its outputs as channels after a global average
    over time, so its head has no per-row dense mixing; the output dict is the 24-key contract of every architecture.

conv1d_net: ``n_blocks`` x [Conv1d(k=5, dilation 2^i) -> BatchNorm1d -> LeakyReLU -> Dropout] with a residual, then the
average AND the max over time, one hidden Linear, and the shared head (``chart_cnn.nig_params`` / ``output_dict``).
no_dense_net: ``n_blocks`` x [Conv1d(k=3, dilation 2^i) -> BatchNorm1d -> LeakyReLU] with a residual, then three
Conv1d(k=3) output maps (NIG parameters 4 x targets, direction 1 x targets, confidence 1 x targets from the detached
features), each averaged over time.
"""

import torch
import torch.nn.functional as F
from torch import nn

from nig_timenet_torch.core.schema import TARGET_COLUMNS
from nig_timenet_torch.model.chart_cnn import HeadView, nig_params, output_dict

__all__ = ["Conv1DNet", "NoDenseNet", "build_conv1d_net", "build_no_dense_net", "CONV1D_NET_DEFAULT_CONFIG",
           "NO_DENSE_NET_DEFAULT_CONFIG", "squash"]

CONV1D_NET_DEFAULT_CONFIG = {"price_targets": list(TARGET_COLUMNS), "channels": 64, "n_blocks": 3, "hidden": 64,
                             "dropout": 0.2, "out_scale": 0.01}
NO_DENSE_NET_DEFAULT_CONFIG = {"price_targets": list(TARGET_COLUMNS), "channels": 64, "n_blocks": 4,
                               "out_scale": 0.01}


def squash(x: torch.Tensor) -> torch.Tensor:
    """sign(x) log1p(|x|): identity near 0, logarithmic for outliers (no learned state)."""
    return torch.sign(x) * torch.log1p(x.abs())


def _check(head_types):
    if head_types is not None and any(sorted(v) != sorted(["nig_regression", "binary_classification"])
                                      for v in head_types.values()):
        raise NotImplementedError("this model builds the nig_regression + binary_classification heads only")


class _ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=dilation, padding=dilation * (k - 1) // 2)
        self.norm = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, h):                                  # (B, C, T) -> (B, C_out, T)
        return self.drop(F.leaky_relu(self.norm(self.conv(h)), 0.01))


class _ScratchBase(nn.Module):
    def head_modules(self) -> dict[str, HeadView]:
        return {t: HeadView(self.nig_out, self.scale) for t in self.targets}


class Conv1DNet(_ScratchBase):
    default_probe_points = ("trunk",)

    def __init__(self, seq_len: int, n_features: int, price_targets=TARGET_COLUMNS, channels: int = 64,
                 n_blocks: int = 3, hidden: int = 64, dropout: float = 0.2, out_scale: float = 0.01,
                 head_types: dict | None = None):
        super().__init__()
        _check(head_types)
        self.targets, self.scale, n = tuple(price_targets), float(out_scale), len(price_targets)
        self.stem = nn.Conv1d(n_features, channels, 5, padding=2)
        self.blocks = nn.ModuleList(_ConvBlock(channels, channels, 5, 2 ** i, dropout) for i in range(n_blocks))
        self.trunk = nn.Sequential(nn.Linear(2 * channels, hidden), nn.LeakyReLU(0.01), nn.Dropout(dropout))
        self.nig_out = nn.Linear(hidden, 4 * n)
        self.dir_out = nn.Linear(hidden, n)
        self.conf_out = nn.Sequential(nn.Linear(hidden + 3 * n, 32), nn.LeakyReLU(0.01), nn.Linear(32, n))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.stem(squash(x).transpose(1, 2))
        for b in self.blocks:
            h = h + b(h)
        h = self.trunk(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1))
        mu, nu, alpha, beta = nig_params(self.nig_out(h).view(-1, len(self.targets), 4), self.scale)
        conf_in = torch.cat([h, nu.log(), (alpha - 1.0).log(), beta.log()], dim=1).detach()
        conf = 0.05 + 0.93 * torch.sigmoid(self.conf_out(conf_in))
        return output_dict(self.targets, mu, nu, alpha, beta, conf, torch.sigmoid(self.dir_out(h)))


class NoDenseNet(_ScratchBase):
    default_probe_points = ("blocks",)

    def __init__(self, seq_len: int, n_features: int, price_targets=TARGET_COLUMNS, channels: int = 64,
                 n_blocks: int = 4, out_scale: float = 0.01, head_types: dict | None = None):
        super().__init__()
        _check(head_types)
        self.targets, self.scale, n = tuple(price_targets), float(out_scale), len(price_targets)
        self.stem = nn.Conv1d(n_features, channels, 3, padding=1)
        self.blocks = nn.Sequential(*(_ConvBlock(channels, channels, 3, 2 ** i, 0.0) for i in range(n_blocks)))
        self.nig_out = nn.Conv1d(channels, 4 * n, 3, padding=1)
        self.dir_out = nn.Conv1d(channels, n, 3, padding=1)
        self.conf_out = nn.Conv1d(channels, n, 3, padding=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.stem(squash(x).transpose(1, 2))
        for b in self.blocks:
            h = h + b(h)
        mu, nu, alpha, beta = nig_params(self.nig_out(h).mean(dim=2).view(-1, len(self.targets), 4), self.scale)
        conf = 0.05 + 0.93 * torch.sigmoid(self.conf_out(h.detach()).mean(dim=2))
        return output_dict(self.targets, mu, nu, alpha, beta, conf, torch.sigmoid(self.dir_out(h).mean(dim=2)))


def build_conv1d_net(seq_len: int, n_features: int, **config) -> Conv1DNet:
    return Conv1DNet(int(seq_len), int(n_features), **config)


def build_no_dense_net(seq_len: int, n_features: int, **config) -> NoDenseNet:
    return NoDenseNet(int(seq_len), int(n_features), **config)
