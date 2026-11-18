"""PURPOSE: The HAR range terms of the RAW input window as an extra readout input (owner 2026-11-18, H3 of
    docs/results/batch-by-wf-002.md): mean bar range over the last 1 / 6 / all bars and their logs, batch-normalised.
TAGS: har range, volatility features, readout input, raw input, high low range, h3, switch, skip path
PITFALLS: it reads the input BEFORE the input scaler and the instance norm (both erase the window's volatility level,
    which is exactly what the ridge bound used); ``channels`` = (high, low) indices of the input's channels (the
    artifact's ``feature_order``: high, low, close -> (0, 1)); off by default (None: no module, the default graph and
    its parameter names are unchanged).

Why: a ridge on these terms beat every arm of the 60-coin walk-forward on the high and low heads in all four folds,
and the doctor showed the volatility level erased by ``inst_norm`` and only partly carried by the stats path. The
module hands the six numbers to the readout directly: features (B, 6) -> BatchNorm (Keras-style, moving statistics)
-> Linear(6, d_model) -> GELU, concatenated with the other readout parts.
"""

import torch
import torch.nn.functional as F
from torch import nn

from nig_timenet_torch.model.layers import BatchNormChannelsLast

__all__ = ["HarRange", "har_terms", "HAR_LAGS"]

HAR_LAGS = (1, 6)   # plus the whole window: the HAR-type daily / weekly / monthly means at the bar scale
_EPS = 1e-6


def har_terms(x: torch.Tensor, high: int, low: int) -> torch.Tensor:
    """(B, 6): the mean of (x[..., high] - x[..., low]) over the last 1 / 6 / all bars, then the log of each (the
    range clamped at 1e-6). ``x`` (B, T, F) raw log-move channels (log(high / prev close), log(low / prev close))."""
    rng = (x[..., high] - x[..., low]).clamp_min(0.0)
    t = x.shape[1]
    means = [rng[:, -min(k, t):].mean(dim=1) for k in HAR_LAGS] + [rng.mean(dim=1)]
    m = torch.stack(means, dim=-1)
    return torch.cat([m, torch.log(m.clamp_min(_EPS))], dim=-1)


class HarRange(nn.Module):
    """``har_terms`` -> BatchNormChannelsLast(6) -> Linear(6, d_model) -> GELU."""

    width = 2 * (len(HAR_LAGS) + 1)

    def __init__(self, channels, d_model: int, bn_warmup: bool = False):
        super().__init__()
        high, low = (int(c) for c in channels)
        if high == low or min(high, low) < 0:
            raise ValueError(f"har_range channels must be two different input channels (high, low), got {channels!r}")
        self.high, self.low = high, low
        self.bn = BatchNormChannelsLast(self.width, warmup=bn_warmup)
        self.proj = nn.Linear(self.width, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(self.bn(har_terms(x, self.high, self.low))))
