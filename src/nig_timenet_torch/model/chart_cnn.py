"""PURPOSE: ChartCNN: a model built from scratch (no project layer): each input window is drawn as a 2-channel chart
    image (high-low range bars + close line), a 2D CNN reads it, one shared head outputs every target's numbers.
TAGS: chart image, cnn, candlestick image, robust to outliers, new architecture, chart_cnn, from scratch
PITFALLS: the image is min-max scaled PER WINDOW (an outlier bar squeezes the others, it cannot blow up the
    activations); the scale it removes is given back as ONE number (log of the window's range); the drawing is a hard
    threshold, so the input gradient is zero (GradNet's "input" token is all zeros for this model).

Drawing (Jiang, Kelly & Xiu 2023, "(Re-)Imag(in)ing Price Trends", adapted to return inputs): the features are per-bar
returns ``high, low, close`` (log moves vs the previous close, standardised); the close path is their cumulative sum,
bar t spans [prev + low_t, prev + high_t] (widened to contain prev and close_t). Channel 0 = the range bar of each
column, channel 1 = the close pixel. Rows = ``height`` price levels, columns = the window's bars.

Body: ``n_blocks`` x [Conv2d(5x3) -> BatchNorm2d -> LeakyReLU -> MaxPool(2x1)] (channels 32, 64, 128, ...), the
mean over a (2, 8) grid (``grid_pool``), Linear -> ``hidden``. Head (written here, not the project's head modules): one
Linear to [mu, nu, alpha, beta] x targets, one Linear to the direction probability per target, and a confidence from
the detached features; the NIG parameters use the label scale ``out_scale`` (labels are ~0.02: mu = s * raw, beta =
s^2 * (softplus + beta_min)). The output dict has the same 24 keys and meanings as every other architecture.
"""

import torch
import torch.nn.functional as F
from torch import nn

from nig_timenet_torch.core.constants import NIG_ALPHA_DEN_MIN
from nig_timenet_torch.core.schema import TARGET_COLUMNS
from nig_timenet_torch.model.heads import class_outputs, head_spec, nig_outputs

__all__ = ["ChartCNN", "HeadView", "build_chart_cnn", "grid_pool", "nig_params", "output_dict", "draw_charts",
           "CHART_CNN_DEFAULT_CONFIG"]

CHART_CNN_DEFAULT_CONFIG = {"price_targets": list(TARGET_COLUMNS), "height": 32, "n_blocks": 3, "hidden": 64,
                            "dropout": 0.3, "out_scale": 0.01}
_NU_MIN, _ALPHA_MIN, _BETA_MIN, _UNC_MAX = 0.1, 2.0, 0.01, 20.0


def draw_charts(x: torch.Tensor, height: int, cols: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """(images (B, 2, height, T) in {0, 1}, log range (B, 1)) of the windows ``x`` (B, T, F); ``cols`` = the feature
    indices of (high, low, close)."""
    hi_r, lo_r, cl_r = (x[:, :, c] for c in cols)
    close = torch.cumsum(cl_r, dim=1)
    prev = torch.cat([torch.zeros_like(close[:, :1]), close[:, :-1]], dim=1)
    top = torch.maximum(prev + hi_r, torch.maximum(prev, close))
    bot = torch.minimum(prev + lo_r, torch.minimum(prev, close))
    lo = bot.min(dim=1, keepdim=True).values
    rng = (top.max(dim=1, keepdim=True).values - lo).clamp_min(1e-6)
    to_row = lambda v: ((v - lo) / rng * (height - 1)).round().clamp(0, height - 1)   # noqa: E731
    rows = torch.arange(height, device=x.device, dtype=x.dtype).view(1, height, 1)   # row 0 = lowest price
    top_r, bot_r, cl_row = to_row(top).unsqueeze(1), to_row(bot).unsqueeze(1), to_row(close).unsqueeze(1)
    bars = ((rows >= bot_r) & (rows <= top_r)).to(x.dtype)
    line = (rows == cl_row).to(x.dtype)
    return torch.stack([bars, line], dim=1).flip(2), rng.log()


def grid_pool(h: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, C * rows * cols): the mean over a rows x cols grid of equal cells (zero padding to a
    multiple). Plain reshape + mean, so its CUDA backward is deterministic (AdaptiveAvgPool2d's is not, and the
    trainer runs with torch.use_deterministic_algorithms)."""
    b, c, hh, ww = h.shape
    ph, pw = -(-hh // rows) * rows, -(-ww // cols) * cols
    h = F.pad(h, (0, pw - ww, 0, ph - hh))
    return h.view(b, c, rows, ph // rows, cols, pw // cols).mean(dim=(3, 5)).flatten(1)


def nig_params(p: torch.Tensor, scale: float) -> tuple[torch.Tensor, ...]:
    """(mu, nu, alpha, beta), each (B, n_targets), from raw (B, n_targets, 4) in label scale ``scale``."""
    return (scale * p[..., 0], F.softplus(p[..., 1]) + _NU_MIN, F.softplus(p[..., 2]) + _ALPHA_MIN,
            scale ** 2 * (F.softplus(p[..., 3]) + _BETA_MIN))


def output_dict(targets, mu, nu, alpha, beta, conf, prob) -> dict[str, torch.Tensor]:
    """The 24-key output contract from (B, n_targets) tensors (the uncertainties follow Amini et al., capped)."""
    a1 = (alpha - 1.0).clamp_min(NIG_ALPHA_DEN_MIN)
    ale = torch.sqrt(beta / a1).clamp(max=_UNC_MAX)
    epi = torch.sqrt(beta / (nu * a1)).clamp(max=_UNC_MAX)
    out = {}
    for i, t in enumerate(targets):
        c = slice(i, i + 1)
        out.update(nig_outputs(head_spec("nig_regression", t), mu[:, c], nu[:, c], alpha[:, c], beta[:, c],
                               epi[:, c], ale[:, c], conf[:, c]))
        out.update(class_outputs(head_spec("binary_classification", t), prob[:, c]))
    return out


class HeadView(nn.Module):
    """The health-report view of one target's NIG head (floors in label units + the shared head's parameters), so
    ``eval.nig_floor`` and ``trainer.grad_report`` read this model like the others; it owns no module."""

    def __init__(self, shared: nn.Linear, scale: float):
        super().__init__()
        object.__setattr__(self, "_shared", shared)     # not registered: the weights stay in ChartCNN.nig_out
        self.nu_min, self.alpha_min, self.beta_min = _NU_MIN, _ALPHA_MIN, scale ** 2 * _BETA_MIN

    def parameters(self, recurse: bool = True):
        return self._shared.parameters(recurse)


class ChartCNN(nn.Module):
    default_probe_points = ("trunk",)                   # the last shared layer

    def __init__(self, seq_len: int, n_features: int, price_targets=TARGET_COLUMNS, height: int = 32,
                 n_blocks: int = 3, hidden: int = 64, dropout: float = 0.3, out_scale: float = 0.01,
                 feature_cols: tuple[int, int, int] = (0, 1, 2), head_types: dict | None = None):
        super().__init__()
        if n_features < 3 or height < 2 ** n_blocks or not out_scale > 0:
            raise ValueError("ChartCNN needs >= 3 features (high, low, close), height >= 2**n_blocks, out_scale > 0")
        if head_types is not None and any(sorted(v) != sorted(["nig_regression", "binary_classification"])
                                          for v in head_types.values()):
            raise NotImplementedError("ChartCNN builds the nig_regression + binary_classification heads only")
        self.targets, self.height, self.cols = tuple(price_targets), int(height), tuple(feature_cols)
        self.scale = float(out_scale)
        layers, c_in = [], 2
        for i in range(n_blocks):
            c_out = 32 * 2 ** i
            layers += [nn.Conv2d(c_in, c_out, (5, 3), padding=(2, 1)), nn.BatchNorm2d(c_out), nn.LeakyReLU(0.01),
                       nn.MaxPool2d((2, 1))]
            c_in = c_out
        self.body = nn.Sequential(*layers)
        self.trunk = nn.Sequential(nn.Dropout(dropout), nn.Linear(c_in * 16 + 1, hidden), nn.LeakyReLU(0.01))
        n = len(self.targets)
        self.nig_out = nn.Linear(hidden, 4 * n)
        self.dir_out = nn.Linear(hidden, n)
        self.conf_out = nn.Sequential(nn.Linear(hidden + 3 * n, 32), nn.LeakyReLU(0.01), nn.Linear(32, n))

    def head_modules(self) -> dict[str, HeadView]:
        return {t: HeadView(self.nig_out, self.scale) for t in self.targets}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        img, log_rng = draw_charts(x, self.height, self.cols)
        h = self.trunk(torch.cat([grid_pool(self.body(img), 2, 8), log_rng], dim=1))
        mu, nu, alpha, beta = nig_params(self.nig_out(h).view(-1, len(self.targets), 4), self.scale)
        conf_in = torch.cat([h, nu.log(), (alpha - 1.0).log(), beta.log()], dim=1).detach()
        conf = 0.05 + 0.93 * torch.sigmoid(self.conf_out(conf_in))
        return output_dict(self.targets, mu, nu, alpha, beta, conf, torch.sigmoid(self.dir_out(h)))

def build_chart_cnn(seq_len: int, n_features: int, **config) -> ChartCNN:
    return ChartCNN(int(seq_len), int(n_features), **config)
