"""NIG (Normal-Inverse-Gamma) head layers, ported from crypto-signal-prediction model/nig_layers.py @ aeafc564
(BUILD_PLAN.md P14): NIGHead, OrderedMeans, NIGUncertainty, ConfidenceHead. Keep outputs float32.

PURPOSE:  The NIG head layers: NIGHead (mu/nu/alpha/beta with floors; FX1 output scale), OrderedMeans (high >= close
          >= low; FX1 scale / detach close), NIGUncertainty, ConfidenceHead.
TAGS:     nig head, ordered means, evidential, uncertainty, confidence head, softplus offset, head parametrisation
PITFALLS: labels are raw returns (~0.02): any softplus / floor in label units (beta_min, the ordered offsets) acts on
          that scale (B-052, FX1). OrderedMeans switches at their defaults reproduce the TF formula bit for bit.
SOURCE:   crypto-signal-prediction model/nig_layers.py @ aeafc564
"""

import torch
import torch.nn.functional as F
from torch import nn

from nig_timenet_torch.core.constants import NIG_ALPHA_DEN_MIN

__all__ = ["NIGHead", "OrderedMeans", "NIGUncertainty", "ConfidenceHead", "grad_scale"]


def grad_scale(x: torch.Tensor, g: float) -> torch.Tensor:
    """x in the forward pass; the gradient that flows back through it is multiplied by ``g`` (FX2 switch
    ``nig_trunk_grad``: g < 1 lets the NIG heads shape the trunk less; g = 0 is a stop-gradient)."""
    return x if g == 1.0 else x * g + x.detach() * (1.0 - g)


def _scale_arg(module: nn.Module, scale: float | None, fitted_scale: bool) -> float | None:
    """The fixed scale (float > 0 or None); with ``fitted_scale`` (FX2, ``model.label_scale``) ``module.fitted`` is
    True and a ``label_scale`` buffer is registered (1.0 until ``fit_label_scale`` writes the training-label scale)."""
    module.fitted = bool(fitted_scale)
    if fitted_scale:
        if scale is not None:
            raise ValueError("a fitted label scale and a fixed nig_scale are exclusive")
        module.register_buffer("label_scale", torch.ones(()))
        return None
    if scale is not None and not scale > 0:
        raise ValueError(f"scale must be > 0 or None, not {scale!r}")
    return None if scale is None else float(scale)


class NIGHead(nn.Module):
    """fc (Dense hidden, swish) -> dropout -> out (Dense 4) = raw [mu, nu, alpha, beta].
    mu is linear (no floor); nu/alpha/beta = softplus(raw) + nu_min/alpha_min/beta_min.
    ``l2`` (upstream default 1e-5) is fc's kernel regulariser: ``regularization_loss() = l2 * sum(fc.weight^2)``.

    ``scale`` s (FX1 switch, None = the formula above): the head works in label units / s, i.e. mu = s * raw_mu and
    beta = s^2 * (softplus + beta_min) (nu, alpha are unit-free). The NIG NLL then differs from the NLL of labels / s
    by a constant, so training equals training on labels scaled by 1 / s, while every output stays in label units.
    TF trained on labels x 100 (reg_target_scale 100); s = 0.01 restores that on a scale-1 artifact (B-052, B-062).
    ``fitted_scale`` (FX2): s is the ``label_scale`` buffer that ``model.label_scale.fit_label_scale`` fits on the
    training labels (beta floor proportional to the label variance)."""

    def __init__(self, in_features: int, hidden: int = 128, dropout: float = 0.1,
                 nu_min: float = 0.1, alpha_min: float = 2.0, beta_min: float = 0.01, l2: float = 1e-5,
                 scale: float | None = None, fitted_scale: bool = False):
        super().__init__()
        self.scale = _scale_arg(self, scale, fitted_scale)
        self.l2 = l2
        self.nu_min, self.alpha_min, self.beta_min = nu_min, alpha_min, beta_min
        self.fc = nn.Linear(in_features, hidden)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 4)

    def regularization_loss(self) -> torch.Tensor:
        return self.l2 * self.fc.weight.square().sum()

    def raw_params(self, h: torch.Tensor) -> torch.Tensor:
        return self.out(self.drop(F.silu(self.fc(h))))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.raw_params(h)
        mu = p[:, 0:1]
        nu = F.softplus(p[:, 1:2]) + self.nu_min
        alpha = F.softplus(p[:, 2:3]) + self.alpha_min
        beta = F.softplus(p[:, 3:4]) + self.beta_min
        s = self.label_scale if self.fitted else self.scale
        if s is not None:
            mu, beta = s * mu, (s * s) * beta
        return mu, nu, alpha, beta


class OrderedMeans(nn.Module):
    """(mu_high, mu_close, mu_low) = (mu_c + softplus(raw_h), mu_c, mu_c - softplus(raw_l)).
    Input order is (close, high, low), output order (high, close, low) — as builder.py calls it.

    Switches (FX1, defaults = the formula above exactly):
    * ``scale`` s: the inputs come from NIGHeads with ``scale=s`` (raw values already multiplied by s), and the offsets
      become ``s * softplus(raw / s)`` = s * softplus(the head's unscaled raw). With s = None and labels of ~0.02 the
      offset lives in softplus's exponential tail (raw ~ -4, slope sigmoid(-4) ~ 0.02) and starts 30x too large.
    * ``detach_close``: the high / low offsets are added to ``mu_c.detach()``, so the high and low losses no longer
      move the close mean (E1: they pulled close mu to corr -0.19 with its true mean, B-063)."""

    def __init__(self, scale: float | None = None, detach_close: bool = False, fitted_scale: bool = False):
        super().__init__()
        self.scale = _scale_arg(self, scale, fitted_scale)
        self.detach_close = bool(detach_close)

    def _offset(self, raw: torch.Tensor) -> torch.Tensor:
        s = self.label_scale if self.fitted else self.scale
        return F.softplus(raw) if s is None else s * F.softplus(raw / s)

    def forward(self, mu_c: torch.Tensor, raw_h: torch.Tensor, raw_l: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = mu_c.detach() if self.detach_close else mu_c
        return base + self._offset(raw_h), mu_c, base - self._offset(raw_l)


class NIGUncertainty(nn.Module):
    """Amini et al.: aleatoric = sqrt(beta / a1), epistemic = sqrt(beta / (nu a1)), a1 = max(alpha - 1,
    NIG_ALPHA_DEN_MIN); with unc_max set, non-finite values become unc_max and both are capped at unc_max.
    Returns (epistemic, aleatoric)."""

    def __init__(self, unc_max: float | None = 20.0):
        super().__init__()
        self.unc_max = None if unc_max is None else float(unc_max)

    def forward(self, nu, alpha, beta) -> tuple[torch.Tensor, torch.Tensor]:
        a1 = torch.clamp_min(alpha - 1.0, NIG_ALPHA_DEN_MIN)
        aleatoric = torch.sqrt(beta / a1)
        epistemic = torch.sqrt(beta / (nu * a1))
        if self.unc_max is not None:
            cap = torch.tensor(self.unc_max, dtype=aleatoric.dtype, device=aleatoric.device)
            aleatoric = torch.minimum(torch.where(torch.isfinite(aleatoric), aleatoric, cap), cap)
            epistemic = torch.minimum(torch.where(torch.isfinite(epistemic), epistemic, cap), cap)
        return epistemic, aleatoric


class ConfidenceHead(nn.Module):
    """0.05 + 0.93 * sigmoid(d2(relu(d1(stop_grad([h, log nu, log(alpha - 1), log beta]))))).
    ``in_features`` is h's width; d1's input is in_features + 3 (nu/alpha/beta are (B, 1) each)."""

    def __init__(self, in_features: int, hidden: int = 32):
        super().__init__()
        self.d1 = nn.Linear(in_features + 3, hidden)
        self.d2 = nn.Linear(hidden, 1)

    def forward(self, h, nu, alpha, beta) -> torch.Tensor:
        u = torch.cat([h, torch.log(nu), torch.log(alpha - 1.0), torch.log(beta)], dim=-1).detach()
        return 0.05 + 0.93 * torch.sigmoid(self.d2(torch.relu(self.d1(u))))
