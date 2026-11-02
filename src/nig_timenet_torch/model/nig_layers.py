"""NIG (Normal-Inverse-Gamma) head layers, ported from crypto-signal-prediction model/nig_layers.py @ aeafc564
(BUILD_PLAN.md P14): NIGHead, OrderedMeans, NIGUncertainty, ConfidenceHead. Keep outputs float32."""

import torch
import torch.nn.functional as F
from torch import nn

from nig_timenet_torch.core.constants import NIG_ALPHA_DEN_MIN

__all__ = ["NIGHead", "OrderedMeans", "NIGUncertainty", "ConfidenceHead"]


class NIGHead(nn.Module):
    """fc (Dense hidden, swish) -> dropout -> out (Dense 4) = raw [mu, nu, alpha, beta].
    mu is linear (no floor); nu/alpha/beta = softplus(raw) + nu_min/alpha_min/beta_min.
    ``l2`` (upstream default 1e-5) is fc's kernel regulariser: ``regularization_loss() = l2 * sum(fc.weight^2)``."""

    def __init__(self, in_features: int, hidden: int = 128, dropout: float = 0.1,
                 nu_min: float = 0.1, alpha_min: float = 2.0, beta_min: float = 0.01, l2: float = 1e-5):
        super().__init__()
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
        return mu, nu, alpha, beta


class OrderedMeans(nn.Module):
    """(mu_high, mu_close, mu_low) = (mu_c + softplus(raw_h), mu_c, mu_c - softplus(raw_l)).
    Input order is (close, high, low), output order (high, close, low) — as builder.py calls it."""

    def forward(self, mu_c: torch.Tensor, raw_h: torch.Tensor, raw_l: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return mu_c + F.softplus(raw_h), mu_c, mu_c - F.softplus(raw_l)


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
