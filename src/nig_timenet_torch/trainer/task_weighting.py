"""Kendall et al. uncertainty task weighting (source: crypto-signal-prediction trainer/task_weighting.py @ aeafc564).

total = sum_t 0.5 * exp(-log_var_t) * L_t + 0.5 * log_var_t. Only valid for POSITIVE task losses: the trainer keeps
evidential tasks (NIG NLL can go negative) out of it (``uncertainty_weighting_exclude_task_types``).
``init_log_var`` (FX2, default 0.0 = as before): the start value of every log-variance (B-060 candidate: a negative
value starts the Kendall tasks with a larger weight 0.5 exp(-init_log_var)).
"""

import torch
from torch import nn

__all__ = ["UncertaintyWeightedLoss"]


class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, task_names: list[str], init_log_var: float = 0.0):
        super().__init__()
        self.task_names = list(task_names)
        for t in self.task_names:
            setattr(self, f"log_var_{t}", nn.Parameter(torch.full((), float(init_log_var))))

    def log_var(self, t: str) -> nn.Parameter:
        return getattr(self, f"log_var_{t}")

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        total = torch.zeros(())
        for t in self.task_names:
            s = self.log_var(t)
            total = total + 0.5 * torch.exp(-s) * losses[t] + 0.5 * s
        return total

    def get_task_variances(self) -> dict[str, float]:
        return {t: float(torch.exp(self.log_var(t)).item()) for t in self.task_names}

    def get_effective_weights(self) -> dict[str, float]:
        var = self.get_task_variances()
        total_precision = sum(1.0 / v for v in var.values())
        return {t: (1.0 / var[t]) / total_precision for t in self.task_names}
