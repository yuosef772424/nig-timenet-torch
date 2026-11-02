"""Trainer state + compute_losses (source: crypto-signal-prediction trainer/trainer.py @ aeafc564, GenericTrainer
__init__ and _compute_losses; BUILD_PLAN.md P18). The train/eval loop arrives in P22.

L2 kernel regularisation: torch has no per-layer kernel_regularizer, so the model exposes
``regularization_loss()`` (``l2 * sum(w^2)``, no 0.5 factor, NOTES.md #6) and compute_losses adds it once to both
total and raw loss, as the TF trainer adds ``model.losses``.
"""

from collections.abc import Mapping

import torch

from nig_timenet_torch.trainer.task_weighting import UncertaintyWeightedLoss
from nig_timenet_torch.trainer.tasks import TASK_REGISTRY

__all__ = ["Trainer", "compute_losses", "weight_for_target"]


def weight_for_target(sample_weight, target_name: str, tcfg: dict):
    """sample_weight is one tensor for every target, or a mapping keyed by target name or true_key."""
    if sample_weight is None:
        return None
    if isinstance(sample_weight, Mapping):
        for key in (target_name, tcfg["true_key"]):
            if key in sample_weight:
                return sample_weight[key]
        return None
    return sample_weight


def compute_losses(model: torch.nn.Module, x, y, targets_cfg: dict, scheduled_vars: dict[str, float],
                   loss_weights: dict[str, float], uncertainty_layer: UncertaintyWeightedLoss | None,
                   kendall_task_names: list[str], fixed_task_names: list[str], sample_weight=None,
                   training: bool = False):
    """-> (total_loss, raw_loss, per_target_losses, per_stat_values, per_constraint_values).
    raw_loss = sum of loss * loss_weight (no Kendall terms) + regularisation; total applies Kendall to
    kendall_task_names and adds fixed_task_names unweighted. ``training`` only gates flooding (the model's own
    train()/eval() mode gates dropout)."""
    outputs = model(x)
    per_target, per_stat = {}, {}
    for t, tcfg in targets_cfg.items():
        entry = TASK_REGISTRY[tcfg["task_type"]]
        w_t = weight_for_target(sample_weight, t, tcfg)
        if w_t is not None:
            if not entry["supports_sample_weight"]:
                raise ValueError(f"target {t!r} (task type {tcfg['task_type']!r}) does not support sample_weight")
            loss, stats = entry["fn"](y[tcfg["true_key"]], outputs, tcfg, scheduled_vars, sample_weight=w_t)
        else:
            loss, stats = entry["fn"](y[tcfg["true_key"]], outputs, tcfg, scheduled_vars)
        loss = loss.to(torch.float32)
        flood = tcfg.get("flood_level")
        if training and flood is not None:
            loss = (loss - float(flood)).abs() + float(flood)
        per_target[t] = loss * float(loss_weights[t])
        for k, v in stats.items():
            per_stat[f"{t}_{k}"] = torch.as_tensor(v, dtype=torch.float32)

    raw = torch.stack(list(per_target.values())).sum()
    if uncertainty_layer is not None:
        total = uncertainty_layer({t: per_target[t] for t in kendall_task_names})
        if fixed_task_names:
            total = total + torch.stack([per_target[t] for t in fixed_task_names]).sum()
    else:
        total = raw
    reg = getattr(model, "regularization_loss", None)
    if reg is not None:
        extra = reg()
        total, raw = total + extra, raw + extra
    return total, raw, per_target, per_stat, {}


class Trainer:
    """Minimal GenericTrainer state (P18): scheduled_vars (at their schedule ``start``), loss_weights, Kendall split.
    ``config`` is the trainer config dict ({"targets", "loss", optional "constraints"})."""

    def __init__(self, model: torch.nn.Module, config: dict):
        self.model, self.config = model, config
        self.target_names = list(config["targets"])
        loss_cfg = config["loss"]
        self.scheduled_vars = {n: float(c.get("start", 0.0)) for n, c in loss_cfg.get("schedules", {}).items()}
        self.loss_weights = {t: float(c.get("loss_weight", 1.0)) for t, c in config["targets"].items()}
        use_kendall = bool(loss_cfg.get("use_uncertainty_weighting", False))
        excluded = set(loss_cfg.get("uncertainty_weighting_exclude_task_types", ["evidential"]))
        self.kendall_task_names = [t for t, c in config["targets"].items()
                                   if use_kendall and c["task_type"] not in excluded]
        self.fixed_task_names = [t for t in self.target_names if t not in self.kendall_task_names]
        self.uncertainty_layer = UncertaintyWeightedLoss(self.kendall_task_names) if self.kendall_task_names else None
        constraints = list(loss_cfg.get("constraints", [])) + list(config.get("constraints", []))
        if constraints:
            raise NotImplementedError("output constraints are not ported yet")

    def parameters(self):
        yield from self.model.parameters()
        if self.uncertainty_layer is not None:
            yield from self.uncertainty_layer.parameters()

    def compute_losses(self, x, y, sample_weight=None, training: bool = False):
        return compute_losses(self.model, x, y, self.config["targets"], self.scheduled_vars, self.loss_weights,
                              self.uncertainty_layer, self.kendall_task_names, self.fixed_task_names,
                              sample_weight=sample_weight, training=training)
