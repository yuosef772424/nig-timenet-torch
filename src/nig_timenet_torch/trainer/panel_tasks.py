"""Task losses for a model whose outputs and targets are (days, slots, K) grids (model/grid_panel.py).

Every task reads one column of the target grid (``column`` in the target config) and the matching column of the
output, and counts only real slots (the model returns its mask under ``output_keys["mask"]``). Nothing is flattened to
a list of coins: the per-day structure is used as it is, which is what makes the rank loss a plain row-wise operation.

    panel_classification  binary cross-entropy on a logit (config: column, label_smoothing); stat: accuracy
    panel_regression      mse, mae or huber on the scaled relative return (config: column, loss_fn, huber_delta)
    panel_rank_ic         minus the mean over days of the Pearson correlation between the output and the day's
                          cross-sectional rank target (config: column, min_slots); stat: ic
"""

import torch
import torch.nn.functional as F

from nig_timenet_torch.trainer.tasks import register_task_type

__all__ = ["panel_classification_loss", "panel_regression_loss", "panel_rank_loss"]

EPS = 1e-8


def _column(tensor: torch.Tensor, cfg: dict) -> torch.Tensor:
    return tensor[..., int(cfg.get("column", 0))].to(torch.float32)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(torch.float32)
    return (x * m).sum() / m.sum().clamp_min(1.0)


def panel_classification_loss(y_true, outputs, cfg, scheduled_vars, sample_weight=None):
    keys = cfg["output_keys"]
    logit, mask, y = _column(outputs[keys["logits"]], cfg), outputs[keys["mask"]], _column(y_true, cfg)
    ls = float(cfg.get("label_smoothing", 0.0))
    target = y * (1.0 - ls) + 0.5 * ls if ls > 0 else y
    per = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    acc = ((logit >= 0).to(torch.float32) == y).to(torch.float32)
    return _masked_mean(per, mask), {"accuracy": _masked_mean(acc, mask)}


def panel_regression_loss(y_true, outputs, cfg, scheduled_vars, sample_weight=None):
    keys = cfg["output_keys"]
    error = _column(y_true, cfg) - _column(outputs[keys["pred"]], cfg)
    mask = outputs[keys["mask"]]
    kind = cfg.get("loss_fn", "mse")
    if kind == "mae":
        per = error.abs()
    elif kind == "huber":
        delta = float(cfg.get("huber_delta", 1.0))
        a = error.abs()
        q = torch.clamp_max(a, delta)
        per = 0.5 * q.square() + delta * (a - q)
    else:
        per = error.square()
    return _masked_mean(per, mask), {"mae": _masked_mean(error.abs(), mask)}


def panel_rank_loss(y_true, outputs, cfg, scheduled_vars, sample_weight=None):
    keys = cfg["output_keys"]
    p, y, mask = _column(outputs[keys["pred"]], cfg), _column(y_true, cfg), outputs[keys["mask"]].to(torch.float32)
    n = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    pc = (p - (p * mask).sum(dim=1, keepdim=True) / n) * mask
    yc = (y - (y * mask).sum(dim=1, keepdim=True) / n) * mask
    ic = (pc * yc).sum(dim=1) / ((pc.square().sum(dim=1) + EPS).sqrt() * (yc.square().sum(dim=1) + EPS).sqrt())
    usable = (mask.sum(dim=1) >= int(cfg.get("min_slots", 3))).to(torch.float32)
    mean_ic = (ic * usable).sum() / usable.sum().clamp_min(1.0)
    return -mean_ic, {"ic": mean_ic.detach()}


register_task_type("panel_classification", panel_classification_loss, ["accuracy"],
                   config_keys=("column", "label_smoothing"))
register_task_type("panel_regression", panel_regression_loss, ["mae"],
                   config_keys=("column", "loss_fn", "huber_delta"))
register_task_type("panel_rank_ic", panel_rank_loss, ["ic"], config_keys=("column", "min_slots"))
