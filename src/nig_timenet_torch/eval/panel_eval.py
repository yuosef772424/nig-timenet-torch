"""Evaluation of a grid model (model/grid_panel.py): predictions and metrics stay (days, slots, K).

``predict_grid`` runs the model over day batches and returns ``{"logit", "mu", "mask"}`` as numpy grids.
``panel_report`` scores them against the target grids of data/panel.py: per target the mean daily rank IC of the
logit and of mu (cross-sectional, one Spearman per day, so a market-wide move cannot inflate it), the pooled AUC of
the logit against "beats the day centre", and the accuracy at 0.5. The pooled numbers read the real slots with a
boolean mask; no per-coin list is ever built before the model has answered.
"""

import numpy as np
import torch

from nig_timenet_torch.stats import metrics
from nig_timenet_torch.trainer.batching import n_rows, take, to_device

__all__ = ["predict_grid", "panel_report"]


def predict_grid(model: torch.nn.Module, inputs, *, batch_days: int = 8, device=None) -> dict[str, np.ndarray]:
    """The model's grids over every day of ``inputs`` (anything with ``len`` and ``[days]``)."""
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    parts: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, n_rows(inputs), batch_days):
            out = model(to_device(take(inputs, slice(start, start + batch_days)), dev))
            for k, v in out.items():
                parts.setdefault(k, []).append(v.detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


def panel_report(pred: dict[str, np.ndarray], target_grids: dict[str, np.ndarray], targets: tuple[str, ...]) -> dict:
    """``{"n_days", "n_rows", "<t>": {"ic_logit", "ic_mu", "auc", "accuracy"}}`` for each target in ``targets``."""
    mask = np.asarray(pred["mask"], dtype=bool)
    day = np.broadcast_to(np.arange(mask.shape[0])[:, None], mask.shape)[mask]
    out: dict = {"n_days": int(mask.any(axis=1).sum()), "n_rows": int(mask.sum())}
    for j, t in enumerate(targets):
        rank = np.asarray(target_grids["y_rank"], dtype=np.float64)[..., j][mask]
        cls = np.asarray(target_grids["y_cls"], dtype=np.float64)[..., j][mask]
        logit = np.asarray(pred["logit"], dtype=np.float64)[..., j][mask]
        mu = np.asarray(pred["mu"], dtype=np.float64)[..., j][mask]
        out[t] = {"ic_logit": metrics.rank_ic(rank, logit, day), "ic_mu": metrics.rank_ic(rank, mu, day),
                  "auc": metrics.auc(cls, logit),
                  "accuracy": float(((logit >= 0).astype(np.float64) == cls).mean()) if len(cls) else float("nan")}
    return out
