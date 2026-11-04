"""SELECTION_METRICS: named, FIXED validation metrics for checkpoint / early-stopping selection (C7 second half;
BUILD_PLAN.md P22). A metric's definition never depends on loss weights or lambda schedules, EXCEPT the default
``val_raw_loss`` (the TF side's own default, workflow/settings.py: early_stopping_monitor), which is the
loss-weighted raw loss by construction.

Each entry's obj is ``fn(outputs, labels, logs) -> float`` (numpy/torch arrays; ``logs`` = eval_step's loss logs).
Registered: val_raw_loss, and for every price target t in (high, low, close):
``val_auc_{t}_class`` (stats.metrics.auc of y_{t}_class vs the head's probability) and ``val_pinball_{t}_reg``
(stats.metrics.pinball_loss at quantile 0.5 of y_{t}_reg vs the NIG mean y_{t}; the NIG head predicts a mean, and
q=0.5 is the only quantile it gives without extra distributional assumptions).
"""

import numpy as np

from nig_timenet_torch.core.registry import Contract, Registry
from nig_timenet_torch.stats import metrics

__all__ = ["SELECTION_METRICS", "compute_selection_metric"]

SELECTION_METRICS = Registry("selection_metric", Contract("selection_metric", required_meta=("higher_is_better",)))


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _val_raw_loss(outputs, labels, logs) -> float:
    return float(logs["raw_loss"])


def _auc_fn(t):
    return lambda outputs, labels, logs: metrics.auc(_np(labels[f"y_{t}_class"]), _np(outputs[f"y_{t}_class_logits"]))


def _pinball_fn(t):
    return lambda outputs, labels, logs: metrics.pinball_loss(_np(labels[f"y_{t}_reg"]), _np(outputs[f"y_{t}"]), 0.5)


def _panel_ic_fn(column: int):
    """Mean over days of the within-day Spearman between the head's logit and the day's relative-return rank target;
    ``outputs["logit"]``, ``labels["y_rank"]`` and ``outputs["mask"]`` are the (days, slots, K) grids of the panel."""
    def fn(outputs, labels, logs):
        mask = np.asarray(outputs["mask"].detach().cpu().numpy() if hasattr(outputs["mask"], "detach")
                          else outputs["mask"], dtype=bool)
        pred = np.asarray(outputs["logit"].detach().cpu().numpy(), dtype=np.float64)[..., column]
        rank = np.asarray(labels["y_rank"].detach().cpu().numpy(), dtype=np.float64)[..., column]
        day = np.broadcast_to(np.arange(mask.shape[0])[:, None], mask.shape)
        return metrics.rank_ic(rank[mask], pred[mask], day[mask])
    return fn


SELECTION_METRICS.register("val_raw_loss", _val_raw_loss, version=1, higher_is_better=False)
for _t in ("high", "low", "close"):
    SELECTION_METRICS.register(f"val_auc_{_t}_class", _auc_fn(_t), version=1, higher_is_better=True)
    SELECTION_METRICS.register(f"val_pinball_{_t}_reg", _pinball_fn(_t), version=1, higher_is_better=False)
for _i, _t in enumerate(("high", "low", "close")):
    SELECTION_METRICS.register(f"val_panel_ic_{_t}", _panel_ic_fn(_i), version=1, higher_is_better=True)


def compute_selection_metric(name: str, outputs, labels, logs) -> float:
    return float(SELECTION_METRICS[name](outputs, labels, logs))
