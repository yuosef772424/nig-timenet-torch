"""SELECTION_METRICS: fixed validation metrics for checkpoint and early-stopping selection

PURPOSE:  SELECTION_METRICS registry: named, fixed validation metrics for checkpoint / early-stopping selection
          (val_raw_loss, per-head AUC and pinball).
TAGS:     selection metric, early stopping, checkpoint selection, validation, auc, pinball
PITFALLS: val_high_class / val_low_class AUC (val_auc_high_class, val_auc_low_class) are direction metrics,
          suspended by the owner (rule D): information only, never use them to select; the default is val_raw_loss.

SELECTION_METRICS: named, FIXED validation metrics for checkpoint / early-stopping selection (C7 second half;
BUILD_PLAN.md P22). A metric's definition never depends on loss weights or lambda schedules, EXCEPT the default
``val_raw_loss`` (the TF side's own default, workflow/settings.py: early_stopping_monitor), which is the
loss-weighted raw loss by construction.

Each entry's obj is ``fn(outputs, labels, logs) -> float`` (numpy/torch arrays; ``logs`` = eval_step's loss logs).
``val_ls_sharpe_close`` (D-032) is the per-timestamp Sharpe of the top / bottom 20 % book ranked by the close mean
(day metric 2 of ``eval/day_metrics.py`` on the validation rows; a cross-section run only).
Registered: val_raw_loss, and for every price target t in (high, low, close):
``val_auc_{t}_class`` (stats.metrics.auc of y_{t}_class vs the head's probability) and ``val_pinball_{t}_reg``
(stats.metrics.pinball_loss at quantile 0.5 of y_{t}_reg vs the NIG mean y_{t}; the NIG head predicts a mean, and
q=0.5 is the only quantile it gives without extra distributional assumptions).
"""

import numpy as np

from nig_timenet_torch.core.registry import Contract, Registry
from nig_timenet_torch.core.schema import TARGET_COLUMNS
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.stats import metrics

__all__ = ["SELECTION_METRICS", "compute_selection_metric", "check_selectable", "NOT_SELECTABLE", "GROUP_KEY"]

#: direction metrics of the high / low heads: suspended by the owner (never evidence, never used for selection); they
#: stay registered for reporting
NOT_SELECTABLE = ("val_auc_high_class", "val_auc_low_class")

SELECTION_METRICS = Registry("selection_metric", Contract("selection_metric",
                                                       required_meta=("higher_is_better", "selectable")))


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _val_raw_loss(outputs, labels, logs) -> float:
    return float(logs["raw_loss"])


def _auc_fn(t):
    spec = head_spec("binary_classification", t)
    return lambda outputs, labels, logs: metrics.auc(_np(labels[spec.label_key]),
                                                     _np(outputs[spec.output_keys["logits"]]))


def _pinball_fn(t):
    spec = head_spec("nig_regression", t)
    return lambda outputs, labels, logs: metrics.pinball_loss(_np(labels[spec.label_key]),
                                                              _np(outputs[spec.output_keys["mu"]]), 0.5)


#: label-dict key under which the trainer passes the rows' group (timestamp) ids to a selection metric
GROUP_KEY = "__group"


def _rank_ic_fn(targets):
    """Mean over ``targets`` of the rank IC of the NIG mean with its label, inside each timestamp when the trainer
    passes the groups (a cross-section run), else over all rows."""
    specs = [head_spec("nig_regression", t) for t in targets]

    def fn(outputs, labels, logs) -> float:
        groups = _np(labels[GROUP_KEY]) if GROUP_KEY in labels else None
        return float(np.mean([metrics.rank_ic(_np(labels[s.label_key]), _np(outputs[s.output_keys["mu"]]), groups)
                              for s in specs]))
    return fn


SELECTION_METRICS.register("val_rank_ic_reg", _rank_ic_fn(TARGET_COLUMNS), version=1, higher_is_better=True,
                           selectable=True)
for _t in TARGET_COLUMNS:
    SELECTION_METRICS.register(f"val_rank_ic_{_t}_reg", _rank_ic_fn((_t,)), version=1, higher_is_better=True,
                               selectable=True)


def _ls_sharpe_fn(target):
    """The market-neutral book of day metric 2 on the validation rows, in label units: per timestamp the top 20 % /
    bottom 20 % spread of the label ranked by the NIG mean (``stats.metrics.long_short_spreads``); the value is its
    mean over timestamps divided by its sd (a per-timestamp Sharpe). Needs the timestamps (a cross-section run)."""
    spec = head_spec("nig_regression", target)

    def fn(outputs, labels, logs) -> float:
        if GROUP_KEY not in labels:
            raise ValueError(f"val_ls_sharpe_{target} needs whole timestamps: run with SplitSettings.cross_section")
        _, v = metrics.long_short_spreads(_np(labels[spec.label_key]), _np(outputs[spec.output_keys["mu"]]),
                                          _np(labels[GROUP_KEY]))
        sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
        return float(v.mean() / sd) if sd > 0 else float("nan")
    return fn


SELECTION_METRICS.register("val_ls_sharpe_close", _ls_sharpe_fn("close"), version=1, higher_is_better=True,
                           selectable=True)
SELECTION_METRICS.register("val_raw_loss", _val_raw_loss, version=1, higher_is_better=False, selectable=True)
for _t in TARGET_COLUMNS:
    SELECTION_METRICS.register(f"val_auc_{_t}_class", _auc_fn(_t), version=1, higher_is_better=True,
                               selectable=f"val_auc_{_t}_class" not in NOT_SELECTABLE)
    SELECTION_METRICS.register(f"val_pinball_{_t}_reg", _pinball_fn(_t), version=1, higher_is_better=False,
                               selectable=True)


def check_selectable(name: str) -> None:
    """Raise UnknownNameError for an unregistered name and ValueError for a registered metric that must not select
    (the high / low direction AUCs: owner decision, direction is suspended)."""
    if not SELECTION_METRICS.entry(name).meta["selectable"]:
        raise ValueError(f"selection metric {name!r} is registered for reporting only: the direction of the high / low "
                         f"heads is suspended (never evidence, never used for selection). Selectable: "
                         f"{sorted(n for n in SELECTION_METRICS if SELECTION_METRICS.entry(n).meta['selectable'])}")


def compute_selection_metric(name: str, outputs, labels, logs) -> float:
    return float(SELECTION_METRICS[name](outputs, labels, logs))
