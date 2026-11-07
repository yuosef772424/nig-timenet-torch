"""PURPOSE: One global train/val/test split with purge by label end and embargo, plus the per-row times it needs
    (feature start, feature end, label end) derived from last_candles.
TAGS: split plan, purge embargo, label end, sample times, leakage check
PITFALLS: times come from the timestamp column of last_candles (core.schema.TS_COL, the OPEN time of the last input
    bar) and the dataset's base_tf / window_sizes / forecast_horizon; every cut time is inclusive.

SplitPlan: ONE global train/val/test split with purge + embargo, as a property of the data (C6; BUILD_PLAN.md
P25). Rolling splits are P37.

Per-sample times are int64 ns UTC instants. The pinned fixture does not carry them explicitly, so
``sample_times`` derives them from ``last_candles``' timestamp column ts (the OPEN time of the window's last bar,
as crypto-signal-prediction data/split.py sample_timestamps reads it), the base timeframe bar and
``forecast_horizon`` h: feature_start = ts - (window - 1) * bar (open of the first input bar, widest window over the
dataset's timeframes); feature_end = ts + bar (the last input bar has closed: the prediction can be made);
label_end = ts + (h + 1) * bar (the last bar the label depends on has closed: the label is known). The prediction
table's prediction_made_at / label_available_at (P27) are exactly feature_end / label_end.

Membership (all cut times inclusive as upstream's ``ts <= train_end``):
train = label_end <= train_end (purge built in: no train label reaches past the cut);
val = feature_start >= val_start, feature_start > train_end + embargo, label_end <= val_end;
test = feature_start >= test_start, feature_start > val_end + embargo.
calibration = val (the TF default ``calibration_source="val"``: val is both SELECTION and CALIBRATION).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nig_timenet_torch.core.roles import Role
from nig_timenet_torch.core.schema import TS_COL

__all__ = ["SplitPlan", "LeakageError", "purge_embargo_split", "check_split_purity", "sample_times", "TS_COL"]


class LeakageError(ValueError):
    """A later split's features start at or before an earlier split's last label."""


def _ns(t) -> int:
    ts = pd.Timestamp(t)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return int(ts.value)


@dataclass(frozen=True)
class SplitPlan:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    calibration_idx: np.ndarray
    label_end: np.ndarray
    feature_end: np.ndarray
    role_of: dict[Role, np.ndarray]
    feature_start: np.ndarray = field(default=None)  # needed by the purity check; not in P25's interface list


def sample_times(dataset: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(feature_start, feature_end, label_end) int64 ns per row, derived as in the module docstring."""
    ts = np.asarray(dataset["last_candles"])[:, TS_COL].astype(np.int64)
    bar = int(pd.Timedelta(dataset["base_tf"]).value)
    span = max(int(dataset["window_sizes"][tf]) * int(pd.Timedelta(tf).value) for tf in dataset["timeframes"])
    feature_start = ts - (span - bar)
    label_end = ts + (int(dataset["forecast_horizon"]) + 1) * bar
    return feature_start, ts + bar, label_end


def purge_embargo_split(label_end: np.ndarray, feature_start: np.ndarray, train_end, val_start, val_end,
                        test_start, embargo: "pd.Timedelta", *, feature_end: np.ndarray | None = None) -> SplitPlan:
    """Global time split (rules in the module docstring); runs check_split_purity before returning.
    ``feature_end`` is only stored (sample_times gives it); if omitted, label_end is stored as its upper bound."""
    label_end = np.asarray(label_end, dtype=np.int64)
    feature_start = np.asarray(feature_start, dtype=np.int64)
    if label_end.shape != feature_start.shape:
        raise ValueError("label_end and feature_start must have the same shape")
    gap = int(pd.Timedelta(embargo).value)
    if gap < 0:
        raise ValueError("embargo must be >= 0")
    tr_end, va_start, va_end, te_start = (_ns(t) for t in (train_end, val_start, val_end, test_start))
    if not tr_end <= va_end <= te_start or va_start > va_end:
        raise ValueError("need train_end <= val_end <= test_start and val_start <= val_end")
    train = np.flatnonzero(label_end <= tr_end)
    val = np.flatnonzero((feature_start >= va_start) & (feature_start > tr_end + gap) & (label_end <= va_end))
    test = np.flatnonzero((feature_start >= te_start) & (feature_start > va_end + gap))
    plan = SplitPlan(train_idx=train, val_idx=val, test_idx=test, calibration_idx=val.copy(), label_end=label_end,
                     feature_end=label_end.copy() if feature_end is None else np.asarray(feature_end, np.int64),
                     role_of={Role.FIT: train, Role.CALIBRATION: val.copy(), Role.SELECTION: val, Role.REPORT: test,
                              Role.SEALED: np.array([], dtype=np.int64)},
                     feature_start=feature_start)
    check_split_purity(plan)
    return plan


def check_split_purity(plan: SplitPlan) -> None:
    """Raises LeakageError unless max(train.label_end) < min(val.feature_start) and likewise val -> test, and the
    FIT role is disjoint from CALIBRATION / SELECTION / REPORT (empty splits are trivially pure)."""
    order = [("train", plan.train_idx), ("val", plan.val_idx), ("test", plan.test_idx)]
    for (a, ia), (b, ib) in zip(order, order[1:], strict=False):
        if len(ia) and len(ib) and plan.label_end[ia].max() >= plan.feature_start[ib].min():
            raise LeakageError(f"{a} -> {b} leak: max({a}.label_end)={plan.label_end[ia].max()} >= "
                               f"min({b}.feature_start)={plan.feature_start[ib].min()}")
    fit = set(plan.role_of[Role.FIT].tolist())
    for role in (Role.CALIBRATION, Role.SELECTION, Role.REPORT):
        overlap = fit & set(plan.role_of[role].tolist())
        if overlap:
            raise LeakageError(f"{len(overlap)} rows are both FIT and {role.name}")
