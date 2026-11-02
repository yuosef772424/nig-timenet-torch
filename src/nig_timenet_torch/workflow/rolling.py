"""P37: rolling-retrain loop — SEAM ONLY (the shape of the loop; no experiment is run through it in this repo).

Fold i (origin = earliest feature_start; cut_i = origin + train_span + i * step):
  train = samples with label_end <= cut_i   (expanding: feature_start >= origin; sliding: >= cut_i - train_span)
  test  = samples with feature_start > cut_i + embargo and label_end <= cut_i + embargo + test_span
Each fold is built through P25's purge_embargo_split (empty val) and re-checked with check_split_purity after the
window bounds are applied. Folds stop when a test window would pass the last label, or at max_folds.
"""

import dataclasses
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.roles import Role
from nig_timenet_torch.data_io.split_plan import SplitPlan, check_split_purity, purge_embargo_split, sample_times
from nig_timenet_torch.trainer.identity import RunIdentity
from nig_timenet_torch.workflow.prediction_table import write_table
from nig_timenet_torch.workflow.settings import RollingSettings

__all__ = ["RollingFold", "rolling_folds", "rolling_folds_from_times", "run_rolling_study"]


@dataclasses.dataclass(frozen=True)
class RollingFold:
    index: int
    plan: SplitPlan
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _restrict(plan: SplitPlan, train: np.ndarray, test: np.ndarray) -> SplitPlan:
    empty = np.array([], dtype=np.int64)
    out = dataclasses.replace(plan, train_idx=train, val_idx=empty, test_idx=test, calibration_idx=empty,
                              role_of={Role.FIT: train, Role.CALIBRATION: empty, Role.SELECTION: empty,
                                       Role.REPORT: test, Role.SEALED: empty})
    check_split_purity(out)
    return out


def rolling_folds_from_times(feature_start: np.ndarray, feature_end: np.ndarray, label_end: np.ndarray,
                             settings: RollingSettings) -> list[RollingFold]:
    fs, le = np.asarray(feature_start, np.int64), np.asarray(label_end, np.int64)
    origin, last = pd.Timestamp(int(fs.min())), pd.Timestamp(int(le.max()))
    folds: list[RollingFold] = []
    i = 0
    while settings.max_folds is None or i < settings.max_folds:
        cut = origin + settings.train_span + i * settings.step
        test_start, test_end = cut + settings.embargo, cut + settings.embargo + settings.test_span
        if test_end > last:
            break
        base = purge_embargo_split(le, fs, cut, cut, cut, cut, settings.embargo, feature_end=feature_end)
        train_start = origin if settings.mode == "expanding" else cut - settings.train_span
        train = base.train_idx[fs[base.train_idx] >= train_start.value]
        test = base.test_idx[le[base.test_idx] <= test_end.value]
        folds.append(RollingFold(i, _restrict(base, train, test), train_start, cut, test_start, test_end))
        i += 1
    return folds


def rolling_folds(dataset, settings: RollingSettings) -> list[RollingFold]:
    """Folds over a P06 dataset dict (times from data_io.split_plan.sample_times). Each fold's ``plan`` is a P25
    SplitPlan (train = FIT, test = REPORT, empty val/calibration)."""
    fs, fe, le = sample_times(dataset)
    return rolling_folds_from_times(fs, fe, le, settings)


def run_rolling_study(dataset, settings: RollingSettings, train_fn: Callable, predict_fn: Callable,
                      parent_identity: RunIdentity, *, out_dir: str | Path | None = None) -> list[pd.DataFrame]:
    """Loops ``rolling_folds`` x ``settings.seeds``. Fold i / seed s gets its own RunIdentity(fold=i, seed=s,
    data_cut=train_end ISO, parent_hashes=(parent.settings_hash,)); ``train_fn(dataset, fold, identity)`` returns a
    model (called only when i % retrain_every_k == 0, else the seed's last model is reused);
    ``predict_fn(model, dataset, fold, identity)`` returns a PredictionTable, written to
    ``out_dir/fold_{i:03d}_seed_{s}.parquet`` when out_dir is given. NOT called with real training by any test
    (that would be the experiment E-roll-001, out of scope)."""
    tables: list[pd.DataFrame] = []
    last_model: dict[int, object] = {}
    for fold in rolling_folds(dataset, settings):
        for s in settings.seeds:
            identity = dataclasses.replace(parent_identity, seed=s, fold=fold.index,
                                           data_cut=fold.train_end.isoformat(),
                                           parent_hashes=(parent_identity.settings_hash,))
            if fold.index % settings.retrain_every_k == 0 or s not in last_model:
                last_model[s] = train_fn(dataset, fold, identity)
            table = predict_fn(last_model[s], dataset, fold, identity)
            if out_dir is not None and table is not None:
                write_table(Path(out_dir) / f"fold_{fold.index:03d}_seed_{s}.parquet", table, identity)
            tables.append(table)
    return tables
