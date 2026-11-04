"""One call from settings to results: ``run(settings)`` (a single split) or the rolling study when
``settings.rolling`` is set, for the flat model and the cross-asset panel model alike.

Both go through the same Trainer. A rolling study trains every fold with ``trainer.rolling_fit.fit_fold`` (its own
run folder, identity and seed; fresh or warm start) on the fold's train window, selects the best epoch on the fold's
validation tail (``rolling.val_span``) and scores the fold's test window; the retargeting of ``settings.targets`` is
fitted per fold on that fold's train rows only. ``retrain`` is the retarget-and-train-again path.
"""

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.config import to_dict, updated
from nig_timenet_torch.core.seeds import seed_everything
from nig_timenet_torch.data_io.loading import LoadedData
from nig_timenet_torch.trainer.identity import RunIdentity
from nig_timenet_torch.trainer.rolling_fit import fit_fold
from nig_timenet_torch.trainer.trainer import FitResult, Trainer
from nig_timenet_torch.workflow import flat_run, panel_run
from nig_timenet_torch.workflow.rolling import RollingFold, rolling_folds, run_rolling_study
from nig_timenet_torch.workflow.run_common import apply_targets, load_data, make_plan, pick_device, run_identity
from nig_timenet_torch.workflow.settings import RunSettings

__all__ = ["RunResult", "run", "retrain", "scalars", "rolling_summary"]

SUMMARY = "run_summary.json"
TEST_TABLE = "test_predictions.parquet"


@dataclass
class RunResult:
    settings: RunSettings
    loaded: LoadedData
    trainer: Trainer | None = None              # single split: the trained model; rolling: the last fold's
    fit: FitResult | None = None
    report: dict = field(default_factory=dict)  # single split: the test report
    table: pd.DataFrame | None = None           # single split: the test PredictionTable
    folds: list[dict] = field(default_factory=list)      # rolling: one entry per (fold, seed)
    tables: list[pd.DataFrame] = field(default_factory=list)   # rolling: one table per (fold, seed)
    target_report: dict = field(default_factory=dict)


def scalars(report: dict, prefix: str = "") -> dict[str, float]:
    """Every numeric leaf of a nested report as ``a.b.c -> value`` (per-coin and decile breakdowns left out)."""
    out = {}
    for k, v in report.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(scalars(v, key + "."))
        elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            out[key] = float(v)
    return {k: v for k, v in out.items() if "per_coin" not in k and "deciles" not in k}


def _epoch_printer(settings: RunSettings, label: str = "") -> Callable[[dict], None] | None:
    if not settings.train.verbose:
        return None

    def show(row: dict) -> None:
        val = row.get("val_total_loss")
        print(f"{label}epoch {row['epoch']}/{settings.train.epochs}  train_loss {row.get('train_total_loss', 0):.4f}  "
              f"val_loss {'-' if val is None else format(val, '.4f')}  "
              f"val_selection {row.get('val_selection_metric', float('nan')):.4f}", flush=True)
    return show


def _model_identity(settings: RunSettings):
    """What describes the model in the run identity: the flat model's settings, or the panel net and its grid."""
    if settings.kind == "flat":
        return settings.model
    return {"net": to_dict(settings.panel_net), "panel": to_dict(settings.panel), "loss": to_dict(settings.panel_loss)}


# ------------------------------------------------------------------------------------------------ one split
def _flat_fit(settings, dataset, plan, loaded, device, run_dir, on_epoch):
    seed_everything(settings.train.seed)
    trainer = flat_run.make_flat_trainer(settings, dataset, device)
    fit_settings = flat_run.fit_settings_of(settings)
    ident = run_identity(trainer, fit_settings, _model_identity(settings), loaded, dataset, settings)
    data = (*flat_run.flat_inputs(dataset, plan.train_idx), *flat_run.flat_inputs(dataset, plan.val_idx))
    res = trainer.fit(data[0], data[1], data[2], data[3], fit_settings, run_dir=run_dir, identity=ident,
                      on_epoch=on_epoch)
    return trainer, res, ident


def _panel_fit(settings, dataset, plan, loaded, device, run_dir, on_epoch):
    seed_everything(settings.train.seed)
    tr = panel_run.panel_data(dataset, plan.train_idx, settings)
    va = panel_run.panel_data(dataset, plan.val_idx, settings, reg_scale=tr.grid.reg_scale)
    trainer = panel_run.make_panel_trainer(settings, dataset, device)
    fit_settings = flat_run.fit_settings_of(settings)
    ident = run_identity(trainer, fit_settings, _model_identity(settings), loaded, dataset, settings)
    res = trainer.fit(tr.inputs, tr.y, va.inputs, va.y, fit_settings, run_dir=run_dir, identity=ident,
                      on_epoch=on_epoch)
    return trainer, res, ident, tr.grid.reg_scale


def _score(settings, trainer, dataset, idx, ident, stage, fold, train_end, reg_scale=None):
    """(table, report) of the rows ``idx`` for the run's kind."""
    if settings.kind == "flat":
        table = flat_run.flat_table(trainer, dataset, idx, ident, settings, stage=stage, fold=fold,
                                    train_end=train_end)
        return table, flat_run.flat_report(table, dataset, idx)
    data = panel_run.panel_data(dataset, idx, settings, reg_scale=reg_scale)
    pred, report = panel_run.panel_scores(trainer, data, settings)
    return panel_run.panel_table(pred, data, dataset, ident, stage=stage, fold=fold, train_end=train_end), report


def _limit(idx: np.ndarray, max_rows: int) -> np.ndarray:
    if not max_rows or len(idx) <= max_rows:
        return idx
    return idx[np.unique(np.linspace(0, len(idx) - 1, max_rows).astype(np.int64))]


def run_single(settings: RunSettings, loaded: LoadedData, on_epoch=None) -> RunResult:
    plan = make_plan(loaded.dataset, settings)
    if not len(plan.test_idx):
        raise ValueError("the split has no test rows")
    dataset, target_report = apply_targets(loaded.dataset, settings, plan.train_idx)
    device = pick_device(settings)
    run_dir = settings.run_dir or None
    on_epoch = on_epoch or _epoch_printer(settings)
    scale = None
    if settings.kind == "flat":
        trainer, res, ident = _flat_fit(settings, dataset, plan, loaded, device, run_dir, on_epoch)
    else:
        trainer, res, ident, scale = _panel_fit(settings, dataset, plan, loaded, device, run_dir, on_epoch)
    train_end = pd.Timestamp(int(np.asarray(dataset["last_candles"])[plan.train_idx, 3].max()), tz="UTC")
    table, report = _score(settings, trainer, dataset, _limit(plan.test_idx, settings.evaluation.max_rows), ident,
                           "test", 0, train_end, scale)
    result = RunResult(settings, loaded, trainer, res, report, table, target_report=target_report)
    if run_dir:
        out = Path(run_dir)
        out.mkdir(parents=True, exist_ok=True)
        table.to_parquet(out / TEST_TABLE, index=False)
        summary = {"kind": settings.kind, "epochs_done": res.epochs_done, "best_epoch": res.best_epoch,
                   "best_value": res.best_value, "stopped_early": res.stopped_early,
                   "n_windows": {"train": len(plan.train_idx), "val": len(plan.val_idx), "test": len(plan.test_idx)},
                   "identity": ident.fingerprint(), "target": target_report, "test": report}
        (out / SUMMARY).write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------------------------- rolling
@dataclass
class _FoldModel:
    trainer: Trainer
    dataset: dict
    reg_scale: np.ndarray | None
    fit: FitResult


def _fold_train(settings, loaded, device, previous, on_epoch):
    r = settings.rolling

    def train_fn(dataset, fold: RollingFold, identity: RunIdentity) -> _FoldModel:
        if not len(fold.plan.val_idx):
            raise ValueError("a rolling fold needs a validation tail: set rolling.val_span > 0")
        ds, _ = apply_targets(dataset, settings, fold.plan.train_idx)
        fit_settings = flat_run.fit_settings_of(settings)
        if settings.kind == "flat":
            data = (*flat_run.flat_inputs(ds, fold.plan.train_idx), *flat_run.flat_inputs(ds, fold.plan.val_idx))
            scale = None
            make = lambda seed: flat_run.make_flat_trainer(settings, ds, device)  # noqa: E731
        else:
            tr = panel_run.panel_data(ds, fold.plan.train_idx, settings)
            va = panel_run.panel_data(ds, fold.plan.val_idx, settings, reg_scale=tr.grid.reg_scale)
            data, scale = (tr.inputs, tr.y, va.inputs, va.y), tr.grid.reg_scale
            make = lambda seed: panel_run.make_panel_trainer(settings, ds, device)  # noqa: E731
        trainer, res = fit_fold(make, data, fit_settings, identity, previous=previous.get(identity.seed),
                                init=r.init, warm_epochs=r.warm_epochs, run_dir=settings.run_dir or None,
                                on_epoch=on_epoch)
        previous[identity.seed] = trainer
        return _FoldModel(trainer, ds, scale, res)
    return train_fn


def run_rolling(settings: RunSettings, loaded: LoadedData, on_epoch=None) -> RunResult:
    device = pick_device(settings)
    on_epoch = on_epoch or _epoch_printer(settings)
    dataset = loaded.dataset
    folds_info: list[dict] = []
    fold_of = {f.index: f for f in rolling_folds(dataset, settings.rolling)}

    def predict_fn(fm: _FoldModel, ds, fold: RollingFold, identity: RunIdentity) -> pd.DataFrame:
        idx = _limit(fold.plan.test_idx, settings.evaluation.max_rows)
        table, report = _score(settings, fm.trainer, fm.dataset, idx, identity, "test", fold.index, fold.train_end,
                               fm.reg_scale)
        folds_info.append({"fold": fold.index, "seed": identity.seed, "train_end": str(fold.train_end),
                           "n_train": len(fold.plan.train_idx), "n_val": len(fold.plan.val_idx), "n_test": len(idx),
                           "best_epoch": fm.fit.best_epoch, "epochs_done": fm.fit.epochs_done, "report": report})
        return table

    seed_everything(settings.train.seed)
    probe = (flat_run.make_flat_trainer if settings.kind == "flat" else panel_run.make_panel_trainer)(
        settings, dataset, device)
    parent = run_identity(probe, flat_run.fit_settings_of(settings), _model_identity(settings), loaded, dataset,
                          settings)
    out_dir = None
    if settings.run_dir:
        out_dir = Path(settings.run_dir) / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
    previous: dict[int, Trainer] = {}
    tables = run_rolling_study(dataset, settings.rolling, _fold_train(settings, loaded, device, previous, on_epoch),
                               predict_fn, parent, out_dir=out_dir)
    last = previous.get(settings.rolling.seeds[-1])
    result = RunResult(settings, loaded, last, None, folds=folds_info, tables=tables)
    if settings.run_dir:
        (Path(settings.run_dir) / SUMMARY).write_text(
            json.dumps({"kind": settings.kind, "folds": folds_info, "n_folds": len(fold_of)}, indent=1, default=float),
            encoding="utf-8")
    return result


def rolling_summary(result: RunResult) -> pd.DataFrame:
    """One row per (fold, seed): sizes, best epoch and every scalar metric of its test report."""
    rows = [{**{k: v for k, v in f.items() if k != "report"}, **scalars(f["report"])} for f in result.folds]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ entry points
def run(settings: RunSettings, loaded: LoadedData | None = None, on_epoch=None) -> RunResult:
    """Load (unless given), then a single split or, when ``settings.rolling`` is set, the rolling study."""
    loaded = loaded or load_data(settings)
    return (run_rolling if settings.rolling is not None else run_single)(settings, loaded, on_epoch)


def retrain(previous: RunResult, changes: dict | None = None, **overrides) -> RunResult:
    """Run again on the same loaded data with changed settings, e.g. ``retrain(r, {"targets.mode": "scaled"})``
    (a new target normalization), a different ``train.seed``, or another model section."""
    settings = updated(previous.settings, changes or {}, **overrides)
    return run(dataclasses.replace(settings), previous.loaded)
