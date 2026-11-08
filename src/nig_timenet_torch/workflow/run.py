"""PURPOSE: The run steps of one flat single-split run (prepare, build, train, health report, test report, summary)
    and run_all() that calls them in order. Jobs and notebooks call the steps; nobody copies them.
TAGS: run steps, single split, flat run, train step, health step, test report step, run summary, notebook steps
PITFALLS: steps share one mutable Run (call them in order; a step that needs an earlier one raises); the test split is
    used only by test_step, never for training or selection; rolling and panel runs are package I4 (not here); the
    training data stay on the CPU and move to the device batch by batch (B-019).
SOURCE: ported from feat/ar-stream aa0f3dd (workflow/run.py); changes: single split only, steps over one Run object,
    health report and test report (D-005) as steps, outputs written to an output folder, the code of
    jobs/baseline.py moved here.

Every step is a function of a ``Run`` (the settings plus what the earlier steps produced).
``run_all(settings, out=...)`` is: prepare -> build -> train -> health_step -> test_step -> battery_step. Outputs in
``out``: ``test_report.json``, ``battery_report.txt``, ``battery_profit.json``, a copy of ``health_report.json`` /
``.md`` and a copy of the best weights (``best.pt`` + ``best_meta.json``, so a job publishes them with its results);
``write_summary`` adds ``baseline_summary.json``. The run folder keeps the originals and the resume checkpoint.
"""

import json
import os
import shutil
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX
from nig_timenet_torch.core.seeds import seed_everything
from nig_timenet_torch.data_io.loading import LoadedData
from nig_timenet_torch.eval.target_report import target_report
from nig_timenet_torch.model.architectures import ModelSettings
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.trainer import checkpoint
from nig_timenet_torch.trainer.checkpoint import BEST, BEST_META
from nig_timenet_torch.trainer.identity import RunIdentity, check_resume
from nig_timenet_torch.trainer.trainer import FitResult, FitSettings, Trainer
from nig_timenet_torch.workflow import flat_run, run_common
from nig_timenet_torch.workflow.battery_run import battery_step
from nig_timenet_torch.workflow.health_report import REPORT_JSON, REPORT_MD, HealthRun, health_report
from nig_timenet_torch.workflow.prediction_table import predict_table, table_meta
from nig_timenet_torch.workflow.settings import RunSettings

__all__ = ["Run", "prepare", "build", "train", "restore", "health_step", "test_step", "battery_step", "write_summary",
           "run_all", "test_lines", "SUMMARY", "TEST_REPORT"]

SUMMARY = "baseline_summary.json"
TEST_REPORT = "test_report.json"


@dataclass
class Run:
    settings: RunSettings
    art: LoadedData
    targets: tuple
    idx: dict                          # {"train" | "val" | "test": row indices of the artifact}
    bounds: tuple                      # (train_end, val_end)
    device: torch.device
    out: Path | None = None
    x: dict = field(default_factory=dict)       # {"train" | "val": inputs}
    y: dict = field(default_factory=dict)       # {"train" | "val": {label key: tensor}}
    ms: ModelSettings | None = None
    trainer: Trainer | None = None
    fit_settings: FitSettings | None = None
    ident: RunIdentity | None = None
    result: FitResult | None = None
    extra: dict = field(default_factory=dict)  # numbers later steps add to the summary

    @property
    def dataset(self) -> dict:
        return self.art.dataset

    def table(self, stage: str) -> pd.DataFrame:
        """PredictionTable of ALL rows of ``stage`` (the model holds the best weights after fit)."""
        _need(self, "result", "train")
        if stage not in self.x:
            self.x[stage], self.y[stage] = flat_run.flat_inputs(self.dataset, self.idx[stage])
        return flat_run.predict_stage(self.trainer, self.dataset, self.idx[stage], self.x[stage], self.ident,
                                      self.device, stage=stage, train_end=self.bounds[0],
                                      chunk=self.settings.evaluation.chunk)


def _need(run: Run, what: str, step: str) -> None:
    if getattr(run, what) is None:
        raise RuntimeError(f"call {step}() before this step (Run.{what} is not set)")


def prepare(settings: RunSettings, out=None) -> Run:
    """Load the artifact (rule D1), read its facts (they raise on a missing or unknown one before any work), split by
    time and build the train / validation inputs."""
    if out is not None:
        os.makedirs(out, exist_ok=True)
    art = run_common.load_data(settings)
    ds = art.dataset
    targets = run_common.price_targets(ds)
    run_common.check_decode(ds, targets)
    run_common.input_key(ds)
    (tr, va, te), bounds = run_common.make_plan(ds, settings.split)
    run = Run(settings, art, targets, {"train": tr, "val": va, "test": te}, bounds, run_common.pick_device(settings),
              None if out is None else Path(out))
    for stage in ("train", "val"):
        run.x[stage], run.y[stage] = flat_run.flat_inputs(ds, run.idx[stage])
    return run


def build(run: Run) -> Run:
    """Seed, build the model and the Trainer, then the run identity (AFTER the Trainer: the optimizer holds the Kendall
    group only once the trainer exists)."""
    s = run.settings
    seed_everything(s.train.seed)
    run.ms = flat_run.model_settings(s.model, run.targets)
    run.trainer = flat_run.make_trainer(s, run.ms, run.targets, run.x["train"].shape[1], run.x["train"].shape[2],
                                        run.device)
    run.fit_settings = flat_run.fit_settings_of(s)
    run.ident = run_common.run_identity(run.trainer, run.fit_settings, run.ms, run.art, s)
    return run


def train(run: Run, *, deadline: float | None = None, on_epoch=None) -> Run:
    """Fit (resumes a run folder that holds a checkpoint). ``deadline`` is a ``time.monotonic()`` value."""
    _need(run, "trainer", "build")
    n = {k: int(len(v)) for k, v in run.idx.items()}
    print(f"data: {n['train']} train / {n['val']} val / {n['test']} test windows "
          f"(max_rows {run.settings.split.max_rows}), split train_end {run.bounds[0]} val_end {run.bounds[1]}; "
          f"{-(-n['train'] // run.fit_settings.batch_size)} batches per epoch on {run.device}", flush=True)
    trace_batches = (flat_run.batch_set(run.dataset, run.idx["train"], run.fit_settings.batch_size,
                                        run.settings.train.seed)
                     if run.settings.train.trace_representative else None)
    run.result = run.trainer.fit(run.x["train"], run.y["train"], run.x["val"], run.y["val"], run.fit_settings,
                                 run_dir=run.settings.run_dir, identity=run.ident, deadline=deadline,
                                 on_epoch=on_epoch, trace_batches=trace_batches)
    _copy_best_weights(run)
    return run


def restore(run: Run, *, allow_changes: tuple[str, ...] = ("code_sha",)) -> Run:
    """Instead of ``train``: read a FINISHED run folder (``settings.run_dir``) and put its best-epoch weights in the
    model, so the evaluation steps re-read a trained model without training (T5r). The saved RunIdentity must equal
    this run's except for ``allow_changes`` (default: the commit; settings, seed and data must be the same)."""
    _need(run, "trainer", "build")
    run_dir = Path(run.settings.run_dir)
    ckpt = checkpoint.load_checkpoint(run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"{run_dir} holds no {checkpoint.CHECKPOINT}: nothing to restore")
    check_resume(RunIdentity.from_json(json.loads(ckpt["identity"])), run.ident,
                 allow_changes=("epochs", *allow_changes))
    tracker = ckpt["tracker"]
    if tracker.get("best_epoch") is None:
        raise ValueError(f"{run_dir}: the checkpoint records no best epoch")
    run.trainer.model.load_state_dict(checkpoint.load_best(run_dir))
    history = list(ckpt["history"])
    run.result = FitResult(history, int(ckpt["epochs_done"]), tracker["best_epoch"], tracker["best"],
                           len(history) < run.fit_settings.epochs, True)
    return run


def _copy_best_weights(run: Run) -> None:
    """``best.pt`` + ``best_meta.json`` of the run folder into ``out``, so a job publishes the weights with its
    results (B-050); skipped without ``out`` or before any best epoch, never copied onto itself."""
    if run.out is None:
        return
    for name in (BEST, BEST_META):
        src, dst = Path(run.settings.run_dir) / name, run.out / name
        if src.is_file() and not (dst.exists() and dst.samefile(src)):
            shutil.copyfile(src, dst)


def health_step(run: Run) -> Run:
    """Health report on an evenly strided subsample of the validation rows (they must fit on the device); written
    to the run folder and copied to ``out``."""
    _need(run, "result", "train")
    ds, dev = run.dataset, run.device
    val_idx = run.idx["val"]
    pick = run_common.stride_rows(np.arange(len(val_idx)), run.settings.evaluation.health_max_rows)
    idx = val_idx[pick]
    xs = run.x["val"][torch.as_tensor(pick)].to(dev)
    meta = table_meta(ds, idx, stage="val", fold=0, train_end=run.bounds[0])
    table = predict_table(run.trainer.model, xs, meta, run.ident)
    lc, codec = np.asarray(ds["last_candles"]), run_common.codec_state(ds)
    probe_target = "close" if "close" in run.targets else run.targets[0]
    labels = {k: v[torch.as_tensor(pick)].numpy() for k, v in run.y["val"].items()}
    hr = HealthRun(run_dir=Path(run.settings.run_dir), model=run.trainer.model, x=xs, predictions=table,
                   labels=labels, anchors=flat_run.anchors_by_name(lc, idx, run.targets, codec),
                   last_close=lc[idx, LAST_COLUMN_INDEX["last_close"]], codec_state=codec,
                   probe_labels=tuple(head_spec(k, probe_target).label_key
                                      for k in ("nig_regression", "binary_classification")))
    health_report(hr)
    run.extra["health_rows"] = int(len(idx))
    if run.out is not None:
        for name in (REPORT_JSON, REPORT_MD):
            shutil.copyfile(Path(run.settings.run_dir) / name, run.out / name)
    return run


def stage_report(run: Run, stage: str, table: pd.DataFrame) -> dict:
    lc, codec = np.asarray(run.dataset["last_candles"]), run_common.codec_state(run.dataset)
    idx = run.idx[stage]
    return target_report(table, {k: v.numpy() for k, v in run.y[stage].items()},
                         anchors=flat_run.anchors_by_name(lc, idx, run.targets, codec),
                         last_close=lc[idx, LAST_COLUMN_INDEX["last_close"]], codec_state=codec)


def test_step(run: Run) -> dict:
    """target_report on ALL test rows (D-005); writes ``test_report.json`` to ``out``. The test split is never used
    for training or selection. The test table is kept in ``run.extra['tables']`` for the battery."""
    if not len(run.idx["test"]):
        raise ValueError("the split has no test rows")
    table = run.table("test")
    run.extra.setdefault("tables", {})["test"] = table
    rep = stage_report(run, "test", table)
    if run.out is not None:
        (run.out / TEST_REPORT).write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    return rep


def _flat(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flat(v, f"{prefix}{k}.")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield f"{prefix}{k}", v


def test_lines(report: dict) -> list[str]:
    """Short printable summary of a test report: every scalar except the per-coin breakdowns."""
    return [f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}"
            for k, v in _flat(report) if "per_coin" not in k and "deciles" not in k]


def write_summary(run: Run, extra: dict | None = None) -> dict:
    """``baseline_summary.json`` in ``out``; ``extra`` (device, gpu name, ...) is merged in by the caller."""
    _need(run, "result", "train")
    s, res, fs, ds = run.settings, run.result, run.fit_settings, run.dataset
    secs = [r["trace_epoch_seconds"] for r in res.history if "trace_epoch_seconds" in r]  # epochs of earlier jobs too
    n_batches = -(-len(run.idx["train"]) // fs.batch_size)
    summary = {"seed": s.train.seed, "epochs_requested": fs.epochs, "epochs_done": res.epochs_done,
               "finished": res.epochs_done >= fs.epochs or res.stopped_early, "stopped_early": res.stopped_early,
               "best_epoch": res.best_epoch, "best_value": res.best_value,
               "step_seconds_per_batch": statistics.median(secs) / n_batches if secs else None,
               "epoch_seconds_median": statistics.median(secs) if secs else None, "batch_size": fs.batch_size,
               "n_windows": {k: int(len(v)) for k, v in run.idx.items()},
               "split": {"train_end": str(run.bounds[0]), "val_end": str(run.bounds[1])},
               "targets": list(run.targets), "label_smoothing": s.train.label_smoothing,
               "reg_target_mode": ds["reg_target_mode"], "reg_target_scale": float(ds["reg_target_scale"]),
               "input_key": run_common.input_key(ds), "max_rows": s.split.max_rows, "model_overrides": run.ms.overrides,
               "data_fingerprint": run.art.manifest.data_fingerprint, "identity_fingerprint": run.ident.fingerprint(),
               "run_dir": str(s.run_dir), "history": res.history,
               **{k: v for k, v in run.extra.items() if k != "tables"}, **(extra or {})}
    if run.out is not None:
        (run.out / SUMMARY).write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    return summary


def run_all(settings: RunSettings, out=None, *, deadline: float | None = None, on_epoch=None) -> Run:
    """prepare -> build -> train -> health report -> test report -> battery (when ``evaluation.battery.enabled``)."""
    r = build(prepare(settings, out))
    train(r, deadline=deadline, on_epoch=on_epoch)
    health_step(r)
    rep = test_step(r)
    print(f"test evaluation ({rep['n_rows']} rows, best-epoch weights):", flush=True)
    print("\n".join(test_lines(rep)), flush=True)
    battery_step(r)
    return r
