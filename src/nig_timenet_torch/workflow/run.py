"""PURPOSE: The run steps of one flat single-split run (prepare, build, train, health report, test report, summary)
    and run_all() that calls them in order. Jobs and notebooks call the steps; nobody copies them.
TAGS: run steps, single split, flat run, train step, health step, test report step, run summary, notebook steps
PITFALLS: steps share one mutable Run (call them in order; a step that needs an earlier one raises); the test split is
    used only by test_step, never for training or selection; a rolling run is one Run per fold
    (workflow/rolling_run.py), the panel run is parked; the training data stay on the CPU and move to the device
    batch by batch (B-019). ``train_staged`` (ST1) shares the Run, the Trainer and the later steps with ``train``.
SOURCE: ported from feat/ar-stream aa0f3dd (workflow/run.py); changes: single split only, steps over one Run object,
    health report and test report (D-005) as steps, outputs written to an output folder, the code of
    jobs/baseline.py moved here; staged training (ST1) and per-fold runs (F2) added.

Every step is a function of a ``Run`` (the settings plus what the earlier steps produced).
``run_all(settings, out=...)`` is: prepare -> build -> train (``train_staged`` when ``train.staged`` is set) ->
health_step -> test_step -> battery_step. Outputs in
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
from nig_timenet_torch.model.label_scale import fit_label_scale
from nig_timenet_torch.model.scalers import fit_scalers
from nig_timenet_torch.model.target_norm import fit_target_norm
from nig_timenet_torch.trainer import checkpoint
from nig_timenet_torch.trainer.checkpoint import BEST, BEST_META
from nig_timenet_torch.trainer.identity import RunIdentity, check_resume
from nig_timenet_torch.trainer.staged import StagedResult, plan_stages, staged_fit
from nig_timenet_torch.trainer.trainer import FitResult, FitSettings, Trainer
from nig_timenet_torch.workflow import flat_run, run_common, staged_report
from nig_timenet_torch.workflow.battery_run import battery_step
from nig_timenet_torch.workflow.health_report import REPORT_JSON, REPORT_MD, HealthRun, health_report
from nig_timenet_torch.workflow.prediction_table import predict_table, table_meta
from nig_timenet_torch.workflow.settings import RunSettings

__all__ = ["Run", "prepare", "prepare_split", "build", "train", "train_staged", "restore", "health_step", "test_step",
           "battery_step", "write_summary", "run_all", "test_lines", "SUMMARY", "TEST_REPORT"]

RESTORE_WEIGHTS = ("best", "last")  # restore(weights=...): best-epoch weights (default) or the last finished epoch
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
    fold: int = 0                      # the fold of a rolling study (0 = a single split); written into the tables
    x: dict = field(default_factory=dict)       # {"train" | "val": inputs}
    y: dict = field(default_factory=dict)       # {"train" | "val": {label key: tensor}}
    ms: ModelSettings | None = None
    trainer: Trainer | None = None
    fit_settings: FitSettings | None = None
    ident: RunIdentity | None = None
    result: FitResult | None = None
    staged: StagedResult | None = None   # ST1: what train_staged produced (the baselines read its steps and stages)
    plan: list | None = None             # ST1: the StageSpec list train_staged ran
    extra: dict = field(default_factory=dict)  # numbers later steps add to the summary

    @property
    def dataset(self) -> dict:
        return self.art.dataset

    def table(self, stage: str) -> pd.DataFrame:
        """PredictionTable of ALL rows of ``stage`` (the model holds the best weights after fit)."""
        _need(self, "result", "train")
        if stage not in self.x:
            self.x[stage], self.y[stage] = flat_run.flat_inputs(self.dataset, self.idx[stage],
                                                                self.settings.split.cross_section)
        return flat_run.predict_stage(self.trainer, self.dataset, self.idx[stage], self.x[stage], self.ident,
                                      self.device, stage=stage, train_end=self.bounds[0],
                                      chunk=self.settings.evaluation.chunk, fold=self.fold)


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
    parts, bounds = run_common.make_plan(ds, settings.split)
    if settings.split.exclude_coins:
        names = ", ".join(settings.split.exclude_coins)
        print(f"excluded coins ({len(settings.split.exclude_coins)}): {names}", flush=True)
    return _assemble(settings, art, targets, parts, bounds, out)


def prepare_split(settings: RunSettings, art: LoadedData, parts: tuple, bounds: tuple, out=None) -> Run:
    """A Run over row sets that are already split, of an artifact that is already loaded and checked (a rolling fold,
    ``workflow/rolling_run.py``): ``parts`` = (train, val, test) rows BEFORE the settings' coin exclusions and smoke
    stride, which are applied here exactly as ``prepare`` does; ``bounds`` = (train_end, val_end) of the rows."""
    if out is not None:
        os.makedirs(out, exist_ok=True)
    parts = run_common.select_rows(art.dataset, settings.split, parts)
    return _assemble(settings, art, run_common.price_targets(art.dataset), parts, bounds, out)


def _assemble(settings: RunSettings, art: LoadedData, targets: tuple, parts: tuple, bounds: tuple, out) -> Run:
    ds = art.dataset
    run = Run(settings, art, targets, dict(zip(("train", "val", "test"), parts, strict=True)), bounds,
              run_common.pick_device(settings), None if out is None else Path(out))
    for stage in ("train", "val"):
        run.x[stage], run.y[stage] = flat_run.flat_inputs(ds, run.idx[stage], settings.split.cross_section)
    return run


def build(run: Run) -> Run:
    """Seed, build the model and the Trainer, then the run identity (AFTER the Trainer: the optimizer holds the Kendall
    group only once the trainer exists)."""
    s = run.settings
    seed_everything(s.train.seed)
    run.ms = flat_run.model_settings(s.model, run.targets)
    run.trainer = flat_run.make_trainer(s, run.ms, run.targets, run.x["train"].shape[1], run.x["train"].shape[2],
                                        run.device)
    tn_state = fit_target_norm(run.trainer.model, run.y["train"])   # TRAIN labels only; None = the model has none
    if tn_state is not None:
        run.extra["target_norm"] = tn_state
    fit_label_scale(run.trainer.model, run.y["train"])   # FX2: a fitted NIG label scale learns from TRAIN only
    # train-fitted state of the model (input scaler, NIG label scale): TRAIN rows only; a resume / restore then
    # loads the same values from the checkpoint. Empty for a model without such modules.
    x_fit = run.x["train"][..., :-1] if s.split.cross_section else run.x["train"]   # without the timestamp channel
    run.extra["fitted_scalers"] = fit_scalers(run.trainer.model, x_fit, run.y["train"])
    run.fit_settings = flat_run.fit_settings_of(s)
    run.ident = run_common.run_identity(run.trainer, run.fit_settings, run.ms, run.art, s)
    return run


def train(run: Run, *, deadline: float | None = None, on_epoch=None, batch_monitor=None) -> Run:
    """Fit (resumes a run folder that holds a checkpoint). ``deadline`` is a ``time.monotonic()`` value;
    ``batch_monitor``: a ``trainer.batch_influence.BatchInfluence`` (log-only; None = none)."""
    _need(run, "trainer", "build")
    n = {k: int(len(v)) for k, v in run.idx.items()}
    groups = flat_run.batch_groups(run.dataset, run.idx["train"], run.settings.train.batch_by)
    n_batches = -(-n['train'] // run.fit_settings.batch_size) if groups is None else len(torch.unique(groups))
    print(f"data: {n['train']} train / {n['val']} val / {n['test']} test windows "
          f"(max_rows {run.settings.split.max_rows}), split train_end {run.bounds[0]} val_end {run.bounds[1]}; "
          f"{n_batches} batches per epoch (batch_by {run.settings.train.batch_by}) on {run.device}", flush=True)
    trace_batches = (flat_run.batch_set(run.dataset, run.idx["train"], run.fit_settings.batch_size,
                                        run.settings.train.seed)
                     if run.settings.train.trace_representative else None)
    run.result = run.trainer.fit(run.x["train"], run.y["train"], run.x["val"], run.y["val"], run.fit_settings,
                                 run_dir=run.settings.run_dir, identity=run.ident, deadline=deadline,
                                 on_epoch=on_epoch, trace_batches=trace_batches, batch_monitor=batch_monitor,
                                 batch_groups=groups)
    _copy_best_weights(run)
    return run


def train_staged(run: Run, *, deadline: float | None = None, on_eval=None, stages=None) -> Run:
    """Staged training (ROADMAP ST1, card F) through the one Trainer: ``settings.train.staged`` says how the rows grow
    and the lr falls (``trainer.staged``); resumes a run folder that holds a checkpoint. ``stages``: a ready plan
    (the equal-step baselines of ``workflow.staged_study``) instead of the staged arm's. The per-stage table lands in
    ``run.extra['stages']``; ``run.result`` is a FitResult over the evaluation points so the later steps read it as
    before. The scalers were fitted once on the whole training split by ``build``."""
    _need(run, "trainer", "build")
    st = run.settings.train.staged
    if st is None:
        raise ValueError("train_staged needs settings.train.staged (a StageSettings)")
    fs = run.fit_settings
    plan = stages or plan_stages(st, staged_report.growth_context(run.dataset, run.idx["train"], fs.seed),
                                 fs.batch_size, run.trainer.optimizer.param_groups[0]["lr"])
    print(f"staged training: {len(plan)} stage(s), rows {[len(s.rows) for s in plan]}, lr "
          f"{[round(s.lr, 8) for s in plan]}; {len(run.idx['val'])} val rows (at most {st.val_max_rows} used)",
          flush=True)
    res = staged_fit(run.trainer, run.x["train"], run.y["train"], run.x["val"], run.y["val"], plan, st, fs,
                     run_dir=run.settings.run_dir, identity=run.ident, deadline=deadline, on_eval=on_eval)
    run.staged, run.plan = res, plan
    _staged_outputs(run, res)
    _copy_best_weights(run)
    return run


def _staged_outputs(run: Run, res: StagedResult) -> None:
    """FitResult for the later steps, the stage table (with the rough independent-label counts) for the summary."""
    run.result = FitResult(res.history, len(res.history), res.best_eval, res.best_value, res.finished,
                           res.restored_best)
    indep = staged_report.independent_labels(run.dataset, run.idx["train"], [r["rows"] for r in res.stages])
    run.extra["stages"] = [{**r, "independent_labels": n} for r, n in zip(res.stages, indep, strict=True)]
    run.extra["staged"] = {"total_steps": res.total_steps, "finished": res.finished, "val_rows_used": res.val_rows,
                           "best_step": res.best_step}


def restore(run: Run, *, weights: str = "best", allow_changes: tuple[str, ...] = ("code_sha",)) -> Run:
    """Instead of ``train``: read a FINISHED run folder (``settings.run_dir``) and put its weights in the model, so the
    evaluation steps re-read a trained model without training (T5r). ``weights``: ``"best"`` (default) = the
    best-epoch weights (``best.pt``); ``"last"`` = the weights after the last finished epoch (``checkpoint.pt``; the EMA
    copy when the run trained with ``ema_decay``, exactly what ``fit`` leaves in the model without
    ``restore_best_weights``). The saved RunIdentity must equal this run's except for ``allow_changes`` (default: the
    commit; settings, seed and data must be the same)."""
    if weights not in RESTORE_WEIGHTS:
        raise ValueError(f"restore: weights must be one of {RESTORE_WEIGHTS}, got {weights!r}")
    _need(run, "trainer", "build")
    run_dir = Path(run.settings.run_dir)
    ckpt = checkpoint.load_checkpoint(run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"{run_dir} holds no {checkpoint.CHECKPOINT}: nothing to restore")
    check_resume(RunIdentity.from_json(json.loads(ckpt["identity"])), run.ident,
                 allow_changes=("epochs", *allow_changes))
    tracker = ckpt["tracker"]
    if weights == "best":
        if tracker.get("best_epoch") is None:
            raise ValueError(f"{run_dir}: the checkpoint records no best epoch")
        run.trainer.model.load_state_dict(checkpoint.load_best(run_dir))
    else:
        run.trainer.model.load_state_dict(ckpt["ema"]["shadow"] if "ema" in ckpt else ckpt["model"])
    history = list(ckpt["history"])
    run.result = FitResult(history, int(ckpt["epochs_done"]), tracker["best_epoch"], tracker["best"],
                           len(history) < run.fit_settings.epochs, weights == "best")
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
    if settings.train.staged is None:
        train(r, deadline=deadline, on_epoch=on_epoch)
    else:
        train_staged(r, deadline=deadline, on_eval=on_epoch)
    health_step(r)
    rep = test_step(r)
    print(f"test evaluation ({rep['n_rows']} rows, best-epoch weights):", flush=True)
    print("\n".join(test_lines(rep)), flush=True)
    battery_step(r)
    return r
