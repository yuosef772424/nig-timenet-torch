"""PURPOSE: The run steps of one flat single-split run (prepare, build, train, health report, test report, summary)
    and run_all() that calls them in order. Jobs and notebooks call the steps; nobody copies them.
TAGS: run steps, single split, flat run, train step, health step, test report step, run summary, notebook steps
PITFALLS: steps share one mutable Run (call them in order; a step that needs an earlier one raises); the test split is
    used only by test_step, never for training or selection; a rolling run is one Run per fold
    (workflow/rolling_run.py), a panel study is one Run per arm and seed (workflow/panel_study.py); the training data
    stay on the CPU and move to the device batch by batch (B-019). ``train_staged`` (ST1) shares the Run, the Trainer
    and the later steps with ``train``.
    The steps live in run_state (Run), run_train, run_restore, run_eval and run_summary; this module keeps
    prepare / build / run_all and re-exports every public step name, so ``from workflow import run as steps`` is the
    one import of callers (C-25).
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

import os
from pathlib import Path

import torch

from nig_timenet_torch.core.seeds import seed_everything
from nig_timenet_torch.data_io.loading import LoadedData
from nig_timenet_torch.model.label_scale import fit_label_scale
from nig_timenet_torch.model.scalers import fit_scalers
from nig_timenet_torch.model.target_norm import fit_target_norm
from nig_timenet_torch.trainer.groups import BATCH_ORDERS
from nig_timenet_torch.workflow import flat_run, run_common
from nig_timenet_torch.workflow.battery_run import battery_step
from nig_timenet_torch.workflow.run_eval import TEST_REPORT, health_step, stage_report, test_step
from nig_timenet_torch.workflow.run_restore import RESTORE_WEIGHTS, restore
from nig_timenet_torch.workflow.run_state import Run, need
from nig_timenet_torch.workflow.run_summary import SUMMARY, flat_scalars, test_lines, write_summary
from nig_timenet_torch.workflow.run_train import (
    DAY_WEIGHTS_CSV,
    batches_per_epoch,
    fit,
    train,
    train_staged,
    training_finished,
    weights_label,
)
from nig_timenet_torch.workflow.settings import RunSettings

__all__ = ["Run", "prepare", "prepare_split", "build", "fit", "train", "train_staged", "training_finished",
           "weights_label",
           "need", "flat_scalars", "restore", "health_step", "test_step", "battery_step", "write_summary",
           "run_all", "test_lines", "SUMMARY", "TEST_REPORT", "DAY_WEIGHTS_CSV", "stage_report", "batches_per_epoch",
           "RESTORE_WEIGHTS"]


def prepare(settings: RunSettings, out=None) -> Run:
    """Load the artifact (rule D1), read its facts (they raise on a missing or unknown one before any work), split by
    time and build the train / validation inputs."""
    if out is not None:
        os.makedirs(out, exist_ok=True)
    art = run_common.load_data(settings)
    ds = art.dataset
    targets = run_common.price_targets(ds)
    run_common.check_decode(ds, targets)
    run_common.input_keys(ds, settings.data.input_timeframes)
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
        run.x[stage], run.y[stage] = flat_run.flat_inputs(ds, run.idx[stage], settings.split.cross_section,
                                                          settings.split.coin_channel,
                                                          settings.data.input_timeframes)
    return run


def build(run: Run) -> Run:
    """Seed, build the model and the Trainer, then the run identity (AFTER the Trainer: the optimizer holds the Kendall
    group only once the trainer exists)."""
    s = run.settings
    seed_everything(s.train.seed)
    run.ms = flat_run.model_settings(s.model, run.targets)
    run.trainer = flat_run.make_trainer(s, run.ms, run.targets, run.x["train"].shape[1], run.x["train"].shape[2],
                                        run.device)
    if BATCH_ORDERS.entry(s.train.batch_order).meta["needs_times"]:      # the TRAIN rows' timestamp ids (no channel)
        for attr, stage in (("row_times", "train"), ("val_times", "val")):
            ids = flat_run.time_ids(run.dataset, run.idx[stage])
            setattr(run.trainer, attr, torch.as_tensor(ids, dtype=torch.int64))
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


def run_all(settings: RunSettings, out=None, *, deadline: float | None = None, on_epoch=None) -> Run:
    """prepare -> build -> train -> health report -> test report -> battery (when ``evaluation.battery.enabled``)."""
    r = build(prepare(settings, out))
    fit(r, deadline=deadline, on_epoch=on_epoch)
    health_step(r)
    rep = test_step(r)
    print(f"test evaluation ({rep['n_rows']} rows, {weights_label(r)} weights):", flush=True)
    print("\n".join(test_lines(rep)), flush=True)
    battery_step(r)
    return r
