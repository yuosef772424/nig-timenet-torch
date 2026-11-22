"""PURPOSE: The training steps of a Run: ``train`` (the epoch loop through the Trainer) and ``train_staged`` (staged
    training, ST1), what they leave in the Run, and the small readers of the result (weights label, finished,
    batches per epoch).
TAGS: train step, staged training step, day weights table, weights label, training finished, batches per epoch
PITFALLS: both resume a run folder that holds a checkpoint; ``deadline`` is a ``time.monotonic()`` value; a stop with
    ``on_interrupt`` keeps the last saved state and the later steps still run; ``train_staged`` needs
    ``settings.train.staged``; the scalers were fitted once by ``build`` on the whole training split.
"""

from pathlib import Path

import pandas as pd
import torch

from nig_timenet_torch.trainer.groups import BATCH_ORDERS, group_batches, row_groups
from nig_timenet_torch.trainer.staged import StagedResult, plan_stages, staged_fit
from nig_timenet_torch.trainer.trainer import FitResult
from nig_timenet_torch.workflow import flat_run, staged_report
from nig_timenet_torch.workflow.run_restore import check_on_interrupt, copy_best_weights, handle_interrupt
from nig_timenet_torch.workflow.run_state import Run, need

__all__ = ["train", "train_staged", "weights_label", "training_finished", "batches_per_epoch", "DAY_WEIGHTS_CSV"]


def train(run: Run, *, deadline: float | None = None, on_epoch=None, on_interrupt: str | None = None,
          batch_monitor=None) -> Run:
    """Fit (resumes a run folder that holds a checkpoint). ``deadline`` is a ``time.monotonic()`` value.
    ``on_interrupt``: ``None`` (default) = a KeyboardInterrupt propagates; ``"last"`` / ``"best"`` = it is caught and
    the model keeps the weights of the last finished epoch / the best epoch so far, ``run.result`` is set.
    ``batch_monitor``: a ``trainer.batch_influence.BatchInfluence`` (log-only; None = none)."""
    need(run, "trainer", "build")
    check_on_interrupt(on_interrupt)
    n = {k: int(len(v)) for k, v in run.idx.items()}
    groups = flat_run.batch_groups(run.dataset, run.idx["train"], run.settings.train.batch_by)
    days = (None if run.settings.train.day_weights is None
            else flat_run.batch_groups(run.dataset, run.idx["train"], "day"))
    n_batches = batches_per_epoch(run, groups)
    print(f"data: {n['train']} train / {n['val']} val / {n['test']} test windows "
          f"(max_rows {run.settings.split.max_rows}), split train_end {run.bounds[0]} val_end {run.bounds[1]}; "
          f"{n_batches} batches per epoch (batch_by {run.settings.train.batch_by}) on {run.device}", flush=True)
    trace_batches = (flat_run.batch_set(run.dataset, run.idx["train"], run.fit_settings.batch_size,
                                        run.settings.train.seed)
                     if run.settings.train.trace_representative else None)
    try:
        run.result = run.trainer.fit(run.x["train"], run.y["train"], run.x["val"], run.y["val"], run.fit_settings,
                                     run_dir=run.settings.run_dir, identity=run.ident, deadline=deadline,
                                     on_epoch=on_epoch, trace_batches=trace_batches, batch_monitor=batch_monitor,
                                     batch_groups=groups, day_groups=days)
    except KeyboardInterrupt as err:
        if on_interrupt is None:
            raise
        handle_interrupt(run, on_interrupt, err)
        return run
    copy_best_weights(run)
    _write_day_table(run)
    return run


#: per-day table of a run with ``train.day_weights`` (rows, measured loss, weight), written next to the checkpoint
DAY_WEIGHTS_CSV = "day_weights.csv"


def _write_day_table(run: Run) -> None:
    """The day weighter's last table (``trainer.day_weights.DayWeighter.table``; the day as a UTC date) in
    ``run_dir`` (D-039; read when looking for the training days the model fails on). No-op without day weights."""
    weighter = run.trainer.day_weighter
    if weighter is None:
        return
    table = pd.DataFrame(weighter.table())
    table.insert(0, "date", pd.to_datetime(table["day"], unit="D", utc=True).dt.strftime("%Y-%m-%d"))
    Path(run.settings.run_dir).mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(run.settings.run_dir) / DAY_WEIGHTS_CSV, index=False)


def train_staged(run: Run, *, deadline: float | None = None, on_eval=None, stages=None,
                 on_interrupt: str | None = None) -> Run:
    """Staged training (ROADMAP ST1, card F) through the one Trainer: ``settings.train.staged`` says how the rows grow
    and the lr falls (``trainer.staged``); resumes a run folder that holds a checkpoint. ``stages``: a ready plan
    (the equal-step baselines of ``workflow.staged_study``) instead of the staged arm's. The per-stage table lands in
    ``run.extra['stages']``; ``run.result`` is a FitResult over the evaluation points so the later steps read it as
    before. The scalers were fitted once on the whole training split by ``build``. ``on_interrupt``: as in ``train``
    (the state is saved at every evaluation point; a stop keeps the last saved one, ``run.staged`` stays None)."""
    need(run, "trainer", "build")
    check_on_interrupt(on_interrupt)
    st = run.settings.train.staged
    if st is None:
        raise ValueError("train_staged needs settings.train.staged (a StageSettings)")
    fs = run.fit_settings
    plan = stages or plan_stages(st, staged_report.growth_context(run.dataset, run.idx["train"], fs.seed),
                                 fs.batch_size, run.trainer.optimizer.param_groups[0]["lr"])
    print(f"staged training: {len(plan)} stage(s), rows {[len(s.rows) for s in plan]}, lr "
          f"{[round(s.lr, 8) for s in plan]}; {len(run.idx['val'])} val rows (at most {st.val_max_rows} used)",
          flush=True)
    try:
        res = staged_fit(run.trainer, run.x["train"], run.y["train"], run.x["val"], run.y["val"], plan, st, fs,
                         run_dir=run.settings.run_dir, identity=run.ident, deadline=deadline, on_eval=on_eval)
    except KeyboardInterrupt as err:
        if on_interrupt is None:
            raise
        handle_interrupt(run, on_interrupt, err)
        return run
    run.staged, run.plan = res, plan
    _staged_outputs(run, res)
    copy_best_weights(run)
    return run


def _staged_outputs(run: Run, res: StagedResult) -> None:
    """FitResult for the later steps, the stage table (with the rough independent-label counts) for the summary."""
    run.result = FitResult(res.history, len(res.history), res.best_eval, res.best_value, False, res.restored_best,
                           res.finished)
    indep = staged_report.independent_labels(run.dataset, run.idx["train"], [r["rows"] for r in res.stages])
    run.extra["stages"] = [{**r, "independent_labels": n} for r, n in zip(res.stages, indep, strict=True)]
    run.extra["staged"] = {"total_steps": res.total_steps, "finished": res.finished, "val_rows_used": res.val_rows,
                           "best_step": res.best_step}


def weights_label(run: Run) -> str:
    """"best-epoch" when the model holds the best epoch's weights (``result.restored_best``), else "last-epoch"
    (EMA, ``restore_best_weights=False``, ``restore(weights="last")``, ``on_interrupt="last"``)."""
    need(run, "result", "train")
    return "best-epoch" if run.result.restored_best else "last-epoch"


def training_finished(run: Run) -> bool:
    """True when the training of ``run`` ended by itself (every epoch or stage, or early stopping); False when a
    ``deadline`` cut it (the checkpoint resumes it). The one definition the studies stop on (C-07)."""
    need(run, "result", "train")
    return run.result.finished


def batches_per_epoch(run: Run, groups) -> int:
    """How many optimizer steps one epoch takes, by the batching ``fit`` uses (C-13: the summary divided the epoch time
    by ceil(rows / batch_size) also for grouped batches): one per group of ``batch_by`` (``groups``), whole-timestamp
    batches of a cross-section run, the batches of a timestamp ``batch_order``, else ceil(rows / batch_size)."""
    fs, tr, n = run.fit_settings, run.trainer, len(run.idx["train"])
    if groups is not None:
        return int(len(torch.unique(groups)))
    if tr.group_channel:
        return len(group_batches(row_groups(run.x["train"]), fs.batch_size, shuffle=False))
    order = BATCH_ORDERS.entry(fs.batch_order)
    if order.meta["needs_times"]:
        return len(order.meta["fn"](n, fs.batch_size, tr.row_times, fs.time_block_bars))
    return -(-n // fs.batch_size)
