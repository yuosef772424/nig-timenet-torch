"""PURPOSE: Rolling walk-forward through the run steps (ROADMAP F2): per fold and seed one Run (purged fold split,
    scalers fitted on the fold's train rows, the one Trainer, test report), per-fold tables, mean and spread per
    fold and across folds.
TAGS: rolling run, walk forward, folds, seeds, retrain every k, inner validation, per fold metrics, fold summary, f2
PITFALLS: a fold has no validation set of its own: the last ``val_share`` of its train window (purged) selects the epoch
    and stops early, the test window is touched only by the report after training (D-031); a fold that REUSES the
    last model (retrain_every_k) is evaluated with the scalers and weights of the fold it was trained on; every fold
    x seed has its own run folder, so a cut job resumes (a finished fold re-reads its checkpoint, nothing retrains);
    with ``fold_ids`` several processes share one ``out``: each writes only its folds, and the summary files are
    written by the process that finds every fold x seed of the geometry on disk (the others return
    ``complete: False``); every ``fold_row.json`` carries the ``study_hash`` (D-042) and a row of another study (or
    one written before D-042) in ``out`` raises instead of entering the summary;
    the fold geometry replaces the split: ``split.train_end`` / ``val_end`` / ``train_share`` / ``val_share`` /
    ``embargo_hours`` away from their defaults are refused (C-11), the coin exclusions, ``max_rows``, ``top_coins``
    and everything under ``train`` / ``model`` / ``evaluation`` apply per fold.
    Exploratory (rule C5): the spread across folds is a look, the number of folds is small.
"""

import dataclasses
import hashlib
import json
import math
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from nig_timenet_torch.core.config import updated
from nig_timenet_torch.trainer.identity import settings_hash
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow import run_common
from nig_timenet_torch.workflow.prediction_table import write_table
from nig_timenet_torch.workflow.rolling import RollingFold, fold_with_val, rolling_folds
from nig_timenet_torch.workflow.settings import RollingSettings, RunSettings, SplitSettings

__all__ = ["run_rolling", "fold_dir", "summarize", "render", "HEADLINE_KEYS", "study_hash", "StudyMismatchError",
           "SPLIT_FIELDS_REPLACED"]

#: substrings of the flat test-report keys the markdown summary shows (the JSON and the CSV hold every scalar)
HEADLINE_KEYS = ("rank_ic_by_timestamp", "share_std_below_move", "coverage", "mae_skill", "nll_skill")
_SKIP = ("per_coin", "deciles", "shuffled", "baseline", "n_rows")


#: the split fields the fold geometry replaces (C-11): a rolling run refuses them away from their defaults
SPLIT_FIELDS_REPLACED = ("train_end", "val_end", "train_share", "val_share", "embargo_hours")
#: RollingSettings fields that only say which share of the study a process runs: not part of the study hash
_SHARE_FIELDS = ("fold_ids", "seeds")
#: RollingSettings fields that shape a fold's train / validation rows (the rest only choose which folds exist)
_FOLD_FIELDS = ("mode", "train_span", "embargo", "val_share")


class StudyMismatchError(ValueError):
    """A ``fold_row.json`` under ``out`` belongs to another rolling study (or was written before D-042)."""


def fold_dir(out, fold: int, seed: int) -> Path:
    return Path(out) / f"fold_{fold:03d}_seed_{seed}"


def _plain(d: dict) -> dict:
    return {k: str(v) if isinstance(v, pd.Timedelta) else v for k, v in d.items()}


def study_hash(settings: RunSettings, rolling: RollingSettings, data_fingerprint: str) -> str:
    """The rolling study a fold row belongs to (D-042, audit C-06): the fold geometry without the share fields, the
    run settings without the per-process ones (``run_dir``, ``code_sha``, the seed, ``RESUMABLE_TRAIN_FIELDS``, where
    the artifact is read from) and the data fingerprint. Processes sharing one study (``fold_ids``) get the same."""
    a = dataclasses.asdict
    skip = (*run_common.RESUMABLE_TRAIN_FIELDS, "seed")
    run = {"split": a(settings.split), "input_timeframes": list(settings.data.input_timeframes),
           "model": a(settings.model), "train": {k: v for k, v in a(settings.train).items() if k not in skip},
           "evaluation": a(settings.evaluation)}
    geometry = {k: v for k, v in _plain(a(rolling)).items() if k not in _SHARE_FIELDS}
    return settings_hash({"rolling": geometry, "run": run, "data": data_fingerprint})


def _fold_identity(ident, fold: RollingFold, rolling: RollingSettings):
    """The model's identity for this fold: fold index and data cut set; parents: the run's settings hash and the hash
    of the fold-shaping geometry (mode, train span, embargo, inner validation share; D-042)."""
    shape = json.dumps({k: _plain(dataclasses.asdict(rolling))[k] for k in _FOLD_FIELDS}, sort_keys=True)
    return dataclasses.replace(ident, fold=fold.index, data_cut=fold.train_end.isoformat(),
                               parent_hashes=(ident.settings_hash, hashlib.sha256(shape.encode()).hexdigest()))


def _refuse_replaced_split(settings: RunSettings) -> None:
    set_ = [f for f in SPLIT_FIELDS_REPLACED if getattr(settings.split, f) != getattr(SplitSettings(), f)]
    if set_:
        raise ValueError(f"a rolling run takes its train / validation / test cut from the fold geometry "
                         f"(RollingSettings); split.{', split.'.join(set_)} would be ignored: leave them at their "
                         f"defaults (C-11)")


def _fold_run(settings: RunSettings, rolling: RollingSettings, art, fold: RollingFold, seed: int, out,
              ckpt_root) -> steps.Run:
    plan, cut_v = fold_with_val(fold, rolling.embargo, rolling.val_share)
    s = updated(settings, {"train.seed": seed, "run_dir": str(fold_dir(ckpt_root or out, fold.index, seed) / "ckpt")})
    parts, bounds = (plan.train_idx, plan.val_idx, plan.test_idx), (cut_v, fold.train_end)
    run = steps.prepare_split(s, art, parts, bounds, fold_dir(out, fold.index, seed))
    run.fold = fold.index
    return run


def _train_fold(run: steps.Run, fold: RollingFold, rolling: RollingSettings, on_epoch, deadline) -> None:
    steps.build(run)       # the scalers and the NIG label scale are fitted on THIS fold's train rows
    run.ident = _fold_identity(run.ident, fold, rolling)
    run.extra["trained_on_fold"] = fold.index
    if run.settings.train.staged is None:
        steps.train(run, on_epoch=on_epoch, deadline=deadline)
    else:
        steps.train_staged(run, on_eval=on_epoch, deadline=deadline)


def _reuse(run: steps.Run, last: steps.Run, fold: RollingFold, rolling: RollingSettings) -> None:
    """The fold evaluates the model of ``last`` (trained on an earlier fold): same weights, scalers and fit record."""
    for name in ("ms", "trainer", "fit_settings", "result", "staged", "plan"):
        setattr(run, name, getattr(last, name))
    run.extra.update({k: v for k, v in last.extra.items()
                      if k in ("fitted_scalers", "stages", "staged", "trained_on_fold")})
    run.ident = _fold_identity(last.ident, fold, rolling)
    run.bounds = (last.bounds[0], run.bounds[1])     # the table says when the model's training data ended


def _rows_on_disk(out, folds: list[int], seeds, study: str) -> list[dict]:
    """The fold rows already written under ``out`` (by this process or another one sharing the study); a row of
    another study raises ``StudyMismatchError``."""
    rows = []
    for f in folds:
        for seed in seeds:
            p = fold_dir(out, f, seed) / "fold_row.json"
            if p.is_file():
                row = json.loads(p.read_text(encoding="utf-8"))
                if row.get("study_hash") != study:
                    raise StudyMismatchError(
                        f"{p} belongs to another rolling study (study_hash {row.get('study_hash')!r}, this study "
                        f"{study!r}; none = written before D-042): use another out folder or delete the row")
                rows.append(row)
    return rows


def _cut(rows: list[dict], art, study: str) -> dict:
    """What ``run_rolling`` returns when the deadline came first (C-07): nothing is summarised."""
    print(f"rolling: cut by the deadline after {len(rows)} fold x seed row(s) in this call; run again to continue",
          flush=True)
    return {"complete": False, "finished": False, "n_rows": len(rows),
            "data_fingerprint": art.manifest.data_fingerprint, "study_hash": study}


def _scalars(report: dict) -> dict:
    return {k: float(v) for k, v in steps.flat_scalars(report) if not any(t in k for t in _SKIP)}


def run_rolling(settings: RunSettings, rolling: RollingSettings, out, *, on_epoch=None,
                on_fold: Callable | None = None, ckpt_root=None, deadline: float | None = None) -> dict:
    """The walk-forward study. For every fold x seed: split (``fold_with_val``), train through the run steps (or reuse
    the seed's last model when ``fold % retrain_every_k != 0``), evaluate on the fold's test window, write
    ``fold_<i>_seed_<s>/`` (``test_report.json``, ``test_table.parquet`` + meta, the battery files when enabled,
    ``fold_row.json``). ``on_fold(run, row)`` is called after each fold before its tensors are freed. Writes
    ``rolling_folds.csv``, ``rolling_summary.json`` and ``rolling_summary.md`` to ``out``; returns the summary.
    ``ckpt_root`` puts the fold checkpoints (the resume points) elsewhere (a Drive folder) than the outputs.
    ``deadline`` (a ``time.monotonic()`` value): no fold x seed starts after it and the training stops at an epoch
    border before it; a cut fold writes no row, and the call returns ``{"complete": False, "finished": False, ...}``
    (run it again: the cut fold resumes from its checkpoint, the finished ones re-read theirs)."""
    _refuse_replaced_split(settings)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    art = run_common.load_data(settings)
    study = study_hash(settings, rolling, art.manifest.data_fingerprint)
    targets = run_common.price_targets(art.dataset)
    run_common.check_decode(art.dataset, targets)
    folds = rolling_folds(art.dataset, rolling)
    if not folds:
        raise ValueError("the rolling settings give no fold: train_span + embargo + test_span passes the last label")
    rows: list[dict] = []
    last: dict[int, steps.Run] = {}
    for fold, seed in ((f, s) for f in folds for s in rolling.seeds):
        if deadline is not None and time.monotonic() >= deadline:
            return _cut(rows, art, study)
        run = _fold_run(settings, rolling, art, fold, seed, out, ckpt_root)
        trained = fold.index % rolling.retrain_every_k == 0 or seed not in last
        if trained:
            _train_fold(run, fold, rolling, on_epoch, deadline)
            if not steps.training_finished(run):
                return _cut(rows, art, study)
        else:
            _reuse(run, last[seed], fold, rolling)
        report = steps.test_step(run)
        steps.battery_step(run)
        write_table(run.out / "test_table.parquet", run.extra["tables"]["test"], run.ident)
        row = {"study_hash": study, "fold": fold.index, "seed": seed, "trained": trained,
               "trained_on_fold": run.extra["trained_on_fold"],
               "n_train": int(len(run.idx["train"])), "n_val": int(len(run.idx["val"])),
               "n_test": int(len(run.idx["test"])), "train_end": str(fold.train_end),
               "test_start": str(fold.test_start), "test_end": str(fold.test_end),
               "best_epoch": run.result.best_epoch, "epochs_done": run.result.epochs_done,
               "metrics": _scalars(report)}
        (run.out / "fold_row.json").write_text(json.dumps(row, indent=1, default=float), encoding="utf-8")
        rows.append(row)
        if on_fold is not None:
            on_fold(run, row)
        if trained:
            last[seed] = run
        run.x.clear(), run.y.clear(), run.extra.pop("tables", None)    # free the tensors; the model stays
    if rolling.fold_ids is not None:     # a share of the study: the summary waits for every fold x seed
        every = rolling_folds(art.dataset, dataclasses.replace(rolling, fold_ids=None))
        rows, rolling = _rows_on_disk(out, [f.index for f in every], rolling.seeds, study), dataclasses.replace(
            rolling, fold_ids=None)
        if len(rows) < len(every) * len(rolling.seeds):
            return {"complete": False, "n_folds": len({r["fold"] for r in rows}),
                    "data_fingerprint": art.manifest.data_fingerprint, "study_hash": study}
    summary = {**summarize(rows, rolling), "data_fingerprint": art.manifest.data_fingerprint, "study_hash": study}
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("metrics", "study_hash")} | r["metrics"]
                  for r in rows]).to_csv(
        out / "rolling_folds.csv", index=False)
    (out / "rolling_summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    (out / "rolling_summary.md").write_text(render(summary), encoding="utf-8")
    return summary


def _stats(values: list[float]) -> dict:
    v = [x for x in values if math.isfinite(x)]
    return {"mean": statistics.fmean(v) if v else None, "sd": statistics.stdev(v) if len(v) > 1 else None,
            "min": min(v) if v else None, "max": max(v) if v else None, "n": len(v)}


def summarize(rows: list[dict], rolling: RollingSettings) -> dict:
    """Per fold: mean and sd over the seeds of every scalar of the test report; across folds: mean, sd, min, max of
    the per-fold means (the stability of the metric). ``sd`` is the sample standard deviation, None for one value."""
    keys = sorted({k for r in rows for k in r["metrics"]})
    per_fold = []
    for f in sorted({r["fold"] for r in rows}):
        fr = [r for r in rows if r["fold"] == f]
        per_fold.append({"fold": f, "seeds": [r["seed"] for r in fr], "n_test": fr[0]["n_test"],
                         "train_end": fr[0]["train_end"], "test_start": fr[0]["test_start"],
                         "test_end": fr[0]["test_end"], "trained": [r["trained"] for r in fr],
                         "metrics": {k: _stats([r["metrics"].get(k, math.nan) for r in fr]) for k in keys}})
    across = {k: _stats([p["metrics"][k]["mean"] for p in per_fold if p["metrics"][k]["mean"] is not None])
              for k in keys}
    return {"settings": {k: str(v) if isinstance(v, pd.Timedelta) else v
                         for k, v in dataclasses.asdict(rolling).items()},
            "n_folds": len(per_fold), "per_fold": per_fold, "across_folds": across}


def render(summary: dict) -> str:
    """rolling_summary.md: the headline metrics per fold (mean over seeds, sd in brackets) and across folds."""
    keys = [k for k in summary["across_folds"] if any(h in k for h in HEADLINE_KEYS)]
    fmt = lambda s: "-" if s["mean"] is None else (  # noqa: E731
        f"{s['mean']:.4f}" + ("" if s["sd"] is None else f" ({s['sd']:.4f})"))
    lines = ["# Rolling walk-forward summary (exploratory, rule C5)", "",
             f"{summary['n_folds']} folds; per fold: mean over seeds (sd over seeds); last rows: across folds "
             f"(mean of the fold means, sd over folds).", "",
             "| metric | " + " | ".join(f"fold {p['fold']}" for p in summary["per_fold"]) + " | across folds |",
             "|---|" + "---|" * (summary["n_folds"] + 1)]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(fmt(p["metrics"][k]) for p in summary["per_fold"]) + " | "
                     + fmt(summary["across_folds"][k]) + " |")
    lines += ["", "| fold | train end | test window | test rows | trained |", "|---|---|---|---|---|"]
    lines += [f"| {p['fold']} | {p['train_end']} | {p['test_start']} .. {p['test_end']} | {p['n_test']} | "
              f"{p['trained']} |" for p in summary["per_fold"]]
    return "\n".join(lines) + "\n"
