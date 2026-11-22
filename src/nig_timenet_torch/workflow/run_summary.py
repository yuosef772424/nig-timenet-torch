"""PURPOSE: The printable and written summary of a finished Run: the test-report lines and ``baseline_summary.json``.
TAGS: run summary, baseline summary json, test lines, flat scalars, step seconds per batch
PITFALLS: ``write_summary`` needs a trained Run; ``extra`` (device, gpu name, ...) is merged in by the caller; the
    per-batch time divides by the batches the fit really used (``batches_per_epoch``), not by rows / batch_size.
"""

import statistics

from nig_timenet_torch.workflow import flat_run, run_common
from nig_timenet_torch.workflow.run_state import Run, need
from nig_timenet_torch.workflow.run_train import batches_per_epoch

__all__ = ["SUMMARY", "flat_scalars", "test_lines", "write_summary"]

SUMMARY = "baseline_summary.json"


def flat_scalars(d, prefix=""):
    """``(dotted key, number)`` of every numeric leaf of a nested report dict (bools skipped)."""
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat_scalars(v, f"{prefix}{k}.")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield f"{prefix}{k}", v


def test_lines(report: dict) -> list[str]:
    """Short printable summary of a test report: every scalar except the per-coin breakdowns."""
    return [f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}"
            for k, v in flat_scalars(report) if "per_coin" not in k and "deciles" not in k]


def write_summary(run: Run, extra: dict | None = None) -> dict:
    """``baseline_summary.json`` in ``out``; ``extra`` (device, gpu name, ...) is merged in by the caller."""
    need(run, "result", "train")
    s, res, fs, ds = run.settings, run.result, run.fit_settings, run.dataset
    secs = [r["trace_epoch_seconds"] for r in res.history if "trace_epoch_seconds" in r]  # epochs of earlier jobs too
    n_batches = batches_per_epoch(run, flat_run.batch_groups(ds, run.idx["train"], s.train.batch_by))
    summary = {"seed": s.train.seed, "epochs_requested": fs.epochs, "epochs_done": res.epochs_done,
               "finished": res.finished, "stopped_early": res.stopped_early,
               "best_epoch": res.best_epoch, "best_value": res.best_value,
               "step_seconds_per_batch": statistics.median(secs) / n_batches if secs else None,
               "epoch_seconds_median": statistics.median(secs) if secs else None, "batch_size": fs.batch_size,
               "n_windows": {k: int(len(v)) for k, v in run.idx.items()},
               "split": {"train_end": str(run.bounds[0]), "val_end": str(run.bounds[1])},
               "targets": list(run.targets), "label_smoothing": s.train.label_smoothing,
               "reg_target_mode": ds["reg_target_mode"], "reg_target_scale": float(ds["reg_target_scale"]),
               "input_key": "+".join(run_common.input_keys(ds, s.data.input_timeframes)),
               "max_rows": s.split.max_rows, "model_overrides": run.ms.overrides,
               "data_fingerprint": run.art.manifest.data_fingerprint, "identity_fingerprint": run.ident.fingerprint(),
               "run_dir": str(s.run_dir), "history": res.history,
               **{k: v for k, v in run.extra.items() if k != "tables"}, **(extra or {})}
    if run.out is not None:
        run_common.write_json(run.out / SUMMARY, summary)
    return summary
