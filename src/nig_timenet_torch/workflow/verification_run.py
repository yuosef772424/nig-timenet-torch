"""The integrated verification as a run step: reads a finished Run, writes verification_summary.csv / .json

PURPOSE: ``verification_step(run)``: runs eval.verification.run_full_verification on the val and test rows of a
    finished Run with its ``evaluation.verification`` settings and the battery's cost; writes the verdict table.
TAGS: verification step, verdict table, integrated verification, run step, verification settings
PITFALLS: needs a trained (or restored) Run, the test split is read only here and by test_step / battery_step; the
    val and test tables are the ones ``test_step`` / ``battery_step`` keep in ``run.extra['tables']`` (predicted once);
    the target mode is read from the artifact's ``target_mode`` stamp (absent = plain returns); the step is NOT part of
    ``run_all``: call it after ``test_step``, like ``battery_step``.
"""

from nig_timenet_torch.eval.verification import run_full_verification
from nig_timenet_torch.workflow import run_common
from nig_timenet_torch.workflow.run_state import Run, need

__all__ = ["verification_step"]


def verification_step(run: Run, *, verbose: bool = False) -> dict:
    """``{"summary": DataFrame, "details": {...}}`` of the integrated verification of ``run`` (``{}`` when
    ``evaluation.verification.enabled`` is False); the CSV and JSON land in ``run.out`` when it is set."""
    v = run.settings.evaluation.verification
    if not v.enabled:
        return {}
    need(run, "result", "train")
    if not len(run.idx["test"]):
        raise ValueError("the split has no test rows")
    cache = run.extra.setdefault("tables", {})
    for stage in ("val", "test"):
        if stage not in cache:
            cache[stage] = run.table(stage)
    ds = run.dataset
    return run_full_verification(
        ds, run.idx, {s: cache[s] for s in ("val", "test")}, targets=run.targets,
        x_keys=run_common.input_keys(ds, run.settings.data.input_timeframes),
        cost_per_side=run.settings.evaluation.battery.cost_per_side, target_mode=ds.get("target_mode"),
        target_acc=v.target_acc, rr=v.rr, stop_mode=v.stop_mode, n_boot=v.n_boot, run_candle=v.run_candle,
        latest_n=v.latest_n, latest_coins=v.latest_coins,
        out_dir=run.out, verbose=verbose)
