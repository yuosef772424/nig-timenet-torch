"""PURPOSE: Putting weights back into a Run's model: ``restore`` a finished run folder, the copy of the best weights to
    ``out``, and the stop (KeyboardInterrupt) handling of the training steps.
TAGS: restore weights, best weights copy, interrupt handling, run folder checkpoint, on_interrupt
PITFALLS: ``restore`` checks the saved RunIdentity against the run's (only ``code_sha`` and ``epochs`` may differ unless
    ``allow_changes``); a stop before the first finished epoch has nothing to restore and is re-raised; the best-weight
    copy never overwrites itself (``out`` may be the run folder).
"""

import difflib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

from nig_timenet_torch.trainer import checkpoint
from nig_timenet_torch.trainer.checkpoint import BEST, BEST_META, CHECKPOINT
from nig_timenet_torch.trainer.identity import RunIdentity, check_resume
from nig_timenet_torch.trainer.trainer import FitResult
from nig_timenet_torch.workflow.run_state import Run, need

__all__ = ["RESTORE_WEIGHTS", "restore", "copy_best_weights", "handle_interrupt", "check_on_interrupt"]

RESTORE_WEIGHTS = ("best", "last")  # restore(weights=...): best-epoch weights (default) or the last finished epoch


def handle_interrupt(run: Run, mode: str, err: KeyboardInterrupt) -> None:
    """A stop (the Colab stop button) during training with ``on_interrupt`` = ``mode``: put the weights of the last
    finished epoch (``"last"``) or of the best epoch so far (``"best"``) in the model through ``restore`` and set
    ``run.result`` (stopped early), so the later steps run. No finished epoch (no checkpoint): the stop is re-raised."""
    if checkpoint.load_checkpoint(Path(run.settings.run_dir)) is None:
        raise KeyboardInterrupt("stopped before the first epoch finished: no epoch was saved, nothing to test") from err
    restore(run, weights=mode)
    res = run.result = replace(run.result, finished=False)
    print(f"STOPPED: {res.epochs_done} finished epoch(s) kept; the model holds the "
          f"{'best-epoch' if mode == 'best' else 'last finished epoch'} weights (a cut epoch is not saved)",
          flush=True)
    copy_best_weights(run)


def check_on_interrupt(on_interrupt) -> None:
    if on_interrupt is not None and on_interrupt not in RESTORE_WEIGHTS:
        raise ValueError(f"on_interrupt must be None or one of {RESTORE_WEIGHTS}, got {on_interrupt!r}")


def _similar_runs(run_dir: Path) -> list[str]:
    """Run folders next to ``run_dir`` (or next to its run name folder, ``<root>/<RUN_NAME>/train``) whose name is close
    to it and that hold a checkpoint: the likely intended folder when the name has a typo."""
    out: list[str] = []
    for level, key in ((run_dir.parent.parent, run_dir.parent.name), (run_dir.parent, run_dir.name)):
        if not level.is_dir():
            continue
        names = [d.name for d in level.iterdir() if d.is_dir() and d.name != key]
        for match in difflib.get_close_matches(key, names, n=3, cutoff=0.75):
            cand = level / match
            if (cand / CHECKPOINT).is_file() or (cand / "train" / CHECKPOINT).is_file():
                out.append(str(cand))
    return out


def _no_checkpoint(run_dir: Path) -> str:
    """Why ``restore`` found nothing: no such folder (a new RUN_NAME or a typo in it), or a folder without the
    checkpoint file; with the run folders of a similar name that DO hold one."""
    if not run_dir.exists():
        what = f"{run_dir} does not exist: no run was ever started there (a new RUN_NAME, or a typo in it)"
    elif not run_dir.is_dir():
        what = f"{run_dir} is not a folder"
    else:
        held = sorted(p.name for p in run_dir.iterdir())[:8]
        what = (f"{run_dir} exists but holds no {CHECKPOINT} (it holds {held}); a run writes {CHECKPOINT} (the last "
                f"finished epoch, what a resume continues from) and {BEST} (the best epoch) into its train folder")
    near = _similar_runs(run_dir)
    return f"{what}: nothing to restore" + (f"; did you mean {near}?" if near else "")


def restore(run: Run, *, weights: str = "best", allow_changes: tuple[str, ...] = ("code_sha",),
            from_dir=None) -> Run:
    """Instead of ``train``: read a FINISHED run folder (``settings.run_dir``, ``<RUN_NAME>/train`` in the notebook; or
    ``from_dir``, another run's folder, whose saved identity must still equal this run's) and put its weights in the
    model, so the evaluation steps re-read a trained model without training (T5r). ``weights``: ``"best"`` (default) =
    the best-epoch weights (``best.pt``); ``"last"`` = the weights after the last finished epoch (``checkpoint.pt``;
    the EMA copy when the run trained with ``ema_decay``, exactly what ``fit`` leaves in the model without
    ``restore_best_weights``); it is never a path: use ``from_dir`` for another folder. The saved RunIdentity must
    equal this run's except for ``allow_changes`` (default: the commit; settings, seed and data must be the same)."""
    if weights not in RESTORE_WEIGHTS:
        hint = (f" (a path is not accepted here: pass from_dir={weights!r} to read another run's folder)"
                if os.sep in str(weights) or str(weights).endswith(".pt") else "")
        raise ValueError(f"restore: weights must be one of {RESTORE_WEIGHTS}, got {weights!r}{hint}")
    need(run, "trainer", "build")
    run_dir = Path(from_dir if from_dir is not None else run.settings.run_dir)
    ckpt = checkpoint.load_checkpoint(run_dir)
    if ckpt is None:
        raise FileNotFoundError(_no_checkpoint(run_dir))
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
    stopped = int(tracker["wait"]) >= run.fit_settings.early_stop.patience
    run.result = FitResult(history, int(ckpt["epochs_done"]), tracker["best_epoch"], tracker["best"], stopped,
                           weights == "best", stopped or len(history) >= run.fit_settings.epochs)
    return run


def copy_best_weights(run: Run) -> None:
    """``best.pt`` + ``best_meta.json`` of the run folder into ``out``, so a job publishes the weights with its
    results (B-050); skipped without ``out`` or before any best epoch, never copied onto itself."""
    if run.out is None:
        return
    for name in (BEST, BEST_META):
        src, dst = Path(run.settings.run_dir) / name, run.out / name
        if src.is_file() and not (dst.exists() and dst.samefile(src)):
            shutil.copyfile(src, dst)
