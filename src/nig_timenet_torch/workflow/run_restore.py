"""PURPOSE: Putting weights back into a Run's model: ``restore`` a finished run folder, the copy of the best weights to
    ``out``, and the stop (KeyboardInterrupt) handling of the training steps.
TAGS: restore weights, best weights copy, interrupt handling, run folder checkpoint, on_interrupt
PITFALLS: ``restore`` checks the saved RunIdentity against the run's (only ``code_sha`` and ``epochs`` may differ unless
    ``allow_changes``); a stop before the first finished epoch has nothing to restore and is re-raised; the best-weight
    copy never overwrites itself (``out`` may be the run folder).
"""

import json
import shutil
from dataclasses import replace
from pathlib import Path

from nig_timenet_torch.trainer import checkpoint
from nig_timenet_torch.trainer.checkpoint import BEST, BEST_META
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


def restore(run: Run, *, weights: str = "best", allow_changes: tuple[str, ...] = ("code_sha",)) -> Run:
    """Instead of ``train``: read a FINISHED run folder (``settings.run_dir``) and put its weights in the model, so the
    evaluation steps re-read a trained model without training (T5r). ``weights``: ``"best"`` (default) = the
    best-epoch weights (``best.pt``); ``"last"`` = the weights after the last finished epoch (``checkpoint.pt``; the EMA
    copy when the run trained with ``ema_decay``, exactly what ``fit`` leaves in the model without
    ``restore_best_weights``). The saved RunIdentity must equal this run's except for ``allow_changes`` (default: the
    commit; settings, seed and data must be the same)."""
    if weights not in RESTORE_WEIGHTS:
        raise ValueError(f"restore: weights must be one of {RESTORE_WEIGHTS}, got {weights!r}")
    need(run, "trainer", "build")
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
