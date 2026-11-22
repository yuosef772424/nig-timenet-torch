"""PURPOSE: The machinery ``Trainer.fit`` and ``staged._Loop`` share: the precondition checks, resume / restore of the
    common checkpoint part, the checkpoint state dict, best-weights bookkeeping and the end-of-run restore.
TAGS: fit state, resume, restore, checkpoint state dict, best weights, preconditions, seed check, kendall, balancer
PITFALLS: the state dict's key NAMES and ORDER are the checkpoint format (``checkpoint.py``): an old checkpoint must
    still resume, so a key is never renamed or dropped here. The EMA and day-weight parts belong to ``Trainer.fit``
    alone (it adds / loads them after the common part); the ``staged`` part belongs to ``staged._Loop`` (passed in as
    ``staged``, it lands right after ``history`` as before). Helpers take the trainer as a duck: its ``model``,
    ``optimizer``, ``uncertainty_layer``, ``balancer`` and ``selection_metric``.

One place for what the two fit loops did twice (hardening HB4b, audit A-07 / A-20).
"""

import json
from pathlib import Path

import torch

from nig_timenet_torch.trainer import checkpoint
from nig_timenet_torch.trainer.identity import RunIdentity, SeedMismatchError, check_resume

__all__ = ["clone_state", "require_optimizer", "check_run_args", "load_resume_checkpoint", "base_state",
           "restore_base", "check_checkpoint_identity", "save_best_weights", "finish_weights"]


def clone_state(model: torch.nn.Module) -> dict:
    """A detached copy of the model's state dict (the best weights, the live weights put back after an evaluation)."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def require_optimizer(trainer) -> None:
    if trainer.optimizer is None:
        raise RuntimeError("Trainer has no optimizer")


def check_run_args(run_dir, identity: RunIdentity | None, seed: int) -> None:
    """A run folder needs an identity (so a resume can be checked) and the identity must describe the seed trained."""
    if run_dir is not None and identity is None:
        raise ValueError("run_dir needs an identity (RunIdentity) so that a resume can be checked")
    if identity is not None and identity.seed != seed:
        raise SeedMismatchError(f"identity.seed ({identity.seed}) != settings.seed ({seed}): the identity "
                                f"would describe a different run than the one that is trained")


def load_resume_checkpoint(run_dir, resume: bool) -> dict | None:
    """The folder's checkpoint (None = a fresh run); ``resume=False`` refuses to overwrite an existing one."""
    ckpt = None if run_dir is None else checkpoint.load_checkpoint(Path(run_dir))
    if ckpt is not None and not resume:
        raise FileExistsError(f"{run_dir} already holds a checkpoint; resume=False refuses to overwrite it")
    return ckpt


def base_state(trainer, epochs_done: int, tracker, history: list[dict], identity: RunIdentity,
               staged: dict | None = None) -> dict:
    """The checkpoint dict both loops write (``epochs_done`` is the number of history rows)."""
    state = {"epochs_done": epochs_done, "model": trainer.model.state_dict(), "tracker": tracker.get_state(),
             "optimizer": trainer.optimizer.state_dict(), "history": history}
    if staged is not None:
        state["staged"] = staged
    state["kendall"] = None if trainer.uncertainty_layer is None else trainer.uncertainty_layer.state_dict()
    state["identity"] = json.dumps(identity.to_json(), sort_keys=True)
    if trainer.balancer.needs_shared:  # FX2 balancer state; a default checkpoint keeps its keys
        state["balance"] = trainer.balancer.state_dict()
    return state


def check_checkpoint_identity(ckpt: dict, identity: RunIdentity) -> None:
    """Refuse a resume whose checkpoint was written for another run (ResumeMismatchError)."""
    check_resume(RunIdentity.from_json(json.loads(ckpt["identity"])), identity)


def restore_base(trainer, ckpt: dict, tracker) -> list[dict]:
    """Load the common part (model, Kendall, optimizer, balancer, tracker) of ``ckpt``; returns the history."""
    trainer.model.load_state_dict(ckpt["model"])
    if trainer.uncertainty_layer is not None:
        trainer.uncertainty_layer.load_state_dict(ckpt["kendall"])
    trainer.optimizer.load_state_dict(ckpt["optimizer"])
    if trainer.balancer.needs_shared:
        trainer.balancer.load_state_dict(ckpt["balance"])
    tracker.load_state(ckpt["tracker"])
    return list(ckpt["history"])


def save_best_weights(run_dir, weights: dict, tracker, metric: str) -> None:
    """``best.pt`` + ``best_meta.json`` of the tracker's current best (a no-op without a run folder)."""
    if run_dir is not None:
        checkpoint.save_best(run_dir, weights, {"best": tracker.best, "best_epoch": tracker.best_epoch,
                                                "metric": metric})


def finish_weights(model: torch.nn.Module, best_weights: dict | None, restore_best: bool, ema=None) -> bool:
    """Put the final weights into the model: the best ones (True returned), else the EMA copy when there is one."""
    restored = restore_best and best_weights is not None
    if restored:
        model.load_state_dict(best_weights)
    elif ema is not None:
        model.load_state_dict(ema.weights())
    return restored
