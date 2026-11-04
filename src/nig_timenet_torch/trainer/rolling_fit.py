"""Training one fold of a rolling (walk-forward) study with the one Trainer; any model, flat or panel.

A rolling study retrains on a window that moves forward in time. What differs from a single fit is only the
bookkeeping, and it lives here so that every model gets it the same way:

* every fold has its own run folder (``<run_dir>/fold_003_seed_0``) and its own RunIdentity (fold, seed, data cut),
  so a cut session resumes at the right fold and a checkpoint of another fold is never picked up;
* ``init="fresh"``: a new model from the factory, seeded for (fold, seed), so folds are independent;
  ``init="warm"``: the previous fold's Trainer (weights and optimizer state) continues on the new window, for
  ``warm_epochs`` epochs (0 = the full ``settings.epochs``); the first fold is always fresh;
* the fit settings carry the identity's seed, so ``Trainer.fit`` accepts the identity.

``make_trainer(seed)`` returns a new Trainer; ``data`` is the 4-tuple ``(x_train, y_train, x_val, y_val)`` the
Trainer takes (tensors, dicts of tensors or lazy inputs, see trainer/batching.py).
"""

import dataclasses
from collections.abc import Callable
from pathlib import Path

from nig_timenet_torch.core.seeds import seed_everything
from nig_timenet_torch.trainer.identity import RunIdentity
from nig_timenet_torch.trainer.trainer import FitResult, FitSettings, Trainer

__all__ = ["fit_fold", "fold_dir", "INITS"]

INITS = ("fresh", "warm")


def fold_dir(run_dir: str | Path, fold: int, seed: int) -> Path:
    return Path(run_dir) / f"fold_{fold:03d}_seed_{seed}"


def fit_fold(make_trainer: Callable[[int], Trainer], data: tuple, settings: FitSettings, identity: RunIdentity, *,
             previous: Trainer | None = None, init: str = "fresh", warm_epochs: int = 0,
             run_dir: str | Path | None = None, deadline: float | None = None,
             on_epoch: Callable[[dict], None] | None = None) -> tuple[Trainer, FitResult]:
    """Fit fold ``identity.fold`` and return ``(trainer, result)``; the trainer holds the best weights afterwards."""
    if init not in INITS:
        raise ValueError(f"init must be one of {INITS}, not {init!r}")
    seed = identity.seed
    warm = init == "warm" and previous is not None
    fold_settings = dataclasses.replace(settings, seed=seed,
                                        epochs=warm_epochs if warm and warm_epochs else settings.epochs)
    if warm:
        trainer = previous
    else:
        seed_everything(seed)
        trainer = make_trainer(seed)
    folder = None if run_dir is None else fold_dir(run_dir, identity.fold or 0, seed)
    result = trainer.fit(*data, fold_settings, run_dir=folder, identity=identity, deadline=deadline,
                         on_epoch=on_epoch)
    return trainer, result
