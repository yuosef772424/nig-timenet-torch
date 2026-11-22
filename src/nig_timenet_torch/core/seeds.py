"""PURPOSE: ONE seeding entry point (``seed_everything``) and the named per-epoch / per-stage reseeding formulas that
    ``Trainer.fit`` and the staged loop share, so a resumed run reseeds exactly as an uninterrupted one.
TAGS: seeds, determinism, seed everything, epoch seed, reseed, resume, reproducibility
PITFALLS: the multipliers below are part of every run's numbers: changing one changes every trained model and breaks
    resume of old checkpoints. ``seed_everything`` turns on ``torch.use_deterministic_algorithms`` and one thread.

C3, BUILD_PLAN.md P23: call ``seed_everything(seed)`` before any dataset / model construction. Mirrors
reference/export_reference.py's determinism (python/numpy/backend seeds, deterministic ops, single thread) for torch.
"""

import os
import random

import numpy as np
import torch

__all__ = ["seed_everything", "epoch_seed", "stage_probe_seed", "bn_refresh_seed", "EPOCH_SEED_MULT",
           "STAGE_PROBE_SEED_MULT", "BN_REFRESH_SEED_MULT"]

#: ``torch.manual_seed(seed * EPOCH_SEED_MULT + epoch)`` at the start of every epoch (and every staged pass)
EPOCH_SEED_MULT = 1_000_003
#: the staged loop's own generator that draws the stage's probe rows: ``seed * STAGE_PROBE_SEED_MULT + stage``
STAGE_PROBE_SEED_MULT = 7919
#: the staged loop's own generator of the BatchNorm refresh batches: ``seed * BN_REFRESH_SEED_MULT + stage``
BN_REFRESH_SEED_MULT = 104729


def seed_everything(seed: int) -> None:
    """random.seed, numpy.random.seed, torch.manual_seed, torch.use_deterministic_algorithms(True),
    torch.set_num_threads(1) (and PYTHONHASHSEED for child processes)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def epoch_seed(seed: int, index: int) -> int:
    """The global RNG seed of epoch (or staged pass) ``index``: a pure function of (seed, index)."""
    return seed * EPOCH_SEED_MULT + index


def stage_probe_seed(seed: int, stage: int) -> int:
    """Seed of the staged loop's own generator that picks a stage's probe rows."""
    return seed * STAGE_PROBE_SEED_MULT + stage


def bn_refresh_seed(seed: int, stage: int) -> int:
    """Seed of the staged loop's own generator that draws a stage's BatchNorm refresh batches."""
    return seed * BN_REFRESH_SEED_MULT + stage
