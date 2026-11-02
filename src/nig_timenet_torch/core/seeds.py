"""ONE seeding entry point (C3; BUILD_PLAN.md P23): call ``seed_everything(seed)`` before any dataset / model
construction. Mirrors reference/export_reference.py's determinism (python/numpy/backend seeds, deterministic
ops, single thread) for torch."""

import os
import random

import numpy as np
import torch

__all__ = ["seed_everything", "set_determinism"]


def seed_everything(seed: int) -> None:
    """random.seed, numpy.random.seed, torch.manual_seed, torch.use_deterministic_algorithms(True),
    torch.set_num_threads(1) (and PYTHONHASHSEED for child processes)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def set_determinism(seed: int = 0) -> None:
    """P00 name, kept as an alias of seed_everything."""
    seed_everything(seed)
