"""Determinism stub (full version in BUILD_PLAN.md P23).

Mirrors the determinism pattern of ``reference/export_reference.py`` (seeds python/numpy/backend, deterministic ops,
single thread) for torch: ``torch.use_deterministic_algorithms(True)``, ``torch.set_num_threads(1)``,
``torch.manual_seed``.
"""

import os
import random

import numpy as np
import torch


def set_determinism(seed: int = 0) -> None:
    """Seed python, numpy and torch with ``seed`` and switch torch to deterministic, single-threaded CPU execution."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
