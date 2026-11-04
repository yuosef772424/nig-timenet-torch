"""Row access shared by the trainer for every kind of input: a tensor (M, T, F), a dict of tensors (a panel input is
``{"x": (D, N, T, F), "mask": (D, N)}``: the row is a day, not a coin), or any object that has ``__len__`` and
``__getitem__(index) -> tensor | dict`` (a lazy input that reads from disk only the rows asked for).

For a plain tensor every function below is exactly the expression the trainer used before (``len(x)``, ``x[ix]``,
``x.to(dev)``), so the flat model is unchanged.
"""

import numpy as np
import torch

__all__ = ["n_rows", "take", "to_device"]


def n_rows(x) -> int:
    return n_rows(next(iter(x.values()))) if isinstance(x, dict) else len(x)


def take(x, index):
    """Rows ``index`` (a tensor of positions or a slice) of ``x``, same container type."""
    return {k: take(v, index) for k, v in x.items()} if isinstance(x, dict) else x[index]


def to_device(x, device: torch.device):
    """``x`` on ``device``; numpy arrays inside a dict (a lazy input reads numpy) become tensors first."""
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return (torch.as_tensor(x) if isinstance(x, np.ndarray) else x).to(device)
