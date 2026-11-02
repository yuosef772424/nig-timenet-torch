"""Checkpoint files of a run folder (ROADMAP T1; TF source: trainer/checkpoints.py, simplified).

``checkpoint.pt``  training state after the last finished epoch: model, Kendall log-variances, optimizer, tracker
                   state, history, RunIdentity. Resume continues from exactly this, so it is NOT replaced by the
                   best weights at the end of fit (TF re-saved after restoring the best weights, which made a
                   resumed run differ from an uninterrupted one).
``best.pt``        model weights of the best epoch (+ ``best_meta.json``: value, epoch, metric).
Every write goes to a temp file first and is moved into place with os.replace (a Colab disconnect cannot leave a
half-written file). Loading uses ``weights_only=True``: only tensors and plain containers are accepted.
"""

import json
import os
from pathlib import Path

import torch

__all__ = ["CHECKPOINT", "BEST", "BEST_META", "save_checkpoint", "load_checkpoint", "save_best", "load_best"]

CHECKPOINT, BEST, BEST_META = "checkpoint.pt", "best.pt", "best_meta.json"


def _atomic_write(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def save_checkpoint(run_dir: Path, state: dict) -> None:
    _atomic_write(Path(run_dir) / CHECKPOINT, lambda p: torch.save(state, p))


def load_checkpoint(run_dir: Path) -> dict | None:
    path = Path(run_dir) / CHECKPOINT
    return torch.load(path, map_location="cpu", weights_only=True) if path.is_file() else None


def save_best(run_dir: Path, weights: dict, meta: dict) -> None:
    _atomic_write(Path(run_dir) / BEST, lambda p: torch.save(weights, p))
    _atomic_write(Path(run_dir) / BEST_META, lambda p: p.write_text(json.dumps(meta, sort_keys=True), encoding="utf-8"))


def load_best(run_dir: Path) -> dict:
    path = Path(run_dir) / BEST
    if not path.is_file():
        raise FileNotFoundError(f"{path}: the checkpoint records a best epoch but best weights are missing")
    return torch.load(path, map_location="cpu", weights_only=True)
