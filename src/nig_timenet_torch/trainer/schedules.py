"""Lambda / learning-rate schedules (ROADMAP T1; source: crypto-signal-prediction trainer/schedules.py, identical at
aeafc564 and cea9cd58). Every function is PURE in the epoch number, which is what makes resume safe: nothing here
is saved in a checkpoint. Unit parity against values computed from the TF source:
tests/trainer/test_schedules_parity.py (reference/data_schedules/schedules.json).

Differences from the TF source, on purpose (rule B5, unknown names raise): an unknown ``schedule`` curve or LR
``kind`` raises ValueError (TF silently used linear / constant); a negative ``warmup_epochs`` raises (TF clamped it
to 0). Defaults of the optional fields equal the ``.get(key, default)`` values of the TF source (see
docs/decisions/D-001-trainer-fit-defaults.md).
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["ParamSchedule", "LrSettings", "schedule_value", "apply_schedules", "cosine_warm_restarts", "lr_at",
           "PROGRESS_CURVES", "LR_KINDS"]

#: progress p in [0, 1) -> weight in [0, 1); the curve of a lambda schedule
PROGRESS_CURVES = {
    "linear": lambda p: p,
    "cosine": lambda p: 0.5 * (1 - np.cos(np.pi * p)),
    "exponential": lambda p: p ** 2,
    "sqrt": lambda p: np.sqrt(p),
}


@dataclass(frozen=True)
class ParamSchedule:
    """One entry of ``config["loss"]["schedules"]``: value ``start`` at epoch 0, ``end`` from ``warmup_epochs`` on."""
    start: float
    end: float
    warmup_epochs: int = 0
    schedule: str = "linear"

    def __post_init__(self):
        if self.schedule not in PROGRESS_CURVES:
            raise ValueError(f"unknown schedule {self.schedule!r}; known: {sorted(PROGRESS_CURVES)}")
        if int(self.warmup_epochs) < 0:
            raise ValueError("warmup_epochs must be >= 0")


def schedule_value(p: ParamSchedule, epoch: int) -> float:
    """Value of a scheduled parameter at ``epoch`` (0-based)."""
    warmup = int(p.warmup_epochs)
    if warmup == 0 or epoch >= warmup:
        return p.end
    return float(p.start + (p.end - p.start) * PROGRESS_CURVES[p.schedule](epoch / warmup))


def apply_schedules(scheduled_vars: dict[str, float], schedules: dict[str, ParamSchedule], epoch: int) -> None:
    """Set every scheduled variable to its value at ``epoch`` (in place)."""
    for name, p in schedules.items():
        scheduled_vars[name] = schedule_value(p, epoch)


def cosine_warm_restarts(epoch: int, lr_initial: float, lr_min: float, cycle_length: int = 10,
                         cycle_mult: float = 1.5) -> float:
    cycle_epoch, current_len = epoch, cycle_length
    while cycle_epoch >= current_len:
        cycle_epoch -= current_len
        current_len = int(current_len * cycle_mult)
    progress = cycle_epoch / current_len
    return float(lr_min + 0.5 * (lr_initial - lr_min) * (1 + np.cos(np.pi * progress)))


def _constant(s: "LrSettings", e: int) -> float:
    return s.lr_initial


def _cosine_once(s: "LrSettings", e: int) -> float:
    progress = min(e / max(s.total_epochs, 1), 1.0)
    return float(s.lr_min + 0.5 * (s.lr_initial - s.lr_min) * (1 + np.cos(np.pi * progress)))


def _restarts(s: "LrSettings", e: int) -> float:
    return cosine_warm_restarts(e, s.lr_initial, s.lr_min, s.cycle_length, s.cycle_mult)


#: LR schedule kind -> f(settings, epochs after warm-up)
LR_KINDS = {"constant": _constant, "cosine": _cosine_once, "cosine_restarts": _restarts}


@dataclass(frozen=True)
class LrSettings:
    """Learning-rate schedule: linear warm-up over ``warmup_epochs`` (lr_initial * (epoch + 1) / warmup), then
    ``kind`` counted from the end of the warm-up. ``cycle_*`` only matter for cosine_restarts, ``total_epochs`` for
    cosine (once, then it stays at lr_min)."""
    lr_initial: float
    lr_min: float
    warmup_epochs: int = 0
    kind: str = "constant"
    cycle_length: int = 10
    cycle_mult: float = 1.5
    total_epochs: int = 100

    def __post_init__(self):
        if self.kind not in LR_KINDS:
            raise ValueError(f"unknown lr kind {self.kind!r}; known: {sorted(LR_KINDS)}")
        if not (self.lr_initial > 0 and self.lr_min >= 0 and self.warmup_epochs >= 0):
            raise ValueError("need lr_initial > 0, lr_min >= 0, warmup_epochs >= 0")
        if self.cycle_length < 1 or self.cycle_mult < 1.0:
            raise ValueError("need cycle_length >= 1 and cycle_mult >= 1.0 (else the restart loop never ends)")


def lr_at(s: LrSettings, epoch: int) -> float:
    """Learning rate used during ``epoch`` (0-based)."""
    if s.warmup_epochs and epoch < s.warmup_epochs:
        return s.lr_initial * (epoch + 1) / s.warmup_epochs
    return LR_KINDS[s.kind](s, epoch - s.warmup_epochs)
