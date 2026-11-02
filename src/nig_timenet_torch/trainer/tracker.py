"""Best epoch + early stopping in one resumable object (ROADMAP T1; source: crypto-signal-prediction
trainer/best_tracker.py, same at aeafc564 and cea9cd58). What "better" means comes from the registered selection
metric (``higher_is_better`` of its SELECTION_METRICS entry, rule C3), never from loss weights.

The tracker holds numbers only (smoothed metric, best, wait). The weights of the best epoch are kept by the
Trainer (trainer/checkpoint.py). Not ported: ``weights_snapshot="ema_weights"`` (needs optimizer EMA).
"""

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["EarlyStopSettings", "BestTracker", "SMOOTHERS"]


def _smooth_none(t: "BestTracker", x: float) -> float:
    return x


def _smooth_window(t: "BestTracker", x: float) -> float:
    return float(np.mean(t.history[-t.settings.window:]))


def _smooth_ema(t: "BestTracker", x: float) -> float:
    beta = t.settings.ema_beta
    t.ema = x if t.ema is None else beta * t.ema + (1.0 - beta) * x
    return float(t.ema)


#: smoothing name -> f(tracker, raw value) -> smoothed value (history already contains the raw value)
SMOOTHERS = {"none": _smooth_none, "window": _smooth_window, "ema": _smooth_ema}


@dataclass(frozen=True)
class EarlyStopSettings:
    patience: int = 15
    min_delta: float = 1e-4
    smoothing: str = "window"          # "none" | "window" (mean of the last `window` epochs) | "ema"
    window: int = 3
    ema_beta: float = 0.7              # s_t = beta * s_(t-1) + (1 - beta) * x_t, s_1 = x_1
    restore_best_weights: bool = True

    def __post_init__(self):
        if self.smoothing not in SMOOTHERS:
            raise ValueError(f"unknown smoothing {self.smoothing!r}; known: {sorted(SMOOTHERS)}")
        if self.patience < 1 or self.window < 1 or not 0.0 <= self.ema_beta < 1.0 or self.min_delta < 0:
            raise ValueError("need patience >= 1, window >= 1, 0 <= ema_beta < 1, min_delta >= 0")


class BestTracker:
    def __init__(self, settings: EarlyStopSettings, higher_is_better: bool):
        self.settings, self.higher_is_better = settings, bool(higher_is_better)
        self.history: list[float] = []
        self.ema: float | None = None
        self.best: float | None = None
        self.best_epoch: int | None = None
        self.wait = 0

    def update(self, epoch: int, value: float) -> bool:
        """Feed the selection metric of the finished ``epoch`` (1-based). True if it is the new best. A non-finite
        value is ignored (it neither improves nor counts as a wasted epoch), as in the TF source."""
        if value is None or not math.isfinite(value):
            return False
        self.history.append(float(value))
        smoothed = SMOOTHERS[self.settings.smoothing](self, float(value))
        if self._is_better(smoothed):
            self.best, self.best_epoch, self.wait = smoothed, epoch, 0
            return True
        self.wait += 1
        return False

    def _is_better(self, s: float) -> bool:
        if self.best is None:
            return True
        delta = self.settings.min_delta
        return s > self.best + delta if self.higher_is_better else s < self.best - delta

    def should_stop(self) -> bool:
        return self.wait >= self.settings.patience

    def get_state(self) -> dict:
        return {"higher_is_better": self.higher_is_better, "history": self.history[-200:], "ema": self.ema,
                "best": self.best, "best_epoch": self.best_epoch, "wait": int(self.wait)}

    def load_state(self, state: dict) -> None:
        if state["higher_is_better"] != self.higher_is_better:
            raise ValueError("saved tracker state was made with the opposite metric direction")
        self.history, self.ema = list(state["history"]), state["ema"]
        self.best, self.best_epoch, self.wait = state["best"], state["best_epoch"], int(state["wait"])
