"""Workflow settings (BUILD_PLAN.md P37): the RunSettings-equivalent section that carries the rolling seam.

``RunSettings.rolling`` defaults to None = today's single global split (P25), unchanged. RollingSettings has NO
defaults for the spans: the plan leaves them to the experiment card (E-roll-001, out of scope), so building one
without them raises instead of inventing values.
"""

from dataclasses import dataclass

import pandas as pd

__all__ = ["RollingSettings", "RunSettings", "ROLLING_MODES"]

ROLLING_MODES = ("expanding", "sliding")


@dataclass(frozen=True, kw_only=True)
class RollingSettings:
    mode: str = "expanding"            # "expanding" | "sliding"
    train_span: pd.Timedelta           # expanding: span of the FIRST fold's train window; sliding: every fold's
    test_span: pd.Timedelta
    step: pd.Timedelta                 # how far train_end advances per fold
    embargo: pd.Timedelta
    max_folds: int | None = None
    retrain_every_k: int = 1           # retrain on fold i iff i % k == 0, else reuse the last trained model
    seeds: tuple[int, ...] = (0,)

    def __post_init__(self):
        if self.mode not in ROLLING_MODES:
            raise ValueError(f"mode must be one of {ROLLING_MODES}, not {self.mode!r}")
        for name in ("train_span", "test_span", "step", "embargo"):
            object.__setattr__(self, name, pd.Timedelta(getattr(self, name)))
        if min(self.train_span, self.test_span, self.step) <= pd.Timedelta(0) or self.embargo < pd.Timedelta(0):
            raise ValueError("train_span, test_span, step must be > 0 and embargo >= 0")
        if self.retrain_every_k < 1 or (self.max_folds is not None and self.max_folds < 1):
            raise ValueError("retrain_every_k and max_folds must be >= 1")
        object.__setattr__(self, "seeds", tuple(int(s) for s in self.seeds))


@dataclass(frozen=True)
class RunSettings:
    rolling: RollingSettings | None = None   # None = the single global split (P25), unchanged
