"""PURPOSE: The periods of a backtest (train / val / test) from the split dates or the signals' stage, and the one rule
    that results are never mixed across them: train is always alone; val and test may be merged on request.
TAGS: periods, train val test, split dates, stage, merge val test, no mixing, holdout, protocol
PITFALLS: a row's period comes from its ``stage`` column when the signals have it (enabled), else from ``time``
    against ``train_end`` / ``val_end`` (time < train_end = train, < val_end = val, the rest = test), else every row is
    "all". A trade belongs to the period of its SIGNAL. ``groups`` never returns a group holding train and another
    period together.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["Periods", "assign", "groups", "TRAIN", "VAL", "TEST", "VAL_TEST", "ALL"]

TRAIN, VAL, TEST, VAL_TEST, ALL = "train", "val", "test", "val+test", "all"


@dataclass(frozen=True)
class Periods:
    """The split dates (UTC; the dates the model's run split on): ``train_end`` < ``val_end``."""
    train_end: str | None = None
    val_end: str | None = None

    def __post_init__(self):
        if (self.train_end is None) != (self.val_end is None):
            raise ValueError("Periods needs both train_end and val_end, or neither")
        if self.train_end is not None and pd.Timestamp(self.train_end) >= pd.Timestamp(self.val_end):
            raise ValueError("Periods: train_end must be before val_end")


def assign(rows: pd.DataFrame, periods: Periods | None) -> np.ndarray:
    """The period of each row: its stage, else its time against the dates, else "all"."""
    if "stage" in rows:
        return rows["stage"].astype(str).to_numpy()
    if periods is None or periods.train_end is None:
        return np.full(len(rows), ALL, dtype=object)
    t = pd.to_datetime(rows["time"]).to_numpy()
    a, b = np.datetime64(pd.Timestamp(periods.train_end)), np.datetime64(pd.Timestamp(periods.val_end))
    return np.where(t < a, TRAIN, np.where(t < b, VAL, TEST)).astype(object)


def groups(labels, merge_val_test: bool = False) -> dict[str, np.ndarray]:
    """{group name: row mask}: one group per period, with val and test also merged into "val+test" when asked; train
    is never part of a merged group."""
    labels = np.asarray(labels, dtype=object)
    out = {str(p): labels == p for p in pd.unique(labels)}
    if merge_val_test and VAL in out and TEST in out:
        out[VAL_TEST] = out[VAL] | out[TEST]
    order = [TRAIN, VAL, TEST, VAL_TEST]
    return dict(sorted(out.items(), key=lambda kv: (order.index(kv[0]) if kv[0] in order else 9, kv[0])))
