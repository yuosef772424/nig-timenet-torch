"""P38: staleness seam — pure functions over already-written PredictionTables (P27); nothing is trained here and
nothing in this repo concludes anything from them (no multi-fold trained model exists in this plan).

Distance from train_end is measured at the moment the prediction is made: ``days_since_train_end =
floor((prediction_made_at - train_end) / 1 day)``. A PredictionTable carries outputs only, not labels, so the
caller joins the label column it wants to permute (default name ``"label"``) before calling
``label_permutation_null``.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

__all__ = ["staleness_curve", "label_permutation_null", "DAY"]

DAY = pd.Timedelta("1D")


def staleness_curve(prediction_tables: list[pd.DataFrame], metric_fn: Callable[[pd.DataFrame], float]
                    ) -> pd.DataFrame:
    """One row per (fold, days_since_train_end): metric_fn applied to that row group, plus its row count."""
    rows = []
    for table in prediction_tables:
        days = ((table["prediction_made_at"] - table["train_end"]) // DAY).astype(np.int64)
        if (days < 0).any():
            raise ValueError("prediction_made_at before train_end: not an out-of-sample table")
        for (fold, d), group in table.groupby([table["fold"], days], sort=True):
            rows.append({"fold": int(fold), "days_since_train_end": int(d), "metric": float(metric_fn(group)),
                         "n": int(len(group))})
    return pd.DataFrame(rows, columns=["fold", "days_since_train_end", "metric", "n"])


def label_permutation_null(table: pd.DataFrame, metric_fn: Callable[[pd.DataFrame], float], n_perm: int = 200,
                           seed: int = 0, *, label_col: str = "label") -> np.ndarray:
    """Null distribution of metric_fn: ``label_col`` shuffled n_perm times with np.random.default_rng(seed)
    (fresh per call, so the same seed gives the identical array)."""
    if label_col not in table.columns:
        raise KeyError(f"{label_col!r} not in table; join the label column first")
    rng = np.random.default_rng(seed)
    labels = table[label_col].to_numpy()
    out = np.empty(n_perm, dtype=np.float64)
    shuffled = table.copy()
    for i in range(n_perm):
        shuffled[label_col] = labels[rng.permutation(len(labels))]
        out[i] = metric_fn(shuffled)
    return out
