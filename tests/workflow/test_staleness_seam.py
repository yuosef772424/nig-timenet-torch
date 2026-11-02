"""P38: staleness seam on SYNTHETIC PredictionTables only (no conclusion about retraining is drawn anywhere)."""

import numpy as np
import pandas as pd

from nig_timenet_torch.stats.metrics import auc
from nig_timenet_torch.workflow.staleness import label_permutation_null, staleness_curve


def synthetic_table(fold: int, train_end: str, n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    te = pd.Timestamp(train_end, tz="UTC")
    made = te + pd.to_timedelta(rng.integers(0, 5 * 24, n), unit="h")
    label = (rng.random(n) < 0.5).astype(float)
    return pd.DataFrame({"timestamp": made - pd.Timedelta("1h"), "prediction_made_at": made,
                         "train_end": pd.Series([te] * n), "fold": np.full(n, fold, dtype=np.int64),
                         "y_close_class_logits": np.clip(label * 0.3 + rng.random(n) * 0.7, 0, 1), "label": label})


def metric(df: pd.DataFrame) -> float:
    return auc(df["label"].to_numpy(), df["y_close_class_logits"].to_numpy())


def test_staleness_curve_runs_on_synthetic_tables():
    tables = [synthetic_table(0, "2025-01-10"), synthetic_table(1, "2025-01-20", seed=1)]
    curve = staleness_curve(tables, metric)
    assert list(curve.columns) == ["fold", "days_since_train_end", "metric", "n"]
    assert set(curve["fold"]) == {0, 1} and set(curve["days_since_train_end"]) == {0, 1, 2, 3, 4}
    assert curve.groupby("fold")["n"].sum().tolist() == [120, 120]
    assert np.isfinite(curve["metric"]).all()


def test_label_permutation_null_is_reproducible_with_seed():
    t = synthetic_table(0, "2025-01-10")
    a, b = label_permutation_null(t, metric, n_perm=50, seed=3), label_permutation_null(t, metric, n_perm=50, seed=3)
    c = label_permutation_null(t, metric, n_perm=50, seed=4)
    assert a.shape == (50,) and np.array_equal(a, b) and not np.array_equal(a, c)
    assert t["label"].equals(synthetic_table(0, "2025-01-10")["label"])  # input not mutated
