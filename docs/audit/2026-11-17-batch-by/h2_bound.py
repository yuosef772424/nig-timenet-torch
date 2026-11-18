"""H2 of PREREGISTERED_2.md: the model-free bound on the SAME test rows of every walk-forward fold, against the model
arms. (i) the 6-bar mean range of the input window; (ii) a ridge on ``doctor.common.simple_features`` fitted on the
fold's TRAINING rows, penalty chosen on the fold's inner validation (rank IC of the head). Scored by rank IC by
timestamp. Prints markdown and the verdict of the pre-registered rule.

    python docs/audit/2026-11-17-batch-by/h2_bound.py ARTIFACT STUDY
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.data_io.loading import LoaderSettings, load_dataset
from nig_timenet_torch.doctor.common import simple_features
from nig_timenet_torch.stats import metrics
from nig_timenet_torch.workflow.rolling import fold_with_val, rolling_folds
from nig_timenet_torch.workflow.run_common import input_key
from nig_timenet_torch.workflow.settings import RollingSettings

# the geometry of run_arms.py (kept equal by hand: a study does not import another study, rule G6)
ROLLING = RollingSettings(mode="expanding", train_span="365D", test_span="90D", step="90D", embargo="1D", max_folds=4)
LAMBDAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
ARMS = ("A0", "A2", "C34")


def ridge_fit(a: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Coefficients (with intercept last) of a ridge on standardised columns; the scaling is folded in."""
    mu, sd = a.mean(axis=0), a.std(axis=0)
    sd[sd == 0] = 1.0
    z = (a - mu) / sd
    w = np.linalg.solve(z.T @ z + lam * np.eye(z.shape[1]), z.T @ (y - y.mean()))
    return np.concatenate([w / sd, [y.mean() - (mu / sd) @ w]])


def predict(a: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return a @ coef[:-1] + coef[-1]


def ic(y, pred, ts) -> float:
    return metrics.rank_ic(np.asarray(y), np.asarray(pred), np.asarray(ts))


def main(artifact: str, study: str) -> int:
    ds = load_dataset(LoaderSettings(name=artifact, mmap=True, local_dir="")).dataset
    x = np.asarray(ds[input_key(ds)])
    feats, _ = simple_features(x, ds["feature_order"])
    ts = np.asarray(ds["last_candles"])[:, TS_COL].astype(np.int64)
    rng6 = (x[:, -6:, 0] - x[:, -6:, 1]).mean(axis=1)
    models = {}
    for arm in ARMS:      # the arm's folder, or its per-seed folders <arm>_s<seed>
        files = [d / "rolling_folds.csv" for d in [Path(study) / arm, *sorted(Path(study).glob(f"{arm}_s*"))]]
        files = [f for f in files if f.exists()]
        if files:
            models[arm] = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    rows = []
    for fold in rolling_folds(ds, ROLLING):
        plan, _ = fold_with_val(fold, ROLLING.embargo, ROLLING.val_share)
        tr, va, te = plan.train_idx, plan.val_idx, fold.plan.test_idx
        for head, sign in (("high", 1.0), ("low", -1.0)):
            y = np.asarray(ds[f"y_{head}_reg"])
            best = max(LAMBDAS, key=lambda lam: ic(y[va], predict(feats[va], ridge_fit(feats[tr], y[tr], lam)),
                                                   ts[va]))
            ridge = ic(y[te], predict(feats[te], ridge_fit(feats[tr], y[tr], best)), ts[te])
            row = {"fold": fold.index, "head": head, "range6": ic(y[te], sign * rng6[te], ts[te]), "ridge": ridge,
                   "lambda": best, "test_rows": len(te)}
            for arm, f in models.items():
                row[arm] = float(f[f.fold == fold.index][f"targets.{head}.skill.rank_ic_by_timestamp"].mean())
            rows.append(row)
    df = pd.DataFrame(rows)
    df["bound"] = df[["range6", "ridge"]].max(axis=1)
    print("## H2: model-free bound vs the model, rank IC by timestamp on each fold's test rows\n")
    cols = ["fold", "head", "test_rows", "range6", "ridge", "lambda", *models, "bound"]
    print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in df[cols].itertuples(index=False):
        print("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    print("\n## Pre-registered verdict (bound above A0 on BOTH heads in >= 3 of 4 folds)\n")
    for arm in models:
        wins = {h: int((g["bound"] > g[arm]).sum()) for h, g in df.groupby("head")}
        verdict = "supported" if all(w >= 3 for w in wins.values()) else "refuted"
        tag = "" if arm == "A0" else " (read, not judged)"
        print(f"- {arm}{tag}: bound above the model in {wins} folds -> H2 {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
