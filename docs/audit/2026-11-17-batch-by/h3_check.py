"""H3 of PREREGISTERED_3.md: A0 vs H (har_range) on one split (train 2023 / test 2024), per quarter of the test year,
against the model-free bound (6-bar range, ridge on the HAR features fitted on the training rows). Prints markdown
and the verdict of rules (a) and (b).

    python docs/audit/2026-11-17-batch-by/h3_check.py ARTIFACT STUDY [CANDIDATE]

CANDIDATE: the version compared with A0 (H in PREREGISTERED_3, N in PREREGISTERED_4).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.data_io.loading import LoaderSettings, load_dataset
from nig_timenet_torch.doctor.common import simple_features
from nig_timenet_torch.stats import metrics
from nig_timenet_torch.workflow.flat_run import row_coins
from nig_timenet_torch.workflow.rolling import fold_with_val, rolling_folds
from nig_timenet_torch.workflow.run_common import input_key
from nig_timenet_torch.workflow.settings import RollingSettings

# the "single" geometry of run_arms.py (kept equal by hand: a study does not import another study)
ROLLING = RollingSettings(mode="expanding", train_span="365D", test_span="350D", step="350D", embargo="1D",
                          max_folds=1)
LAMBDAS = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
VERSIONS = ["A0", "H"]
HEADS = (("high", 1.0), ("low", -1.0))


def ridge_fit(a, y, lam):
    mu, sd = a.mean(axis=0), a.std(axis=0)
    sd[sd == 0] = 1.0
    z = (a - mu) / sd
    w = np.linalg.solve(z.T @ z + lam * np.eye(z.shape[1]), z.T @ (y - y.mean()))
    return np.concatenate([w / sd, [y.mean() - (mu / sd) @ w]])


def ic(y, pred, ts) -> float:
    return metrics.rank_ic(np.asarray(y), np.asarray(pred), np.asarray(ts))


def main(artifact: str, study: str, candidate: str = "H") -> int:
    VERSIONS[1] = candidate
    c = candidate
    ds = load_dataset(LoaderSettings(name=artifact, mmap=True, local_dir="")).dataset
    x = np.asarray(ds[input_key(ds)])
    feats, _ = simple_features(x, ds["feature_order"])
    ts = np.asarray(ds["last_candles"])[:, TS_COL].astype(np.int64)
    rng6 = (x[:, -6:, 0] - x[:, -6:, 1]).mean(axis=1)
    fold = rolling_folds(ds, ROLLING)[0]
    plan, _ = fold_with_val(fold, ROLLING.embargo, ROLLING.val_share)
    tr, va, te = plan.train_idx, plan.val_idx, fold.plan.test_idx
    quarter = pd.to_datetime(ts[te], utc=True).quarter.to_numpy()
    key = pd.DataFrame({"asset": row_coins(ds, te).astype(str),
                        "timestamp": pd.to_datetime(ts[te], utc=True), "row": te})
    preds = {}   # (version, seed) -> {head: predicted mean aligned with te}
    for v in VERSIONS:
        for p in sorted(Path(study).glob(f"{v}*/fold_*_seed_*/test_table.parquet")):
            t = pd.read_parquet(p)
            t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True)
            m = key.merge(t, on=["asset", "timestamp"], how="left", validate="one_to_one")
            seed = int(p.parent.name.split("_seed_")[1])
            preds[(v, seed)] = {h: m[f"y_{h}"].to_numpy() for h, _ in HEADS}
    seeds = sorted({s for _, s in preds})
    rows, whole = [], []
    for h, sign in HEADS:
        y = np.asarray(ds[f"y_{h}_reg"])
        lam = max(LAMBDAS, key=lambda la: ic(y[va], feats[va] @ ridge_fit(feats[tr], y[tr], la)[:-1], ts[va]))
        coef = ridge_fit(feats[tr], y[tr], lam)
        ridge_pred = feats[te] @ coef[:-1] + coef[-1]
        for q in (1, 2, 3, 4):
            s = quarter == q
            r = {"head": h, "quarter": f"Q{q}", "rows": int(s.sum()), "range6": ic(y[te][s], sign * rng6[te][s],
                                                                               ts[te][s]),
                 "ridge": ic(y[te][s], ridge_pred[s], ts[te][s]), "lambda": lam}
            for v in VERSIONS:
                r[v] = float(np.mean([ic(y[te][s], preds[(v, sd)][h][s], ts[te][s]) for sd in seeds]))
            r["bound"] = max(r["range6"], r["ridge"])
            rows.append(r)
        for sd in seeds:
            whole.append({"head": h, "seed": sd, **{v: ic(y[te], preds[(v, sd)][h], ts[te]) for v in VERSIONS}})
    df, wf = pd.DataFrame(rows), pd.DataFrame(whole)
    print(f"seeds {seeds}; test rows {len(te)}; train rows {len(tr)}; validation rows {len(va)}\n")
    print("## Rank IC by timestamp per quarter of 2024 (versions: mean of the seeds)\n")
    cols = ["head", "quarter", "rows", "range6", "ridge", "lambda", "A0", c, "bound"]
    print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in df[cols].itertuples(index=False):
        print("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    print("\n## Whole test year per seed\n")
    print(f"| head | seed | A0 | {c} |\n|---|---|---|---|")
    for r in wf.to_dict("records"):
        print(f"| {r['head']} | {r['seed']} | {r['A0']:.4f} | {r[c]:.4f} |")
    a = {h: int((g[c] >= g["bound"]).sum()) for h, g in df.groupby("head")}
    b_q = {h: int((g[c] > g["A0"]).sum()) for h, g in df.groupby("head")}
    b_s = bool((wf[c] > wf["A0"]).all())
    gap = {h: float(((g[c] - g["A0"]).sum()) / max((g["bound"] - g["A0"]).sum(), 1e-12)) for h, g in
           df.groupby("head")}
    rule_a = all(v >= 3 for v in a.values())
    rule_b = all(v >= 3 for v in b_q.values()) and b_s
    print(f"\n- rule (a), {c} >= bound in quarters: {a} -> {rule_a}")
    print(f"- rule (b), {c} > A0 in quarters: {b_q}; {c} > A0 on every seed and head (whole year): {b_s} -> {rule_b}")
    print(f"- read: share of the A0-to-bound gap closed by {c} (sum over quarters): {json.dumps(gap)}")
    print(f"\n**{c} {'supported' if rule_a and rule_b else 'refuted'}**")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
