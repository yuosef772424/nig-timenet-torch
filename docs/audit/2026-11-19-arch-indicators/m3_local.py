"""M3 local look (PREREGISTERED_M3.md): the ridge bound and the reading of arms A (raw) and M (market residual).

    python m3_local.py ridge RAW_ARTIFACT RES_ARTIFACT
    python m3_local.py read --arm A=OUT_DIR [--arm ...] --arm M=OUT_DIR [--arm ...]

``read`` joins every arm's TEST rows to M's residual returns (same rows, by coin and time). A study script (rule G6).
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.stats.metrics import block_bootstrap_ci

COST = 0.0016
FIRST_TEST, SPAN, FOLDS, EMBARGO = pd.Timestamp("2024-01-01"), pd.Timedelta("180D"), 4, pd.Timedelta("2D")


def ic_by_time(pred, y, t) -> float:
    df = pd.DataFrame({"p": pred, "y": y, "t": t})
    vals = [g["p"].rank().corr(g["y"].rank()) for _, g in df.groupby("t") if len(g) >= 5]
    return float(np.nanmean(vals))


def _feats_raw(x):
    c, h, lo = x[..., 2], x[..., 0], x[..., 1]
    return np.column_stack([c[:, -1], c[:, -6:].sum(1), c[:, -18:].sum(1), c.sum(1), h[:, -1] - lo[:, -1],
                            (h - lo)[:, -6:].mean(1)])


def _feats_res(x):
    c = x[..., 2]
    return np.column_stack([c[:, -1], c[:, -6:].sum(1), c[:, -18:].sum(1), c.sum(1), x[:, -1, 0] - x[:, -1, 1],
                            x[:, -1, 3], x[:, -1, 4], x[:, -1, 6], x[:, -1, 7], x[:, -6:, 7].sum(1), x[:, -1, 9]])


def _ridge(xtr, ytr, xte, alpha=1.0):
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-12
    a, b = (xtr - mu) / sd, (xte - mu) / sd
    w = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ (ytr - ytr.mean()))
    return b @ w


def ridge(raw: str, res: str) -> None:
    dr, dm = npy_dir.load(raw, mmap=False), npy_dir.load(res, mmap=False)
    t = pd.to_datetime(np.asarray(dr["last_candles"])[:, LAST_COLUMN_INDEX["timestamp"]].astype(np.int64))
    y_own, y_res = np.asarray(dr["y_close_reg"], float), np.asarray(dm["y_close_reg"], float)
    fr, fm = _feats_raw(np.asarray(dr["X_4h"], float)), _feats_res(np.asarray(dm["X_4h"], float))
    ok = (np.abs(np.asarray(dm["X_4h"])).sum(axis=(1, 2)) > 0)
    rows = []
    for k in range(FOLDS):
        lo = FIRST_TEST + k * SPAN
        tr = ok & (t < lo - EMBARGO)
        te = ok & (t >= lo) & (t < lo + SPAN)
        r = {"fold": k, "test_from": str(lo.date()), "train": int(tr.sum()), "test": int(te.sum())}
        for name, f, y_fit in (("raw->own", fr, y_own), ("raw->res", fr, y_res), ("res->res", fm, y_res)):
            p = _ridge(f[tr], y_fit[tr], f[te])
            r[f"{name} IC own"] = ic_by_time(p, y_own[te], t[te])
            r[f"{name} IC res"] = ic_by_time(p, y_res[te], t[te])
        rows.append(r)
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        print(df.to_string(index=False))
        print(df.drop(columns=["fold", "test_from", "train", "test"]).mean().to_string())


def _folds(path: Path):
    for d in sorted(path.glob("fold_*_seed_*")):
        m = re.match(r"fold_(\d+)_seed_(\d+)", d.name)
        yield int(m.group(1)), int(m.group(2)), pd.read_parquet(d / "battery_table_val.parquet"), \
            pd.read_parquet(d / "battery_table_test.parquet")


def read(arms: list[str]) -> None:
    specs = [s.split("=", 1) for s in arms]
    res_y = {}
    for name, path in specs:
        if name == "M":
            for f, s, _, te in _folds(Path(path)):
                res_y[(f, s)] = te.set_index(["coin", "end_time"])["y_ret"]
    rows = []
    for name, path in specs:
        per_seed: dict[int, list] = {}
        for f, s, va, te in _folds(Path(path)):
            yr = res_y[(f, s)]
            te = te.join(yr.rename("y_res"), on=["coin", "end_time"]).dropna(subset=["y_res"])
            te["long"] = te["mu"] >= float(np.quantile(va["mu"], 0.9))
            te["fold"] = f
            per_seed.setdefault(s, []).append(te)
        for s, parts in per_seed.items():
            te = pd.concat(parts)
            lg = te[te["long"]]
            days = lg.groupby(pd.to_datetime(lg["end_time"]).dt.floor("D"))["y_res"].mean()
            m, lo, hi = block_bootstrap_ci(days.to_numpy(), np.arange(len(days)), level=0.95, n_boot=2000)
            fold_ic = [ic_by_time(g["mu"], g["y_res"], g["end_time"]) for _, g in te.groupby("fold")]
            rows.append({"arm": name, "seed": s, "rows": len(te), "IC vs residual": ic_by_time(te["mu"], te["y_res"],
                         te["end_time"]), "IC per fold": " ".join(f"{v:+.3f}" for v in fold_ic),
                         "top10 trades": int(len(lg)), "top10 residual gross per trade": float(lg["y_res"].mean()),
                         "top10 residual day mean": m, "CI lo": lo, "CI hi": hi,
                         "without best 5% days": float(days[days < days.quantile(0.95)].mean())})
    with pd.option_context("display.width", 300, "display.max_columns", 20, "display.float_format", "{:.5f}".format):
        print(pd.DataFrame(rows).to_string(index=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("ridge")
    r.add_argument("raw")
    r.add_argument("res")
    q = sub.add_parser("read")
    q.add_argument("--arm", action="append", required=True)
    a = ap.parse_args(argv)
    ridge(a.raw, a.res) if a.cmd == "ridge" else read(a.arm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
