"""Read the DW1 arms (PREREGISTERED.md in this folder): day-loss tail, day-IC tail, long book, paired per-day diffs.

    python dw1_read.py --arm NAME=OUT_DIR [--arm ...] --ref R --out DIR [--stage test]

OUT_DIR is a rolling job's ``--out`` for ONE seed (folders ``fold_<k>_seed_<s>`` holding ``battery_table_val.parquet``
and ``battery_table_<stage>.parquet``); give the same NAME once per seed. ``--ref``: the arm the paired differences are
taken against. Writes ``dw1_<stage>.csv`` (one row per arm x seed), ``dw1_folds_<stage>.csv`` and ``dw1_days_<stage>
.parquet`` (the per-day values of every arm x seed). A study script (rule G6): reads files, no training.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.stats.metrics import block_bootstrap_ci

COST = 2 * 0.0008
Q = 0.90
MIN_COINS = 5


def stage_table(out_dir: Path, stage: str) -> pd.DataFrame:
    """Every fold's stage table of one seed with the long flag (VAL 90th percentile of mu per fold)."""
    parts = []
    for d in sorted(out_dir.glob("fold_*_seed_*")):
        m = re.match(r"fold_(\d+)_seed_(\d+)", d.name)
        vp, sp = d / "battery_table_val.parquet", d / f"battery_table_{stage}.parquet"
        if not (vp.is_file() and sp.is_file()):
            continue
        val, st = pd.read_parquet(vp), pd.read_parquet(sp).copy()
        st["long"] = st["mu"] >= float(np.quantile(val["mu"], Q))
        st["fold"], st["seed"] = int(m.group(1)), int(m.group(2))
        parts.append(st)
    if not parts:
        raise FileNotFoundError(f"no fold tables under {out_dir}")
    st = pd.concat(parts, ignore_index=True)
    st["day"] = pd.to_datetime(st["end_time"]).dt.floor("D")
    sig = st["sigma"].clip(lower=1e-6)
    st["nll"] = 0.5 * np.log(2 * np.pi * sig ** 2) + 0.5 * ((st["y_ret"] - st["mu"]) / sig) ** 2
    return st


def day_values(st: pd.DataFrame) -> pd.DataFrame:
    """Per TEST day: mean NLL, mean timestamp rank IC, long-book mean net, always-long mean net, trades."""
    ics = []
    for _t, g in st.groupby("end_time"):
        if len(g) >= MIN_COINS and g["mu"].nunique() > 1 and g["y_ret"].nunique() > 1:
            ics.append((g["day"].iloc[0], g["mu"].rank().corr(g["y_ret"].rank())))
    ic = pd.DataFrame(ics, columns=["day", "ic"]).groupby("day")["ic"].mean()
    lg = st[st["long"]]
    days = pd.DataFrame({"nll": st.groupby("day")["nll"].mean(), "ic": ic,
                         "always_long": st.groupby("day")["y_ret"].mean() - COST})
    days["long"] = (lg["y_ret"] - COST).groupby(lg["day"]).mean()
    days["trades"] = lg.groupby("day").size()
    days["fold"] = st.groupby("day")["fold"].first()
    return days.fillna({"trades": 0})


def tail(v: pd.Series, share: float, worst_high: bool) -> float:
    v = v.dropna().sort_values(ascending=not worst_high)
    k = max(1, int(round(share * len(v))))
    return float(v.iloc[:k].mean())


def summary(days: pd.DataFrame) -> dict:
    lng = days["long"].dropna()
    return {"days": int(len(days)), "nll_mean": float(days["nll"].mean()),
            "nll_cvar10": tail(days["nll"], 0.10, worst_high=True),
            "ic_mean": float(days["ic"].mean()), "ic_low10": tail(days["ic"], 0.10, worst_high=False),
            "ic_pos_days": float((days["ic"].dropna() > 0).mean()),
            "long_mean": float(lng.mean()), "long_wo_best5": float(lng[lng < lng.quantile(0.95)].mean()),
            "long_cvar5": tail(lng, 0.05, worst_high=False), "long_win_days": float((lng > 0).mean()),
            "long_days": int(len(lng)), "trades": int(days["trades"].sum()),
            "always_long_mean": float(days["always_long"].mean())}


def paired(a: pd.DataFrame, b: pd.DataFrame, col: str) -> dict:
    d = (a[col] - b[col]).dropna()
    if len(d) < 10:
        return {f"d_{col}": np.nan}
    mean, lo, hi = block_bootstrap_ci(d.to_numpy(), np.arange(len(d)) // 5, level=0.95, n_boot=2000)  # 5-day blocks
    return {f"d_{col}": mean, f"d_{col}_lo": lo, f"d_{col}_hi": hi}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--ref", default="R")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="test", choices=("val", "test"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    days_of, frows = {}, []
    for spec in a.arm:
        name, path = spec.split("=", 1)
        st = stage_table(Path(path), a.stage)
        for seed, s in st.groupby("seed"):
            days = day_values(s)
            days_of[(name, int(seed))] = days
            for fold, f in days.groupby("fold"):
                frows.append({"arm": name, "seed": int(seed), "fold": int(fold), **summary(f)})
    rows = []
    for (name, seed), days in sorted(days_of.items()):
        r = {"arm": name, "seed": seed, **summary(days)}
        ref = days_of.get((a.ref, seed))
        if ref is not None and name != a.ref:
            for col in ("nll", "ic", "long"):
                r.update(paired(days, ref, col))
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(out / f"dw1_{a.stage}.csv", index=False)
    pd.DataFrame(frows).to_csv(out / f"dw1_folds_{a.stage}.csv", index=False)
    pd.concat({f"{n}_s{s}": d for (n, s), d in days_of.items()}, names=["arm_seed", "day"]).to_parquet(
        out / f"dw1_days_{a.stage}.parquet")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.5f}".format):
        print(res.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
