"""B1 of docs/results/btc-hedge-long-001.md: buy BTC at the timestamps where the model is bullish across coins.

    python b1_btc_timing.py --arm NAME=OUT_DIR [--arm ...] --out DIR [--level 0.9875]

Rule and baselines: PREREGISTERED_B1.md in this folder (written before any number). Reads the folds' saved
``battery_table_{val,test}.parquet``; no training. A study script (rule G6).
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.stats.metrics import block_bootstrap_ci

COST = 2 * 0.0008
Q_ROW = 0.90
BTC = "BTCUSDT"


def per_time(df: pd.DataFrame, q: float) -> pd.DataFrame:
    """Per timestamp: long share, mean mu, BTC's y_ret and mu (timestamps without a BTC row are dropped)."""
    g = df.groupby("end_time")
    t = pd.DataFrame({"long_share": g["mu"].apply(lambda m: float((m >= q).mean())), "mean_mu": g["mu"].mean()})
    b = df[df["coin"] == BTC].drop_duplicates("end_time").set_index("end_time")
    t["btc_y"], t["btc_mu"] = b["y_ret"], b["mu"]
    t = t.dropna(subset=["btc_y"])
    t["day"] = pd.to_datetime(t.index).floor("D")
    return t


def day_stats(t: pd.DataFrame, mask: np.ndarray, level: float, prefix: str) -> dict:
    sel = t[mask]
    days = (sel["btc_y"] - COST).groupby(sel["day"]).mean()
    if len(days) < 2:
        return {f"{prefix}_days": len(days)}
    m, lo, hi = block_bootstrap_ci(days.to_numpy(), np.arange(len(days)), level=level, n_boot=2000)
    return {f"{prefix}_mean": m, f"{prefix}_lo": lo, f"{prefix}_hi": hi,
            f"{prefix}_wo_best5": float(days[days < days.quantile(0.95)].mean()),
            f"{prefix}_win_days": float((days > 0).mean()), f"{prefix}_days": int(len(days)),
            f"{prefix}_trades": int(mask.sum()), f"{prefix}_per_trade": float((sel["btc_y"] - COST).mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", type=float, default=1 - 0.05 / 4)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, frows = [], []
    for spec in a.arm:
        name, path = spec.split("=", 1)
        per_seed: dict[int, list] = {}
        for d in sorted(Path(path).glob("fold_*_seed_*")):
            m = re.match(r"fold_(\d+)_seed_(\d+)", d.name)
            val = pd.read_parquet(d / "battery_table_val.parquet")
            test = pd.read_parquet(d / "battery_table_test.parquet")
            q = float(np.quantile(val["mu"], Q_ROW))
            tv, tt = per_time(val, q), per_time(test, q)
            for col, p in (("long_share", 0.8), ("long_share", 0.7), ("long_share", 0.9), ("mean_mu", 0.8),
                           ("btc_mu", 0.8)):
                tt[f"sig_{col}_{int(p * 100)}"] = tt[col] >= float(np.quantile(tv[col], p))
            tt["fold"] = int(m.group(1))
            per_seed.setdefault(int(m.group(2)), []).append(tt)
            prim = tt["sig_long_share_80"]
            frows.append({"arm": name, "seed": int(m.group(2)), "fold": int(m.group(1)), "trades": int(prim.sum()),
                          "per_trade": float((tt.loc[prim, "btc_y"] - COST).mean()),
                          "btc_always_per_trade": float((tt["btc_y"] - COST).mean())})
        for seed, parts in per_seed.items():
            t = pd.concat(parts)
            r = {"arm": name, "seed": seed, "timestamps": len(t)}
            r.update(day_stats(t, t["sig_long_share_80"].to_numpy(), a.level, "B1"))
            r.update(day_stats(t, np.ones(len(t), bool), a.level, "btc_always"))
            r.update(day_stats(t, ~t["sig_long_share_80"].to_numpy(), a.level, "btc_rest"))
            for k in ("sig_long_share_70", "sig_long_share_90", "sig_mean_mu_80", "sig_btc_mu_80"):
                r.update(day_stats(t, t[k].to_numpy(), a.level, "info_" + k[4:]))
            # same days as B1 for condition 3
            b1_days = set(t.loc[t["sig_long_share_80"], "day"])
            al = t[t["day"].isin(b1_days)]
            r["btc_always_same_days_mean"] = float((al["btc_y"] - COST).groupby(al["day"]).mean().mean())
            r["accept"] = bool(r.get("B1_lo", -1) > 0 and r.get("B1_wo_best5", -1) > 0
                               and r.get("B1_mean", -1) > r["btc_always_same_days_mean"])
            rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(out / "b1_seeds.csv", index=False)
    pd.DataFrame(frows).to_csv(out / "b1_folds.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(res.round(5).T.to_string())
        print(pd.DataFrame(frows).round(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
