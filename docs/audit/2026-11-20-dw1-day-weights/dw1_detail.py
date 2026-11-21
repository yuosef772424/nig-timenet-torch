"""DW1 failure-reading detail (rule C7): which TRAINING days DW weighted up, and the TEST error map by market day.

    python dw1_detail.py --runs RUNS_DIR --artifact ART_DIR --out DIR

RUNS_DIR holds ``rolling-dw1-{r,dw,bal}-001``. Training side: every DW fold's ``day_weights.csv`` (under
``train_s<s>/fold_*/ckpt``) joined to the artifact's close label per UTC day (market move = mean label of the day,
dispersion = mean |label|). TEST side: the battery tables of the three arms, the TEST days grouped by the day's
equal-weight market return and by its cross-coin dispersion (terciles), per group: days, mean day NLL (Gaussian with
the predictive mu / sigma), mean day rank IC, long-book mean net (VAL 90th percentile), share of rows with mu > 0
against the share of y_ret > 0; and the 10 worst long-book days of DW next to R on the same days.
A study script (rule G6): no training.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import TS_COL

COST = 2 * 0.0008
ARMS = {"R": "rolling-dw1-r-001", "DW": "rolling-dw1-dw-001", "BAL": "rolling-dw1-bal-001"}


def label_days(art: Path) -> pd.DataFrame:
    ts = np.load(art / "last_candles.npy", mmap_mode="r")[:, TS_COL].astype(np.int64)
    y = np.load(art / "y_close_reg.npy", mmap_mode="r").reshape(len(ts), -1)[:, 0].astype(np.float64)
    df = pd.DataFrame({"day": ts // 86_400_000_000_000, "y": y})
    return df.groupby("day").agg(mkt=("y", "mean"), disp=("y", lambda v: float(np.abs(v).mean())), coins=("y", "size"))


def training_days(runs: Path, art: Path) -> dict:
    lab = label_days(art)
    tabs = []
    for f in sorted((runs / ARMS["DW"]).glob("train_s*/fold_*/ckpt/day_weights.csv")):
        t = pd.read_csv(f)
        t["fold"], t["seed"] = f.parent.parent.name, f.parent.parent.parent.name
        tabs.append(t)
    t = pd.concat(tabs, ignore_index=True).join(lab, on="day")
    t["abs_mkt"] = t["mkt"].abs()
    per = t.groupby(["seed", "fold"])
    out = {"folds": int(per.ngroups),
           "corr_weight_vs": {c: float(per.apply(lambda g, c=c: g["weight_rel"].corr(g[c], method="spearman")).mean())
                              for c in ("loss_ema", "abs_mkt", "mkt", "disp", "rows")},
           "top10_share_of_weight": float(per.apply(
               lambda g: g["weight_rel"].nlargest(max(1, len(g) // 10)).sum() / g["weight_rel"].sum()).mean()),
           "max_weight_rel": float(t["weight_rel"].max())}
    last = t[t["fold"] == t["fold"].max()]
    top = last.sort_values("weight_rel", ascending=False).drop_duplicates("date").head(12)
    cols = ["date", "rows", "weight_rel", "loss_ema", "mkt", "disp"]
    out["top_days_last_fold"] = top[cols].round(5).to_dict("records")
    q = t.groupby(pd.qcut(t["disp"].rank(method="first"), 5, labels=False))["weight_rel"].mean()
    out["mean_weight_by_dispersion_quintile"] = [round(float(v), 3) for v in q]
    q = t.groupby(pd.qcut(t["mkt"].rank(method="first"), 5, labels=False))["weight_rel"].mean()
    out["mean_weight_by_market_quintile"] = [round(float(v), 3) for v in q]
    return out


def test_rows(runs: Path, arm: str) -> pd.DataFrame:
    parts = []
    for d in sorted((runs / ARMS[arm]).glob("out_s*/fold_*_seed_*")):
        val, st = pd.read_parquet(d / "battery_table_val.parquet"), pd.read_parquet(d / "battery_table_test.parquet")
        st = st[["coin", "end_time", "mu", "sigma", "y_ret"]].copy()
        st["long"] = st["mu"] >= float(np.quantile(val["mu"], 0.9))
        st["seed"] = int(d.name.rsplit("_", 1)[1])
        parts.append(st)
    st = pd.concat(parts, ignore_index=True)
    st["day"] = pd.to_datetime(st["end_time"]).dt.floor("D")
    sig = st["sigma"].clip(lower=1e-6)
    st["nll"] = 0.5 * np.log(2 * np.pi * sig ** 2) + 0.5 * ((st["y_ret"] - st["mu"]) / sig) ** 2
    return st


def day_frame(st: pd.DataFrame) -> pd.DataFrame:
    g = st.groupby(["seed", "day"])
    ic = st.groupby(["seed", "end_time"]).apply(
        lambda x: x["mu"].rank().corr(x["y_ret"].rank()) if len(x) >= 5 else np.nan)
    ic = ic.rename("ic").reset_index()
    ic["day"] = pd.to_datetime(ic["end_time"]).dt.floor("D")
    lg = st[st["long"]]
    return pd.DataFrame({"nll": g["nll"].mean(), "mkt": g["y_ret"].mean(), "disp": g["y_ret"].apply(
        lambda v: float((v - v.mean()).abs().mean())), "up_pred": g["mu"].apply(lambda v: float((v > 0).mean())),
        "up_real": g["y_ret"].apply(lambda v: float((v > 0).mean())),
        "ic": ic.groupby(["seed", "day"])["ic"].mean(),
        "long": (lg["y_ret"] - COST).groupby([lg["seed"], lg["day"]]).mean(),
        "sigma": g["sigma"].mean()})


def error_map(days: dict) -> dict:
    ref = days["R"]
    groups = {"market": pd.qcut(ref["mkt"].rank(method="first"), 3, labels=["down", "flat", "up"]),
              "dispersion": pd.qcut(ref["disp"].rank(method="first"), 3, labels=["calm", "mid", "wild"])}
    out = {}
    for gname, lab in groups.items():
        rows = []
        for arm, d in days.items():
            d = d.join(lab.rename("grp"))
            for grp, x in d.groupby("grp", observed=True):
                rows.append({"arm": arm, "group": str(grp), "days": len(x), "nll": x["nll"].mean(),
                             "ic": x["ic"].mean(), "long": x["long"].mean(), "sigma": x["sigma"].mean(),
                             "up_pred": x["up_pred"].mean(), "up_real": x["up_real"].mean()})
        out[gname] = pd.DataFrame(rows).round(5).to_dict("records")
    worst = days["DW"]["long"].nsmallest(10)
    out["worst_dw_long_days"] = [{"seed": int(s), "day": str(dd.date()), "dw_long": round(float(v), 5),
                                  "r_long": round(float(ref["long"].get((s, dd), np.nan)), 5),
                                  "mkt": round(float(ref["mkt"].get((s, dd), np.nan)), 5)}
                                 for (s, dd), v in worst.items()]
    out["bias"] = {arm: {"up_pred": float(d["up_pred"].mean()), "up_real": float(d["up_real"].mean()),
                         "sigma": float(d["sigma"].mean())} for arm, d in days.items()}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"training_days": training_days(runs, Path(a.artifact))}
    days = {arm: day_frame(test_rows(runs, arm)) for arm in ARMS}
    res["test"] = error_map(days)
    (out / "dw1_detail.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
