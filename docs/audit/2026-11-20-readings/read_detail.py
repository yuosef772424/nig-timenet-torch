"""The failure reading of rule C7 from the saved tables of rolling arms (2026-11-20; arm I, BR1, M3).

    python read_detail.py --arm NAME=OUT_DIR [--arm ...] --out DIR [--ret-col y_ret]

OUT_DIR is a rolling job's ``--out`` (``fold_<k>_seed_<s>/battery_table_{val,test}.parquet`` and ``fold_row.json``).
The long book is the one of ``2026-11-19-arch-indicators/read_arms.py`` (VAL 90th percentile of mu, net = y_ret -
0.0016). Writes CSVs to DIR and prints them:

- ``training.csv``: best epoch, epochs done, the report's TEST close rank IC per fold x seed (learned, underfit,
  memorised?);
- ``error_map.csv``: the long book per group (decile of sigma and of mu inside the book, coin volatility tercile,
  market regime = sign of the equal-weight move of the timestamp, year): trades, days, hit rate after cost, gross,
  net, net minus the market, and the trivial baseline on the SAME group = always long on all rows of the group
  (empty for the groups inside the book, whose baseline is the market column);
- ``worst.csv``: the 20 long trades with the largest loss among the top 2 % of mu, and the 10 worst days;
- ``bias.csv``: per fold, share of rows with mu > 0 against share with y_ret > 0, mean mu against mean y_ret;
- ``selection.csv``: four rules fitted on VAL only (top 5 % mu; top 10 % mu with sigma under its VAL median; top 10 %
  of mu / sigma; top 10 % mu with p_dir > 0.5), applied to TEST: share of rows kept, net and market-hedged net per
  trade (4 rules tried: multiple testing);
- ``overlap.csv``: with two arms, the share of the first arm's long trades that the second arm also takes.

y_ret is whatever the table holds (the own simple return, or the BTC-relative / residual return for BR1 / M3).
A study script (rule G6): reads files, no training.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COST = 2 * 0.0008
Q = 0.90


def load(out_dir: Path):
    for d in sorted(out_dir.glob("fold_*_seed_*")):
        m = re.match(r"fold_(\d+)_seed_(\d+)", d.name)
        vp, tp = d / "battery_table_val.parquet", d / "battery_table_test.parquet"
        if not (vp.is_file() and tp.is_file()):
            continue
        fr = d / "fold_row.json"
        meta = json.loads(fr.read_text()) if fr.is_file() else {}
        yield int(m.group(1)), int(m.group(2)), pd.read_parquet(vp), pd.read_parquet(tp), meta


def prepare(val: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    st = st.copy()
    st["mkt"] = st.groupby("end_time")["y_ret"].transform("mean")
    st["day"] = pd.to_datetime(st["end_time"]).dt.floor("D")
    st["year"] = pd.to_datetime(st["end_time"]).dt.year
    st["long"] = st["mu"] >= float(np.quantile(val["mu"], Q))
    st["net"] = st["y_ret"] - COST
    return st


def group_rows(st: pd.DataFrame, key: pd.Series, name: str, baseline: bool = True) -> list[dict]:
    rows = []
    for g, part in st.groupby(key, observed=True):
        lg = part[part["long"]]
        rows.append({"group_by": name, "group": str(g), "rows": len(part), "trades": len(lg),
                     "days": int(lg["day"].nunique()), "hit_after_cost": float((lg["net"] > 0).mean()),
                     "gross": float(lg["y_ret"].mean()), "net": float(lg["net"].mean()),
                     "net_minus_mkt": float((lg["net"] - lg["mkt"]).mean()),
                     "always_long_same_rows": float(part["net"].mean()) if baseline else np.nan})
    return rows


def error_map(st: pd.DataFrame) -> list[dict]:
    lg_mask = st["long"]
    out = []
    sig_dec = pd.Series(np.nan, index=st.index)
    sig_dec[lg_mask] = pd.qcut(st.loc[lg_mask, "sigma"].rank(method="first"), 5, labels=False) + 1
    out += group_rows(st[lg_mask], sig_dec[lg_mask], "sigma quintile in book (5 = widest)", False)
    mu_dec = pd.qcut(st.loc[lg_mask, "mu"].rank(method="first"), 5, labels=False) + 1
    out += group_rows(st[lg_mask], mu_dec, "mu quintile in book (5 = most bullish)", False)
    vol = st.groupby("coin")["y_ret"].std()
    terc = pd.qcut(vol.rank(method="first"), 3, labels=["calm", "mid", "wild"])
    out += group_rows(st, st["coin"].map(terc), "coin volatility tercile")
    out += group_rows(st, np.where(st["mkt"] > 0, "market up", "market down"), "regime (realised, same timestamp)")
    out += group_rows(st, st["year"], "year")
    return out


def selection(val: pd.DataFrame, st: pd.DataFrame) -> list[dict]:
    rules = {
        "top 5 % mu": lambda t: t["mu"] >= np.quantile(val["mu"], 0.95),
        "top 10 % mu, sigma < VAL median": lambda t: (t["mu"] >= np.quantile(val["mu"], Q))
        & (t["sigma"] < np.median(val["sigma"])),
        "top 10 % mu / sigma": lambda t: (t["mu"] / t["sigma"]) >= np.quantile(val["mu"] / val["sigma"], Q),
        "top 10 % mu, p_dir > 0.5": lambda t: (t["mu"] >= np.quantile(val["mu"], Q)) & (t["p_dir"] > 0.5),
    }
    out = []
    for name, rule in rules.items():
        k = rule(st)
        out.append({"rule": name, "kept_share": float(k.mean()), "trades": int(k.sum()),
                    "days": int(st.loc[k, "day"].nunique()), "net": float(st.loc[k, "net"].mean()),
                    "net_minus_mkt": float((st.loc[k, "net"] - st.loc[k, "mkt"]).mean())})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train, emap, worst, bias, sel, books = [], [], [], [], [], {}
    for spec in a.arm:
        name, path = spec.split("=", 1)
        per_seed = {}
        for fold, seed, val, st, meta in load(Path(path)):
            st = prepare(val, st)
            train.append({"arm": name, "fold": fold, "seed": seed,
                          "best_epoch": meta.get("best_epoch"), "epochs_done": meta.get("epochs_done"),
                          "test_close_rank_ic": meta.get("metrics", {}).get(
                              "targets.close.skill.rank_ic_by_timestamp")})
            bias.append({"arm": name, "fold": fold, "seed": seed, "share_mu_up": float((st["mu"] > 0).mean()),
                         "share_y_up": float((st["y_ret"] > 0).mean()), "mean_mu": float(st["mu"].mean()),
                         "mean_y": float(st["y_ret"].mean())})
            for r in selection(val, st):
                sel.append({"arm": name, "fold": fold, "seed": seed, **r})
            per_seed.setdefault(seed, []).append(st)
        for seed, parts in per_seed.items():
            st = pd.concat(parts, ignore_index=True)
            books[(name, seed)] = st
            for r in error_map(st):
                emap.append({"arm": name, "seed": seed, **r})
            top = st[st["mu"] >= st["mu"].quantile(0.98)].nsmallest(20, "y_ret")
            for _, r in top.iterrows():
                worst.append({"arm": name, "seed": seed, "kind": "confident loss", "coin": r["coin"],
                              "end_time": r["end_time"], "mu": r["mu"], "sigma": r["sigma"], "y_ret": r["y_ret"],
                              "mkt": r["mkt"]})
            lg = st[st["long"]]
            days = lg.groupby("day").agg(net=("net", "mean"), mkt=("mkt", "mean"), trades=("net", "size"))
            for d, r in days.nsmallest(10, "net").iterrows():
                worst.append({"arm": name, "seed": seed, "kind": "worst day", "end_time": d, "y_ret": r["net"],
                              "mkt": r["mkt"], "trades": r["trades"]})
    tables = {"training": pd.DataFrame(train), "error_map": pd.DataFrame(emap), "worst": pd.DataFrame(worst),
              "bias": pd.DataFrame(bias), "selection": pd.DataFrame(sel)}
    names = list(dict.fromkeys(n for n, _ in books))
    if len(names) >= 2:
        ov = []
        for seed in sorted({s for _, s in books}):
            a0, a1 = books.get((names[0], seed)), books.get((names[1], seed))
            if a0 is None or a1 is None:
                continue
            k0 = set(map(tuple, a0.loc[a0["long"], ["coin", "end_time"]].astype(str).to_numpy()))
            k1 = set(map(tuple, a1.loc[a1["long"], ["coin", "end_time"]].astype(str).to_numpy()))
            ov.append({"seed": seed, "first": names[0], "second": names[1], "first_trades": len(k0),
                       "share_also_in_second": len(k0 & k1) / max(len(k0), 1)})
        tables["overlap"] = pd.DataFrame(ov)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 500,
                           "display.float_format", "{:.5f}".format):
        for key, df in tables.items():
            df.to_csv(out / f"{key}.csv", index=False)
            print(f"\n## {key}")
            if key == "selection" and not df.empty:
                print(df.groupby(["arm", "rule"])[["kept_share", "net", "net_minus_mkt"]].mean().to_string())
            else:
                print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
