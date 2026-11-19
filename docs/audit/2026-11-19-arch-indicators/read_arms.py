"""Read the arms of the 2026-11-19 architecture / indicator study (PREREGISTERED.md in this folder) and draw charts.

    python read_arms.py --arm NAME=OUT_DIR [--arm ...] --out DIR [--stage val|test] [--level 0.9833] [--charts]

OUT_DIR is a rolling job's ``--out`` (folders ``fold_<k>_seed_<s>`` holding ``battery_table_val.parquet`` and
``battery_table_test.parquet``). The long book: threshold = the 90th percentile of mu on the fold's VAL table; buy every
row of the read stage with mu >= threshold; net = y_ret - 2 * 0.0008 (y_ret is the simple close return). The SHORT
book is read apart (owner 2026-11-19: buys and sells are never mixed in one number): the 10th percentile of mu on VAL,
sell every row with mu <= it, net = -y_ret - 2 * 0.0008; it is information only (the trading candidate is long). Writes
``arms_<stage>.csv`` (per arm x seed, pooled over folds), ``folds_<stage>.csv`` and, with ``--charts``, PNG charts.
A study script (rule G6): reads files, no training.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.stats.metrics import block_bootstrap_ci

COST = 2 * 0.0008
Q = 0.90


def _rank_ic(df: pd.DataFrame) -> float:
    vals = []
    for _, g in df.groupby("end_time"):
        if len(g) >= 5 and g["mu"].nunique() > 1 and g["y_ret"].nunique() > 1:
            vals.append(g["mu"].rank().corr(g["y_ret"].rank()))
    return float(np.nanmean(vals)) if vals else float("nan")


def fold_frames(out_dir: Path, stage: str):
    for d in sorted(out_dir.glob("fold_*_seed_*")):
        m = re.match(r"fold_(\d+)_seed_(\d+)", d.name)
        val_p, st_p = d / "battery_table_val.parquet", d / f"battery_table_{stage}.parquet"
        if not (val_p.is_file() and st_p.is_file()):
            continue
        val, st = pd.read_parquet(val_p), pd.read_parquet(st_p)
        thr = float(np.quantile(val["mu"], Q))
        st = st.copy()
        st["mkt"] = st.groupby("end_time")["y_ret"].transform("mean")
        btc = st.loc[st["coin"] == "BTCUSDT"].drop_duplicates("end_time").set_index("end_time")["y_ret"]
        st["btc"] = st["end_time"].map(btc)
        st["day"] = pd.to_datetime(st["end_time"]).dt.floor("D")
        st["long"] = st["mu"] >= thr
        st["short"] = st["mu"] <= float(np.quantile(val["mu"], 1 - Q))
        st["fold"], st["seed"] = int(m.group(1)), int(m.group(2))
        yield int(m.group(1)), int(m.group(2)), thr, st


def book_days(st: pd.DataFrame) -> pd.DataFrame:
    """Per day: the long book's mean net, its market-hedged mean net, always-long's mean net (all rows)."""
    lg = st[st["long"]]
    days = pd.DataFrame({"always_long": st.groupby("day")["y_ret"].mean() - COST})
    days["long"] = (lg["y_ret"] - COST).groupby(lg["day"]).mean()
    days["long_hedged"] = (lg["y_ret"] - lg["mkt"] - COST).groupby(lg["day"]).mean()
    sh = st[st["short"]]
    days["short"] = (-sh["y_ret"] - COST).groupby(sh["day"]).mean()
    days["short_hedged"] = (-(sh["y_ret"] - sh["mkt"]) - COST).groupby(sh["day"]).mean()
    lb = lg[lg["btc"].notna()]
    days["long_btc"] = (lb["y_ret"] - lb["btc"] - COST).groupby(lb["day"]).mean()
    days["long_btc_real"] = (lb["y_ret"] - lb["btc"] - 2 * COST).groupby(lb["day"]).mean()
    days["btc_buy_hold_same_times"] = (lb["btc"] - COST).groupby(lb["day"]).mean()
    days["trades"] = lg.groupby("day").size()
    return days.fillna({"trades": 0})


def summarise(days: pd.DataFrame, col: str, level: float) -> dict:
    v = days[col].dropna()
    if v.empty:
        return {f"{col}_mean": np.nan}
    mean, lo, hi = block_bootstrap_ci(v.to_numpy(), np.arange(len(v)), level=level, n_boot=2000)
    cut = v[v < v.quantile(0.95)]
    return {f"{col}_mean": mean, f"{col}_lo": lo, f"{col}_hi": hi, f"{col}_wo_best5": float(cut.mean()),
            f"{col}_win_days": float((v > 0).mean()), f"{col}_days": int(len(v))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="test", choices=("val", "test"))
    ap.add_argument("--level", type=float, default=1 - 0.05 / 3)
    ap.add_argument("--charts", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, frows, all_days, tables = [], [], {}, {}
    for spec in a.arm:
        name, path = spec.split("=", 1)
        per_seed = {}
        for fold, seed, thr, st in fold_frames(Path(path), a.stage):
            lg = st[st["long"]]
            frows.append({"arm": name, "fold": fold, "seed": seed, "threshold": thr, "rows": len(st),
                          "trades": len(lg), "net_per_trade": float((lg["y_ret"] - COST).mean()),
                          "hedged_per_trade": float((lg["y_ret"] - lg["mkt"] - COST).mean()),
                          "always_long_per_trade": float((st["y_ret"] - COST).mean()),
                          "short_net_per_trade": float((-st.loc[st["short"], "y_ret"] - COST).mean()),
                          "rank_ic_close": _rank_ic(st)})
            per_seed.setdefault(seed, []).append(st)
        for seed, parts in per_seed.items():
            st = pd.concat(parts, ignore_index=True)
            tables[(name, seed)] = st
            days = book_days(st)
            all_days[(name, seed)] = days
            r = {"arm": name, "seed": seed, "folds": len(parts), "trades": int(st["long"].sum()),
                 "net_per_trade": float((st.loc[st["long"], "y_ret"] - COST).mean())}
            lg = st[st["long"]]
            ok = lg["btc"].notna()
            beta = float(np.polyfit(lg.loc[ok, "btc"], lg.loc[ok, "y_ret"], 1)[0]) if ok.sum() > 10 else np.nan
            beta_net = lg.loc[ok, "y_ret"] - beta * lg.loc[ok, "btc"] - 2 * COST
            r.update({"trades_without_btc": int((~ok).sum()), "beta_to_btc": beta,
                      "long_btc_beta_per_trade": float(beta_net.mean())})
            for col in ("long", "long_hedged", "always_long", "short", "short_hedged", "long_btc", "long_btc_real",
                        "btc_buy_hold_same_times"):
                r.update(summarise(days, col, a.level))
            rows.append(r)
    res, fres = pd.DataFrame(rows), pd.DataFrame(frows)
    res.to_csv(out / f"arms_{a.stage}.csv", index=False)
    fres.to_csv(out / f"folds_{a.stage}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.5f}".format):
        print(res.to_string(index=False))
        print(fres.groupby("arm")[["net_per_trade", "hedged_per_trade", "always_long_per_trade",
                                   "short_net_per_trade", "rank_ic_close"]].mean().to_string())
    if a.charts:
        charts(all_days, tables, fres, out, a.stage)
    return 0


def charts(all_days, tables, fres, out: Path, stage: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. equity curves of the long book (cumulative sum of day net), hedged, and always-long
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for (name, seed), d in sorted(all_days.items()):
        ax.plot(d.index, d["long"].fillna(0).cumsum() * 100, label=f"{name} s{seed} long top 10 %")
    d0 = next(iter(all_days.values()))
    ax.plot(d0.index, d0["always_long"].fillna(0).cumsum() * 100, "k--", lw=1, label="always long (all rows)")
    ax.set_title(f"Long book, cumulative mean net per day (%), {stage}, cost 0.08 % per side")
    ax.set_ylabel("cumulative %")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"01_equity_{stage}.png", dpi=110)
    plt.close(fig)
    # 2. hedged equity (is it only market beta?)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for (name, seed), d in sorted(all_days.items()):
        ax.plot(d.index, d["long_hedged"].fillna(0).cumsum() * 100, label=f"{name} s{seed}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"Long book minus the equal-weight market move, cumulative (%), {stage}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"02_equity_hedged_{stage}.png", dpi=110)
    plt.close(fig)
    # 3. per fold net per trade
    piv = fres.pivot_table(index="fold", columns="arm", values="net_per_trade", aggfunc="mean") * 100
    ax = piv.plot.bar(figsize=(11, 5))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"Long top 10 % net per trade by fold (%), mean of seeds, {stage}")
    ax.figure.tight_layout()
    ax.figure.savefig(out / f"03_folds_{stage}.png", dpi=110)
    plt.close(ax.figure)
    # 4. deciles of mu -> realised mean return (calibration of the ranking)
    fig, ax = plt.subplots(figsize=(11, 5))
    for (name, seed), st in sorted(tables.items()):
        dec = pd.qcut(st["mu"].rank(method="first"), 10, labels=False)
        ax.plot(range(1, 11), st.groupby(dec)["y_ret"].mean() * 100, marker="o", label=f"{name} s{seed}")
    ax.axhline(COST * 100, color="r", lw=0.8, ls="--", label="round-trip cost")
    ax.set_xlabel("decile of predicted mu (10 = most bullish)")
    ax.set_ylabel("mean realised return %")
    ax.set_title(f"Realised close return by decile of the prediction, {stage}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"04_deciles_{stage}.png", dpi=110)
    plt.close(fig)
    # 5. a small sample: 4 liquid coins, the realised path and the long signals of the first arm / seed
    (name, seed), st = sorted(tables.items())[0]
    coins = [c for c in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT") if c in set(st["coin"])][:4]
    if coins:
        fig, axes = plt.subplots(len(coins), 1, figsize=(11, 3 * len(coins)), sharex=False)
        axes = np.atleast_1d(axes)
        for ax, c in zip(axes, coins, strict=False):
            g = st[st["coin"] == c].sort_values("end_time").tail(360)
            path = g["y_ret"].to_numpy(np.float64) * 100      # the table holds returns, not prices
            ax.plot(pd.to_datetime(g["end_time"]), path, lw=0.9, color="grey")
            sig = g["long"].to_numpy()
            ok = sig & (g["y_ret"] > COST).to_numpy()
            bad = sig & ~(g["y_ret"] > COST).to_numpy()
            t = pd.to_datetime(g["end_time"]).to_numpy()
            ax.scatter(t[ok], np.asarray(path)[ok], color="green", s=14, label="long, won after cost")
            ax.scatter(t[bad], np.asarray(path)[bad], color="red", s=14, label="long, lost")
            ax.axhline(COST * 100, color="r", lw=0.6, ls="--")
            ax.set_ylabel("realised close return %")
            ax.set_title(f"{c}: last 360 test rows, realised return to the label horizon, {name} seed {seed}")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"05_sample_coins_{stage}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
