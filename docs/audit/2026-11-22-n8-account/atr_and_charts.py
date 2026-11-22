"""Take-profit at k x ATR on the saved battery tables, and consistency charts of the model and its predictions.

Usage (Colab): . /content/nig_env.sh; python atr_and_charts.py <runs root> <out dir>
1. ATR per (coin, prediction time) from the contract artifact with `workflow.battery_run.stage_atr` (the AT1 code
   path: the window's high / low channels when present, else its last NATR_14), joined to the saved battery tables
   (they predate the `atr` column).
2. Per run x seed, every fold's TEST: the rows of the battery's take-profit grids (top 5 / 10 / 20 % by |mu| on each
   side, side from p_dir, flat rows dropped) exited at the close, at a take-profit at mu, and at a take-profit at
   k x ATR; the same take-profit on the same NUMBER of random rows of that side (one draw per fold); pooled over the
   folds. Writes `atr_tp.txt` / `atr_tp.csv`.
3. Charts per run x seed (PNG): equity curves, CAGR per fold, rolling prediction IC and timing correlation, mu
   decile calibration per fold, the take-profit comparison; one overview chart over every run of the scan.
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nig_timenet_torch.colab.paths import ARTIFACT_DRIVE  # noqa: E402
from nig_timenet_torch.data_io.asset_index import row_coins  # noqa: E402
from nig_timenet_torch.data_io.loading import LoaderSettings, load_dataset  # noqa: E402
from nig_timenet_torch.data_io.split_plan import sample_times  # noqa: E402
from nig_timenet_torch.eval.account import (  # noqa: E402
    account_curve,
    account_frame,
    book_weights,
    placebo_weights,
    row_step,
)
from nig_timenet_torch.eval.battery_lib import (  # noqa: E402
    net_pnl,
    scored_rows,
    take_profit_at_atr,
    take_profit_at_mu,
    top_abs,
    trade_side,
    trade_stats,
)
from nig_timenet_torch.workflow.battery_run import _tables, stage_atr  # noqa: E402

COST, BOOKS, PCTS, KS = 0.0008, 2, (0.05, 0.10, 0.20), (0.5, 1.0, 1.5, 2.0)
RUNS = [("rolling-dw1-r-001/out_s0", "", "single-coin GRU s0"),
        ("rolling-dw1-r-001/out_s1", "", "single-coin GRU s1"),
        ("panel-rolling-std-001/out", "seed_0", "panel std s0"),
        ("panel-rolling-std-001/out", "seed_1", "panel std s1"),
        ("panel-rolling-ls-001/out", "seed_0", "panel ls s0"), ("panel-rolling-ls-001/out", "seed_1", "panel ls s1"),
        ("snapshot-rolling-base-001/out", "seed_0", "snapshot base s0"),
        ("snapshot-rolling-base-001/out", "seed_1", "snapshot base s1")]
# reference palette (dataviz skill, light mode): categorical slots 1-3 + a neutral for benchmarks
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#d6d5d0",
                     "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "lines.linewidth": 2.0, "legend.frameon": False})


def atr_lookup(cache: Path) -> pd.DataFrame:
    if cache.is_file():
        return pd.read_parquet(cache)
    ds = load_dataset(LoaderSettings(name=ARTIFACT_DRIVE)).dataset
    n = int(np.asarray(ds["last_candles"]).shape[0])
    _, feature_end, _ = sample_times(ds)
    t = np.asarray(feature_end)
    t = pd.to_datetime(t.astype("datetime64[ns]") if np.issubdtype(t.dtype, np.datetime64) else t.astype(np.int64),
                       utc=True).tz_localize(None)
    parts = [stage_atr(ds, np.arange(a, min(a + 50_000, n))) for a in range(0, n, 50_000)]
    out = pd.DataFrame({"coin": row_coins(ds, np.arange(n)).astype(str), "end_time": t,
                        "atr": np.concatenate(parts).astype(np.float64)})
    out = out.drop_duplicates(["coin", "end_time"])
    out.to_parquet(cache, index=False)
    return out


def folds_of(root, rel, seed, lookup):
    tables = sorted(glob.glob(f"{root}/{rel}/fold_*{seed}/battery_table_test.parquet"))
    folds = []
    for f in tables:
        val, test = _tables(f.rsplit("/", 1)[0])
        test = test.merge(lookup, on=["coin", "end_time"], how="left")
        folds.append((val, test))
    return folds


def tp_rows(folds, rng):
    """Pooled pnl per (side, top share, rule) over the folds: close exit, TP at mu, TP at k ATR, and random rows."""
    pool: dict[tuple, list] = {}
    hits: dict[tuple, list] = {}
    for _, test in folds:
        test = test.reset_index(drop=True)
        side = trade_side(test)
        for pct in PCTS:
            for name, sg in (("LONG", 1.0), ("SHORT", -1.0)):
                mask = (side == sg) & scored_rows(test) & np.isfinite(test["atr"].to_numpy())
                idx = top_abs(test["mu"].values, pct, mask)
                rnd = rng.choice(np.where(mask)[0], size=len(idx), replace=False)
                pool.setdefault((name, pct, "close exit"), []).append(net_pnl(sg * test["y_ret"].values[idx], COST))
                r = take_profit_at_mu(test, idx, sg, COST)
                pool.setdefault((name, pct, "TP at mu"), []).append(r["pnl"])
                hits.setdefault((name, pct, "TP at mu"), []).append(r["hit"])
                for k in KS:
                    for who, rows in (("model", idx), ("random", rnd)):
                        r = take_profit_at_atr(test, rows, sg, k, COST)
                        key = (name, pct, f"TP {k:g} ATR" + ("" if who == "model" else " (random rows)"))
                        pool.setdefault(key, []).append(r["pnl"])
                        hits.setdefault(key, []).append(r["hit"])
    rows = []
    for key, parts in pool.items():
        s = trade_stats(np.concatenate(parts))
        if s:
            h = np.concatenate(hits[key]).mean() if key in hits else float("nan")
            rows.append({"side": key[0], "top": key[1], "rule": key[2], "n": s["n"], "fill": float(h),
                         "avg": s["avg"], "sharpe": s["sharpe"], "pf": s["pf"], "win": s["win"]})
    return pd.DataFrame(rows)


def accounts(folds, rng):
    frame = account_frame(folds)
    step = row_step(frame["end_time"])
    r = frame.pivot(index="end_time", columns="coin", values="y_ret").sort_index()
    present = r.notna()
    picks = frame.pivot(index="end_time", columns="coin", values="long").reindex_like(r).fillna(False).astype(bool)
    slots = r.shape[1]
    basket = r.mean(axis=1).to_frame("basket")
    breadth = (picks.sum(axis=1) / present.sum(axis=1)).clip(0, 1)
    hold = present.copy()
    hold.loc[:, hold.columns != "BTCUSDT"] = False
    curves = {"model book": account_curve(book_weights(picks, slots), r, COST, BOOKS, step),
              "timing (basket x breadth)": account_curve(breadth.to_frame("basket"), basket, COST, BOOKS, step),
              "BTC buy and hold": account_curve(book_weights(hold, 1), r, COST, BOOKS, step),
              "always long": account_curve(book_weights(present, slots), r, COST, BOOKS, step)}
    plc = pd.concat([account_curve(book_weights(placebo_weights(picks, present, rng), slots), r, COST, BOOKS, step)
                     for _ in range(20)], axis=1)
    return frame, curves, plc, breadth, basket["basket"]


def cagr(e):
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    return (e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan


def chart_equity(curves, plc, title, path):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    lo, hi = plc.quantile(0.05, axis=1), plc.quantile(0.95, axis=1)
    ax.fill_between(plc.index, lo, hi, color=GRAY, alpha=0.18, lw=0, label="random coins, same timing (5-95 %)")
    for (name, eq), c in zip(curves.items(), (BLUE, ORANGE, AQUA, GRAY), strict=True):
        ax.plot(eq.index, eq, color=c, lw=1.4 if name == "always long" else 2.0,
                ls="--" if name == "always long" else "-", label=f"{name}  (CAGR {cagr(eq) * 100:+.1f} %)")
        ax.annotate(f"x{eq.iloc[-1]:.2f}", (eq.index[-1], eq.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    color=MUTED, fontsize=9, va="center")
    ax.set_yscale("log")
    ax.set_ylabel("account value (start = 1, log scale)")
    ax.set_title(f"{title}: account after 0.08 % per side")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def chart_folds(curves, frame, title, path):
    names = ("model book", "timing (basket x breadth)", "BTC buy and hold")
    rows = []
    for k, g in frame.groupby("fold"):
        lo, hi = g["end_time"].min(), g["end_time"].max()
        rows.append([k, f"{lo:%Y-%m}"] + [cagr(curves[n].loc[lo:hi]) * 100 for n in names])
    df = pd.DataFrame(rows, columns=["fold", "start", *names])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x, w = np.arange(len(df)), 0.27
    for i, (n, c) in enumerate(zip(names, (BLUE, ORANGE, AQUA), strict=True)):
        ax.bar(x + (i - 1) * w, df[n], width=w - 0.03, color=c, label=n)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, [f"fold {k}\n{s}" for k, s in zip(df["fold"], df["start"], strict=True)])
    ax.set_ylabel("annualised return of the fold (%)")
    ax.set_title(f"{title}: every 6-month TEST fold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return df


def chart_ic(frame, breadth, basket, title, path):
    f = frame[["end_time", "mu", "y_ret"]].copy()
    f["rm"] = f.groupby("end_time")["mu"].rank()
    f["ry"] = f.groupby("end_time")["y_ret"].rank()
    ic = f.groupby("end_time")[["rm", "ry"]].corr().xs("rm", level=1)["ry"]
    daily_ic = ic.resample("D").mean().rolling(90, min_periods=30).mean()
    tm = pd.concat([breadth.rename("b"), basket.rename("m")], axis=1)
    timing = tm["b"].rolling(540, min_periods=180).corr(tm["m"]).resample("D").last()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(daily_ic.index, daily_ic, color=BLUE, label="coin ranking: IC of mu vs the next move across coins (90 d)")
    ax.plot(timing.index, timing, color=ORANGE, label="timing: corr(breadth, next basket move) (90 d)")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("correlation")
    ax.set_title(f"{title}: are the predictions consistently right?")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return float(ic.mean()), float((daily_ic.dropna() > 0).mean())


def chart_calibration(frame, title, path):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    pooled = []
    for _, g in frame.groupby("fold"):
        q = pd.qcut(g["mu"].rank(method="first"), 10, labels=False)
        m = g.groupby(q).agg(mu=("mu", "mean"), y=("y_ret", "mean"))
        ax.scatter(m["mu"] * 100, m["y"] * 100, s=14, color=BLUE, alpha=0.30, lw=0)
        pooled.append(m)
    p = pd.concat(pooled).groupby(level=0).mean()
    ax.plot(p["mu"] * 100, p["y"] * 100, color=BLUE, marker="o", ms=5, label="mean over folds (dots: each fold)")
    lim = [min(p["mu"].min(), p["y"].min()) * 100, max(p["mu"].max(), p["y"].max()) * 100]
    ax.plot(lim, lim, color=GRAY, lw=1, ls="--", label="predicted = realised")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("predicted move mu, decile mean (%)")
    ax.set_ylabel("realised next move, decile mean (%)")
    ax.set_title(f"{title}: deciles of mu")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def chart_tp(tp, title, path):
    sub = tp[(tp["side"] == "LONG") & (tp["top"] == 0.10)].set_index("rule")
    rules = ["close exit", "TP at mu"] + [f"TP {k:g} ATR" for k in KS]
    model = [sub.loc[r, "avg"] * 100 if r in sub.index else np.nan for r in rules]
    rnd = [sub.loc[f"{r} (random rows)", "avg"] * 100 if f"{r} (random rows)" in sub.index else np.nan for r in rules]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x, w = np.arange(len(rules)), 0.38
    ax.bar(x - w / 2, model, width=w - 0.03, color=BLUE, label="the model's top 10 % LONG rows")
    ax.bar(x + w / 2, rnd, width=w - 0.03, color=GRAY, label="the same number of random LONG rows")
    for xi, v in zip(x, model, strict=True):
        if np.isfinite(v):
            ax.annotate(f"{v:+.3f}", (xi - w / 2, v), xytext=(0, 3 if v >= 0 else -11), textcoords="offset points",
                        ha="center", fontsize=8, color=MUTED)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, rules)
    ax.set_ylabel("mean net P&L per trade (%)")
    ax.set_title(f"{title}: exit rules, after 0.16 % per round trip")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def chart_overview(scan_json: Path, path):
    d = json.loads(scan_json.read_text())
    rows = []
    for k, v in d.items():
        s = v.get("stats", {})
        if "model" in s and s["setup"]["folds"] == 7 and "smoke" not in k:
            rows.append((k.replace("/out", "").replace(" seed ", " s"), s["model"]["cagr"] * 100,
                         s["placebo"]["cagr_p95"] * 100, s["breadth timing"]["cagr"] * 100))
    df = pd.DataFrame(rows, columns=["run", "model", "placebo p95", "timing"]).sort_values("model")
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(df) + 1.5))
    y = np.arange(len(df))
    ax.scatter(df["placebo p95"], y, color=GRAY, s=30, marker="|", lw=2, label="random coins, same timing: 95th pct")
    ax.scatter(df["timing"], y, color=ORANGE, s=30, label="timing (basket x breadth)")
    ax.scatter(df["model"], y, color=BLUE, s=36, label="model book")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvline(45.5, color=AQUA, lw=1.2, ls="--", label="BTC buy and hold +45.5 %")
    ax.set_yticks(y, df["run"], fontsize=8)
    ax.set_xlabel("CAGR after 0.08 % per side, TEST 2023-01 .. 2026-06 (%)")
    ax.set_title("Every 7-fold run on Drive as a capital account")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(root: str, out: str) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lookup = atr_lookup(out_dir / "atr_lookup.parquet")
    print(f"ATR lookup: {len(lookup):,} rows, {lookup['atr'].notna().mean():.1%} with an ATR", flush=True)
    rng = np.random.default_rng(0)
    all_tp, summary = [], []
    for rel, seed, title in RUNS:
        folds = folds_of(root, rel, seed, lookup)
        matched = np.mean([t["atr"].notna().mean() for _, t in folds])
        tp = tp_rows(folds, rng).assign(run=title)
        all_tp.append(tp)
        d = out_dir / "charts" / title.replace(" ", "_")
        d.mkdir(parents=True, exist_ok=True)
        frame, curves, plc, breadth, basket = accounts(folds, rng)
        chart_equity(curves, plc, title, d / "1_equity.png")
        fdf = chart_folds(curves, frame, title, d / "2_folds.png")
        ic_mean, ic_pos = chart_ic(frame, breadth, basket, title, d / "3_consistency.png")
        chart_calibration(frame, title, d / "4_calibration.png")
        chart_tp(tp, title, d / "5_take_profit.png")
        summary.append({"run": title, "atr_matched": matched, "ic_mean": ic_mean, "ic90_positive_share": ic_pos,
                        "folds_model_positive": int((fdf["model book"] > 0).sum()), "folds": len(fdf)})
        print(f"\n== {title}: ATR matched on {matched:.1%} of TEST rows; mean cross-coin IC {ic_mean:+.4f}, "
              f"90-day IC > 0 on {ic_pos:.0%} of days; model folds > 0: {summary[-1]['folds_model_positive']}/"
              f"{len(fdf)}", flush=True)
        show = tp[tp["top"] == 0.10].sort_values(["side", "rule"])
        for _, r in show.iterrows():
            print(f"  {r['side']:<5} top10% {r['rule']:<30} n {r['n']:>7,} fill {r['fill'] * 100:5.1f}% "
                  f"avg {r['avg'] * 100:+.4f}% Sharpe {r['sharpe']:+.3f} PF {r['pf']:.2f}", flush=True)
    pd.concat(all_tp).to_csv(out_dir / "atr_tp.csv", index=False)
    pd.DataFrame(summary).to_csv(out_dir / "consistency_summary.csv", index=False)
    scan = out_dir / "n8_scan.json"
    if scan.is_file():
        chart_overview(scan, out_dir / "charts" / "0_overview_all_runs.png")
    print("\ncharts in", out_dir / "charts", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
