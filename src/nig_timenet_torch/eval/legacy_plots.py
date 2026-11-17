"""PURPOSE: The figures of the TF evaluation (``evaluation/plots.py``) over a battery table: equity and drawdown of the
    trades and of the daily portfolio, calibration curve, sd against error, error distribution, actual against
    predicted, lag scan, batch hit rate, and the best / worst coins and samples.
TAGS: legacy plots, equity chart, portfolio chart, drawdown, calibration curve, best worst predictions, lag plot,
    matplotlib, figures
PITFALLS: matplotlib is an optional dependency (extra ``plots``): it is imported inside ``figures`` only, with the Agg
    backend (no display). Every figure reads the numbers of ``eval/legacy_reports.py``; nothing is computed twice.
    The scatter and the coin series are subsampled for size, the numbers in their titles are on all rows.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.eval import legacy_reports as lr
from nig_timenet_torch.stats import metrics

__all__ = ["figures", "MAX_POINTS"]

MAX_POINTS = 20_000


def _equity(plt, trade_rows, daily, title):
    eq = lr.equity_curve(trade_rows["net"].to_numpy())
    fig, ax = plt.subplots(4, 1, figsize=(12, 11), gridspec_kw={"height_ratios": [3, 1, 3, 1]}, sharex=False)
    ax[0].plot(np.arange(len(eq["equity"])), eq["equity"], color="#264653", lw=1.2)
    ax[0].set_title(f"{title}: equity of the trades in time order ({lr.RISK_PER_TRADE:.0%} of capital per trade, "
                    f"{len(trade_rows):,} trades, net of cost)", fontsize=10)
    ax[1].fill_between(np.arange(len(eq["drawdown"])), 100 * eq["drawdown"], color="#e76f51", alpha=0.6)
    ax[1].set_ylabel("drawdown %")
    x = daily.index.to_pydatetime()
    ax[2].plot(x, daily["equity"], color="#2a9d8f", lw=1.4)
    ax[2].axhline(1.0, color="#6c757d", lw=0.8, ls="--")
    pos = float((daily["ret"] > 0).mean()) if len(daily) else float("nan")
    ax[2].set_title(f"daily portfolio (whole capital split over each day's trades): {len(daily)} days, "
                    f"{pos:.0%} positive, final {daily['equity'].iloc[-1] if len(daily) else float('nan'):.3f}",
                    fontsize=10)
    ax[3].fill_between(x, 100 * daily["drawdown"], color="#e76f51", alpha=0.6)
    ax[3].set_ylabel("drawdown %")
    for a in ax:
        a.grid(alpha=0.25)
    return fig


def _calibration(plt, cal, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    b = pd.DataFrame(cal["bins"]).dropna()
    ax.plot([0, 1], [0, 1], color="#6c757d", ls="--", lw=0.8)
    ax.plot(b["mean_p"], b["mean_up"], marker="o", color="#264653")
    for _, r in b.iterrows():
        ax.annotate(f"{int(r['count']):,}", (r["mean_p"], r["mean_up"]), fontsize=7, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlabel("mean predicted p(up)")
    ax.set_ylabel("share up")
    ax.set_title(f"{title}: calibration of p_dir  ECE {cal['ece']:.3f}  Brier {cal['brier']:.4f} "
                 f"(base rate {cal['brier_base_rate']:.4f})", fontsize=9)
    ax.grid(alpha=0.25)
    return fig


def _errors(plt, table, title):
    err = (table["mu"] - table["y_ret"]).to_numpy()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    lim = np.nanpercentile(np.abs(err), 99)
    ax[0].hist(np.clip(err, -lim, lim), bins=80, color="#264653")
    ax[0].set_title(f"error mu - y (clipped at p99 {lim:.3f}); mean {np.nanmean(err):+.4f}", fontsize=9)
    dec = pd.qcut(table["sigma"], 10, labels=False, duplicates="drop")
    m = pd.DataFrame({"d": dec, "e": np.abs(err)}).groupby("d")["e"].mean()
    ax[1].bar(m.index, m.to_numpy(), color="#2a9d8f")
    ax[1].set_title(f"mean |error| by decile of the predicted sd (rank IC sd vs |err| "
                    f"{metrics.spearman(table['sigma'], np.abs(err)):+.3f})", fontsize=9)
    sub = table.sample(min(len(table), MAX_POINTS), random_state=0)
    ax[2].scatter(sub["mu"], sub["y_ret"], s=2, alpha=0.3, color="#264653")
    ax[2].set_xlabel("predicted mu")
    ax[2].set_ylabel("realised return")
    ax[2].set_title(f"actual vs predicted: Pearson {metrics.pearson(table['mu'], table['y_ret']):+.3f}, "
                    f"rank IC {metrics.spearman(table['mu'], table['y_ret']):+.3f}", fontsize=9)
    for a in ax:
        a.grid(alpha=0.25)
    fig.suptitle(title, fontsize=10)
    return fig


def _lag_and_batches(plt, integ, cond, title):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    scan = pd.DataFrame(integ["lag_scan"])
    ax[0].bar(scan["lag"], scan["mean_r"], color=np.where(scan["lag"] == 0, "#2a9d8f", "#adb5bd"))
    ax[0].set_title("lag scan: mean r of mu(t) with y(t + k) per coin (k < 0 peak = late copying)", fontsize=9)
    b = cond["by_batch"]
    ax[1].errorbar(b["group"], b["hit"], yerr=[b["hit"] - b["ci_lo"], b["ci_hi"] - b["hit"]], fmt="o",
                   color="#264653")
    ax[1].axhline(0.5, color="#6c757d", ls="--", lw=0.8)
    ax[1].set_title("direction hit by consecutive time batch (Wilson 95 %)", fontsize=9)
    for a in ax:
        a.grid(alpha=0.25)
    fig.suptitle(title, fontsize=10)
    return fig


def _coins(plt, table, ranking, title, n_best=2, n_worst=2, last=300):
    names = list(ranking.head(n_best)["coin"]) + list(ranking.tail(n_worst)["coin"])[::-1]
    fig, ax = plt.subplots(len(names), 1, figsize=(13, 2.8 * max(len(names), 1)), squeeze=False)
    for a, coin in zip(ax[:, 0], names, strict=True):
        g = table[table["coin"] == coin].sort_values("end_time").tail(last)
        a.plot(g["end_time"], g["y_ret"], color="#adb5bd", lw=0.9, label="realised")
        a.plot(g["end_time"], g["mu"], color="#e76f51", lw=1.2, label="predicted mu")
        r = ranking.set_index("coin").loc[coin]
        kind = "BEST" if coin in names[:n_best] else "WORST"
        a.set_title(f"{kind} {coin}: r {r['r']:+.3f}  rank IC {r['rank_ic']:+.3f}  hit {r['hit']:.2f}  rows "
                    f"{int(r['rows'])} (last {len(g)} shown)", fontsize=9)
        a.legend(fontsize=7, loc="upper left")
        a.grid(alpha=0.25)
    fig.suptitle(f"{title}: best and worst coins, actual vs predicted", fontsize=10)
    return fig


def _samples(plt, ext, title):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for a, key, color in ((ax[0], "worst", "#e76f51"), (ax[1], "best", "#2a9d8f")):
        d = ext[key]
        labels = [f"{c} {pd.Timestamp(t):%m-%d %H:%M} {'L' if s > 0 else 'S'}"
                  for c, t, s in zip(d["coin"], d["end_time"], d["side"], strict=True)]
        a.barh(labels[::-1], 100 * d["net"].to_numpy()[::-1], color=color)
        a.set_title(f"{key} {len(d)} trades (net %, L long / S short)", fontsize=9)
        a.tick_params(axis="y", labelsize=7)
        a.grid(alpha=0.25)
    fig.suptitle(title, fontsize=10)
    return fig


def figures(stage: str, table: pd.DataFrame, rep: dict, out_dir) -> list[Path]:
    """Write every figure of one stage (``rep`` = the stage part of ``workflow.legacy_report``) as PNG into
    ``out_dir``; returns the paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    title = stage.upper()
    made = {"equity": _equity(plt, rep["trade_rows"], rep["daily"], title),
            "calibration": _calibration(plt, rep["calibration"], title),
            "errors": _errors(plt, table, title),
            "lag_batches": _lag_and_batches(plt, rep["integrity"], rep["conditioned"], title),
            "best_worst_coins": _coins(plt, table, rep["coins"], title),
            "extreme_samples": _samples(plt, rep["extremes"], title)}
    paths = []
    for name, fig in made.items():
        fig.tight_layout()
        path = out_dir / f"{stage}_{name}.png"
        fig.savefig(path, dpi=90)
        plt.close(fig)
        paths.append(path)
    return paths
