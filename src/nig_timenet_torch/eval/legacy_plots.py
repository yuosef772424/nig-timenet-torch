"""PURPOSE: The TF evaluation figures over a battery table (equity, calibration, errors, lag, best / worst coins) and
    per chosen coin predicted vs realised close / high / low with a sigma band and a scatter (``pred_vs_true``).
TAGS: legacy plots, equity chart, portfolio chart, drawdown, calibration curve, best worst predictions, lag plot,
    matplotlib, figures, predicted vs realised, per coin plot, sigma band, top coins
PITFALLS: matplotlib is an optional dependency (extra ``plots``): it is imported inside ``figures``,
    ``pred_vs_true`` and ``price_panels`` only, with the Agg backend (no display). ``pred_vs_true`` needs the battery
    table's ``coin``, ``end_time``, ``mu``, ``sigma``, ``y_ret`` columns (``pred_high`` / ``fut_high`` /
    ``pred_low`` / ``fut_low`` optional) and raises on a coin the table does not hold; its rank IC
    (``stats.metrics.rank_ic``) is over the coin's own rows (one series), not across coins per timestamp.
    ``price_panels`` (TF price-level figures: actual / predicted / naive t-1 over the last rows + a per-coin lag scan)
    needs the table's ``last_close`` column (battery tables written before it was added raise: re-run the battery
    step); price = last_close * (1 + return); the naive line is the previous row's actual price of the same coin,
    so on price level the predicted line always hugs the actual one: judge by "MAE improve", not by the picture.
    Every figure reads the numbers of ``eval/legacy_reports.py``; nothing is computed twice.
    The scatter and the coin series are subsampled for size, the numbers in their titles are on all rows.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import TARGET_COLUMNS
from nig_timenet_torch.eval import legacy_reports as lr
from nig_timenet_torch.stats import metrics

__all__ = ["figures", "pred_vs_true", "price_panels", "improve_vs_naive", "top_coins", "MAX_POINTS"]

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
    hit = b["hit_mu_sign"]
    ax[1].errorbar(b["group"], hit, yerr=[hit - b["ci_lo"], b["ci_hi"] - hit], fmt="o",
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
        a.set_title(f"{kind} {coin}: r {r['r']:+.3f}  rank IC {r['rank_ic']:+.3f}  hit {r['hit_mu_sign']:.2f}  rows "
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


def top_coins(table: pd.DataFrame, n: int = 3) -> list[str]:
    """The ``n`` coins with the most rows in a battery table (ties: first seen)."""
    if "coin" not in table.columns:
        raise ValueError("top_coins: the table has no 'coin' column")
    return [str(c) for c in table["coin"].astype(str).value_counts().head(n).index]


def _series(ax, t, pred, real, label, band=None):
    ax.plot(t, real, color="#264653", lw=0.9, label=f"realised {label}")
    ax.plot(t, pred, color="#e76f51", lw=0.9, label=f"predicted {label}")
    if band is not None:
        ax.fill_between(t, pred - band, pred + band, color="#e76f51", alpha=0.18, label="+-1 sigma")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper left")


def _pred_vs_true_fig(plt, d: pd.DataFrame, coin: str, title: str):
    fig = plt.figure(figsize=(14, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    top, bottom, sc = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    t = d["end_time"].to_numpy()
    mu, y = d["mu"].to_numpy(), d["y_ret"].to_numpy()
    _series(top, t, mu, y, "close return", band=d["sigma"].to_numpy())
    top.set_title(f"{title} {coin}: close return, predicted vs realised (band = predicted sd)", fontsize=10)
    has_hl = {"pred_high", "fut_high", "pred_low", "fut_low"} <= set(d.columns)
    if has_hl:
        bottom.plot(t, d["fut_high"], color="#264653", lw=0.8, label="realised high")
        bottom.plot(t, d["pred_high"], color="#e76f51", lw=0.8, label="predicted high")
        bottom.plot(t, d["fut_low"], color="#2a9d8f", lw=0.8, label="realised low")
        bottom.plot(t, d["pred_low"], color="#e9c46a", lw=0.8, label="predicted low")
        bottom.legend(fontsize=7, loc="upper left")
    else:
        bottom.text(0.5, 0.5, "no high / low heads in the table", ha="center", transform=bottom.transAxes)
    bottom.set_title("high / low return, predicted vs realised", fontsize=10)
    bottom.grid(alpha=0.25)
    sub = d.sample(min(len(d), MAX_POINTS), random_state=0)
    sc.scatter(sub["mu"], sub["y_ret"], s=4, alpha=0.4, color="#264653")
    lim = float(np.nanmax(np.abs(np.r_[mu, y]))) if len(d) else 1.0
    sc.plot([-lim, lim], [-lim, lim], color="#6c757d", lw=0.8, ls="--")
    sc.set_xlabel("predicted close return")
    sc.set_ylabel("realised close return")
    sc.set_title(f"scatter: rank IC {metrics.rank_ic(y, mu):+.3f} on {len(d):,} rows", fontsize=10)
    sc.grid(alpha=0.25)
    return fig


def pred_vs_true(table: pd.DataFrame, coins, out_dir, title: str = "") -> list[Path]:
    """One PNG per coin: the close return predicted vs realised over time (+-1 sigma band), the high / low predicted
    vs realised, and the predicted-vs-realised scatter with the rank IC in its title. ``table`` is a battery table
    (``eval.battery_adapter.to_battery_table``), ``coins`` coin names; a coin that is not in the table raises
    ``ValueError`` naming the coins it has. Files: ``<out_dir>/<title>_pred_vs_true_<coin>.png``; returns the paths."""
    need = {"coin", "end_time", "mu", "sigma", "y_ret"}
    if need - set(table.columns):
        raise ValueError(f"pred_vs_true: the table lacks columns {sorted(need - set(table.columns))}")
    coins = [coins] if isinstance(coins, str) else list(coins)
    if not coins:
        raise ValueError("pred_vs_true: no coin given")
    have = set(table["coin"].astype(str))
    missing = [c for c in coins if c not in have]
    if missing:
        raise ValueError(f"pred_vs_true: coins {missing} are not in the table (it has {len(have)}, e.g. "
                         f"{sorted(have)[:5]})")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for coin in coins:
        d = table[table["coin"].astype(str) == coin].sort_values("end_time")
        fig = _pred_vs_true_fig(plt, d, coin, title)
        fig.tight_layout()
        path = out_dir / f"{title or 'stage'}_pred_vs_true_{coin}.png"
        fig.savefig(path, dpi=90)
        plt.close(fig)
        paths.append(path)
    return paths


def _price_series(d: pd.DataFrame, head: str):
    """(actual price, predicted price) of one head for one coin's rows (sorted by end_time)."""
    lc = d["last_close"].to_numpy(dtype=float)
    ret_p, ret_a = ("mu", "y_ret") if head == "close" else (f"pred_{head}", f"fut_{head}")
    return lc * (1.0 + d[ret_a].to_numpy(dtype=float)), lc * (1.0 + d[ret_p].to_numpy(dtype=float))


def improve_vs_naive(actual: np.ndarray, pred: np.ndarray) -> float:
    """1 - MAE(pred) / MAE(naive) in percent over all rows; naive = the previous row's actual (row 0 is dropped).
    100 = perfect, 0 = as good as repeating the previous value, negative = worse. NaN if the naive MAE is 0."""
    a, p = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    if a.shape[0] < 2:
        return float("nan")
    err_n, err_p = metrics.mae(a[1:], a[:-1]), metrics.mae(a[1:], p[1:])
    return 100.0 * (1.0 - err_p / err_n) if err_n > 0 else float("nan")


def _lag_corr(actual: np.ndarray, pred: np.ndarray, max_lag: int) -> dict[int, float]:
    """corr(pred[t], actual[t + k]) for k in -max_lag..max_lag (``stats.metrics.pearson``, overlapping rows only)."""
    n, out = len(actual), {}
    for k in range(-max_lag, max_lag + 1):
        lo, hi = max(0, -k), min(n, n - k)
        out[k] = metrics.pearson(pred[lo:hi], actual[lo + k:hi + k]) if hi - lo >= 2 else float("nan")
    return out


def _price_head(ax_price, ax_lag, d: pd.DataFrame, coin: str, head: str, last_n: int, max_lag: int) -> None:
    actual, pred = _price_series(d, head)
    imp = improve_vs_naive(actual, pred)
    naive = np.r_[np.nan, actual[:-1]]
    k = min(last_n, len(d))
    x = np.arange(k)
    ax_price.plot(x, actual[-k:], color="#2a9d8f", marker="o", ms=3, lw=1.2, label="Actual")
    ax_price.plot(x, pred[-k:], color="#f4a261", marker="o", ms=3, lw=1.2, label="Predicted")
    ax_price.plot(x, naive[-k:], color="#6c757d", ls="--", lw=1.0, label="Naive Baseline (t-1)")
    ax_price.set_title(f"{coin} - {head} (MAE improve: {imp:.1f}%)", fontsize=10)
    ax_price.legend(fontsize=7)
    ax_price.grid(alpha=0.25)
    lags = _lag_corr(actual, pred, max_lag)
    ax_lag.bar(list(lags), [0.0 if np.isnan(v) else v for v in lags.values()],
               color=["#d62828" if kk == 0 else "#457b9d" for kk in lags])
    ax_lag.set_title(f"Lag Scan - {coin} - {head}", fontsize=10)
    ax_lag.set_xlabel("lag k (corr of predicted(t) with actual(t + k))")
    ax_lag.grid(alpha=0.25)


def price_panels(table: pd.DataFrame, coins, out_dir, heads=TARGET_COLUMNS, last_n: int = 16,
                 max_lag: int = 5, title: str = "") -> list[Path]:
    """TF price-level figures, one PNG per coin: per head a price panel (Actual / Predicted / Naive t-1 over the last
    ``last_n`` rows, title "MAE improve" over ALL the coin's rows) and below it the Lag Scan bars (k = -max_lag..
    max_lag, k = 0 red). close: price = last_close * (1 + mu / y_ret); high / low from ``pred_`` / ``fut_``. Needs
    the battery table's ``last_close`` (else ValueError: re-run the battery step on this commit); a coin that is not
    in the table raises like ``pred_vs_true``; a head without columns is skipped. Files:
    ``<out_dir>/<title>_price_panels_<coin>.png``; returns the paths."""
    if "last_close" not in table.columns:
        raise ValueError("price_panels: the table has no 'last_close' column; re-run the battery step on this "
                         "commit (older battery tables were written without it)")
    need = {"coin", "end_time", "mu", "y_ret"}
    if need - set(table.columns):
        raise ValueError(f"price_panels: the table lacks columns {sorted(need - set(table.columns))}")
    coins = [coins] if isinstance(coins, str) else list(coins)
    if not coins:
        raise ValueError("price_panels: no coin given")
    have = set(table["coin"].astype(str))
    missing = [c for c in coins if c not in have]
    if missing:
        raise ValueError(f"price_panels: coins {missing} are not in the table (it has {len(have)}, e.g. "
                         f"{sorted(have)[:5]})")
    use = [h for h in heads if h == "close" or {f"pred_{h}", f"fut_{h}"} <= set(table.columns)]
    if not use:
        raise ValueError(f"price_panels: none of the heads {list(heads)} is in the table")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for coin in coins:
        d = table[table["coin"].astype(str) == coin].sort_values("end_time")
        fig, ax = plt.subplots(2, len(use), figsize=(6 * len(use), 8), squeeze=False)
        for j, head in enumerate(use):
            _price_head(ax[0, j], ax[1, j], d, coin, head, last_n, max_lag)
        if title:
            fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        path = out_dir / f"{title or 'stage'}_price_panels_{coin}.png"
        fig.savefig(path, dpi=90)
        plt.close(fig)
        paths.append(path)
    return paths
