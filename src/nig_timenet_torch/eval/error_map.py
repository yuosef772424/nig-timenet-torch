"""PURPOSE: Where the model itself errs: per head, error (|mu - y|, NIG NLL), rank IC, bias, coverage, interval width
    and value over the HAR baseline, split by liquidity proxy, coin age, volatility regime, month, market day and hour.
TAGS: error analysis, error map, buckets, bias, coverage, interval width, har baseline, worst cases, model improvement
PITFALLS: works in the label space (log move vs the codec anchor); bucket edges come from VAL only, never TEST; the
    high / low direction is never read (bias is about the size of the move); liquidity is a PROXY (share of flat bars
    in the coin's train windows), the artifact has no volume.

An error frame has one row per prediction: ``coin``, ``end_time``, ``day``, ``month``, ``hour``, the bucket variables
(``stale_share``, ``age_bars``, ``vol``, ``market_day``) and per head ``h``: ``mu_h``, ``real_h``, ``std_h`` (predictive
std), ``half_h`` (80 % halfwidth), ``nll_h``, ``har_h`` (HAR OLS prediction of y fitted on VAL; high / low only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import TARGET_COLUMNS
from nig_timenet_torch.stats import metrics

__all__ = ["BUCKET_VARS", "edges_from", "bucket_rows", "bias", "uncertainty", "worst_cases", "error_map", "render"]

#: bucket variable -> number of quantile buckets (edges from VAL), a tuple of fixed inner edges, or None (categorical:
#: month, hour). stale_share is mostly exactly 0, so it gets fixed edges; NaN (a coin with no TRAIN rows, i.e. listed
#: after the train end) is its own bucket "no value".
BUCKET_VARS = {"stale_share": (1e-12, 0.01, 0.05), "age_bars": 4, "vol": 5, "market_day": 5, "month": None,
               "hour": None}
MIN_ROWS = 200


def edges_from(val: pd.DataFrame, var: str, n: int) -> np.ndarray:
    """Inner quantile edges of ``var`` on VAL (n buckets)."""
    return np.unique(np.quantile(val[var].to_numpy(dtype=np.float64), np.linspace(0, 1, n + 1)[1:-1]))


def _labels(frame: pd.DataFrame, var: str, edges) -> np.ndarray:
    if edges is None:
        return frame[var].astype(str).to_numpy()
    v = frame[var].to_numpy(dtype=np.float64)
    return np.where(np.isnan(v), -1, np.searchsorted(np.asarray(edges), v, side="right"))


def _line_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.std(x) == 0:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _cell(f: pd.DataFrame, h: str) -> dict:
    mu, y, std, half = (f[f"{k}_{h}"].to_numpy(dtype=np.float64) for k in ("mu", "real", "std", "half"))
    err = np.abs(mu - y)
    slope, intercept = _line_fit(mu, y)
    out = {"n": int(len(f)), "mae": float(err.mean()), "nll": float(f[f"nll_{h}"].mean()),
           "rank_ic": metrics.rank_ic(y, mu, f["end_time"].to_numpy()), "coverage": metrics.interval_coverage(
               y, mu - half, mu + half), "width_ratio": float(np.median(half) / max(np.quantile(err, 0.8), 1e-12)),
           "std_err_spearman": metrics.spearman(std, err), "pred_mean": float(mu.mean()), "real_mean": float(y.mean()),
           "slope": slope, "intercept": intercept}
    if f"har_{h}" in f:
        har = f[f"har_{h}"].to_numpy(dtype=np.float64)
        sign = -1.0 if h == "low" else 1.0
        ts = f["end_time"].to_numpy()
        out.update({"har_mae": float(np.abs(har - y).mean()), "har_rank_ic": metrics.rank_ic(y, har, ts),
                    "net_of_har_ic": metrics.partial_rank_ic(sign * y, sign * mu, sign * har, ts)})
    return out


def bucket_rows(frame: pd.DataFrame, h: str, var: str, edges) -> list[dict]:
    """One ``_cell`` per bucket of ``var`` (buckets with < MIN_ROWS rows skipped)."""
    lab = _labels(frame, var, edges)
    rows = []
    for b in sorted(set(lab.tolist())):
        m = lab == b
        if m.sum() < MIN_ROWS:
            continue
        name = str(b) if edges is None else _range_name(np.asarray(edges), int(b))
        rows.append({"bucket": name, **_cell(frame[m], h)})
    return rows


def _range_name(edges, b: int) -> str:
    if b < 0:
        return "no value"
    lo = "-inf" if b == 0 else f"{edges[b - 1]:.4g}"
    hi = "+inf" if b == len(edges) else f"{edges[b]:.4g}"
    return f"[{lo}, {hi})"


def bias(frame: pd.DataFrame, h: str, n: int = 10) -> dict:
    """Predicted vs realised: overall OLS of y on mu, by decile of the prediction and by quintile of the realised
    size |y| (large moves), plus the spread of per-coin slopes (coins with >= MIN_ROWS rows)."""
    mu, y = frame[f"mu_{h}"].to_numpy(dtype=np.float64), frame[f"real_{h}"].to_numpy(dtype=np.float64)
    slope, intercept = _line_fit(mu, y)

    def by(key, k):
        q = pd.qcut(key, k, labels=False, duplicates="drop")
        return [{"bucket": int(b), "pred_mean": float(mu[q == b].mean()), "real_mean": float(y[q == b].mean()),
                 "n": int((q == b).sum())} for b in np.unique(q)]
    coin_slopes = [s for _, g in frame.groupby("coin") if len(g) >= MIN_ROWS
                   for s in [_line_fit(g[f"mu_{h}"].to_numpy(), g[f"real_{h}"].to_numpy())[0]] if np.isfinite(s)]
    return {"slope": slope, "intercept": intercept, "by_pred_decile": by(mu, n), "by_real_size": by(np.abs(y), 5),
            "coin_slope_quantiles": {f"p{q}": float(np.quantile(coin_slopes, q / 100)) for q in (10, 50, 90)}
            if coin_slopes else {}}


def uncertainty(frame: pd.DataFrame, h: str, n: int = 10) -> dict:
    """Does the predicted std grow with the realised error? Spearman(std, |err|) and the binned curve by std decile
    (mean |err|, coverage of the 80 % interval, halfwidth / q80 |err|)."""
    mu, y, std, half = (frame[f"{k}_{h}"].to_numpy(dtype=np.float64) for k in ("mu", "real", "std", "half"))
    err = np.abs(mu - y)
    q = pd.qcut(std, n, labels=False, duplicates="drop") if np.ptp(std) > 0 else np.full(len(std), np.nan)
    curve = [{"bucket": int(b), "std_mean": float(std[q == b].mean()), "abs_err_mean": float(err[q == b].mean()),
              "coverage": metrics.interval_coverage(y[q == b], (mu - half)[q == b], (mu + half)[q == b]),
              "width_ratio": float(np.median(half[q == b]) / max(np.quantile(err[q == b], 0.8), 1e-12))}
             for b in np.unique(q[np.isfinite(q)])]
    return {"std_err_spearman": metrics.spearman(std, err), "curve": curve}


def worst_cases(frame: pd.DataFrame, h: str, k: int = 15, coin_info: pd.DataFrame | None = None) -> dict:
    """Coins and days with the largest mean |err|, and how the worst coins differ from all coins in the
    ``coin_info`` columns (e.g. stale share, age, volatility, gaps)."""
    f = frame.assign(err=np.abs(frame[f"mu_{h}"] - frame[f"real_{h}"]))
    coins = f.groupby("coin").agg(n=("err", "size"), mae=("err", "mean"), vol=("vol", "mean"))
    coins = coins[coins["n"] >= 50].sort_values("mae", ascending=False)
    days = f.groupby("day").agg(n=("err", "size"), mae=("err", "mean"), market=("market_day", "first"))
    days = days.sort_values("mae", ascending=False)
    out = {"coins": coins.head(k).reset_index().to_dict("records"),
           "days": days.head(k).reset_index().to_dict("records")}
    if coin_info is not None:
        info = coin_info.reindex(coins.index)
        out["worst_vs_all"] = {c: {"worst": float(info[c].head(k).mean()), "all": float(info[c].mean())}
                               for c in info.columns}
    return out


def error_map(frames: dict, heads=tuple(TARGET_COLUMNS), coin_info: pd.DataFrame | None = None) -> dict:
    """The whole map for ``frames`` {"VAL": frame, "TEST": frame}: per head and split the overall cell, the bucket
    tables of every BUCKET_VARS variable (edges from VAL), bias, uncertainty and worst cases."""
    val = frames["VAL"]
    edges = {v: (n if n is None or isinstance(n, tuple) else edges_from(val, v, n)) for v, n in BUCKET_VARS.items()}
    out = {}
    for h in heads:
        for split, f in frames.items():
            out[f"{h} | {split}"] = {"overall": _cell(f, h),
                                     "buckets": {v: bucket_rows(f, h, v, edges[v]) for v in BUCKET_VARS},
                                     "bias": bias(f, h), "uncertainty": uncertainty(f, h),
                                     "worst": worst_cases(f, h, coin_info=coin_info)}
    return out


def _fmt_cell(name: str, c: dict) -> str:
    har = (f"  HAR mae {c['har_mae'] * 100:.3f}% IC {c['har_rank_ic']:+.3f} net {c['net_of_har_ic']:+.3f}"
           if "har_mae" in c else "")
    return (f"    {name:<22}n={c['n']:>7,}  mae {c['mae'] * 100:.3f}%  nll {c['nll']:+.3f}  IC {c['rank_ic']:+.3f}  "
            f"cov {c['coverage']:.3f}  width/q80err {c['width_ratio']:.2f}  sp(std,err) {c['std_err_spearman']:+.2f}  "
            f"pred {c['pred_mean'] * 100:+.2f}% real {c['real_mean'] * 100:+.2f}% slope {c['slope']:+.2f}" + har)


def render(m: dict) -> str:
    lines = []
    for key, d in m.items():
        lines += ["", f"=== {key}", _fmt_cell("overall", d["overall"])]
        for var, rows in d["buckets"].items():
            lines.append(f"  by {var}:")
            lines += [_fmt_cell(r["bucket"], r) for r in rows]
        b, u = d["bias"], d["uncertainty"]
        lines.append(f"  bias: realised = {b['intercept'] * 100:+.3f}% + {b['slope']:.2f} x predicted ; per-coin slope "
                     + " ".join(f"{k} {v:+.2f}" for k, v in b["coin_slope_quantiles"].items()))
        lines.append("    by predicted decile (pred -> real %): " + " ".join(
            f"{r['pred_mean'] * 100:+.2f}->{r['real_mean'] * 100:+.2f}" for r in b["by_pred_decile"]))
        lines.append("    by realised |y| quintile (pred -> real %): " + " ".join(
            f"{r['pred_mean'] * 100:+.2f}->{r['real_mean'] * 100:+.2f}" for r in b["by_real_size"]))
        lines.append(f"  uncertainty: Spearman(std, |err|) = {u['std_err_spearman']:+.3f}; by std decile "
                     "(std%, |err|%, cov, width/q80err): " + " ".join(
                         f"({r['std_mean'] * 100:.2f}, {r['abs_err_mean'] * 100:.2f}, {r['coverage']:.2f}, "
                         f"{r['width_ratio']:.2f})" for r in u["curve"]))
        w = d["worst"]
        lines.append("  worst coins (mae%, n, vol%): " + ", ".join(
            f"{r['coin']} {r['mae'] * 100:.2f} ({r['n']}, {r['vol'] * 100:.1f})" for r in w["coins"][:10]))
        lines.append("  worst days (mae%, n, market%): " + ", ".join(
            f"{str(r['day'])[:10]} {r['mae'] * 100:.2f} ({r['n']}, {r['market'] * 100:+.2f})" for r in w["days"][:8]))
        if "worst_vs_all" in w:
            lines.append("  worst 15 coins vs all coins: " + ", ".join(
                f"{k} {v['worst']:.4g} vs {v['all']:.4g}" for k, v in w["worst_vs_all"].items()))
    return "\n".join(lines)
