"""Report C, model-free: what each normalisation of the close label does on a contract artifact (no network, numpy).

    python docs/audit/2026-11-16-panel/normalisation_look.py ARTIFACT OUT [--top-coins 100] [--max-rows N]

Rows: the run's own time split with whole timestamps (``SplitSettings(cross_section=True, top_coins=...)``); every
statistic (beta_i, sigma_i, feature scaling, ridge weights) is fitted on TRAIN rows and scored on VAL rows; the TEST
rows are never read. Per target variant it writes: the cross-coin correlation and the effective number of coins
(the formula of ``doctor.data_checks.overlap``), the rank IC a ridge on simple features reaches (inside the
timestamp, against the variant itself and against the realised relative move) and the two day metrics of that ridge
next to the baselines (``eval/day_metrics.py``). Output: OUT/normalisation_look.json and .md.
"""

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX, TS_COL
from nig_timenet_torch.data_io.loading import LoaderSettings, load_dataset
from nig_timenet_torch.eval import day_metrics
from nig_timenet_torch.stats import metrics
from nig_timenet_torch.workflow import run_common
from nig_timenet_torch.workflow.flat_run import row_coins
from nig_timenet_torch.workflow.panel_report import trailing_return
from nig_timenet_torch.workflow.settings import SplitSettings

VARIANTS = ("raw", "minus_mean", "minus_beta_market", "beta_resid_over_sigma", "cs_rank", "minus_beta_top10",
            "minus_beta_btc")


def by_time(v, t, how="mean"):
    return pd.Series(v).groupby(t).transform(how).to_numpy()


def cross_corr(v, t):
    """(rho, effective coins): the mean pairwise correlation of same-timestamp values implied by the variance of the
    timestamp means (doctor.data_checks.overlap)."""
    g = pd.Series(v).groupby(t)
    n, var = float(g.count().mean()), float(np.var(v, ddof=1))
    rho = (float(g.mean().var()) - var / n) / (var * (1.0 - 1.0 / n))
    return rho, n / (1.0 + (n - 1.0) * max(rho, 0.0))


def rows_of(ds, idx):
    lc = np.asarray(ds["last_candles"])[idx]
    close = lc[:, LAST_COLUMN_INDEX["last_close"]]
    fut = lc[:, LAST_COLUMN_INDEX["future_close"]]
    x = np.asarray(ds[run_common.input_key(ds)])[idx].astype(np.float64)
    fo = list(ds["feature_order"])
    hi, lo, cl = (x[:, :, fo.index(k)] for k in ("high", "low", "close"))
    t = lc[:, TS_COL].astype(np.int64)
    rng = hi - lo
    own = np.stack([cl[:, -1], cl[:, -2:].sum(1), cl[:, -6:].sum(1), cl.sum(1), rng[:, -1], rng[:, -6:].mean(1),
                    rng.mean(1), hi[:, -1] + lo[:, -1] - 2 * cl[:, -1]], axis=1)
    rel = own - np.stack([by_time(own[:, j], t) for j in range(own.shape[1])], axis=1)
    asym = np.log(lc[:, LAST_COLUMN_INDEX["last_high"]] / close) + np.log(lc[:, LAST_COLUMN_INDEX["last_low"]] / close)
    return {"t": t, "coin": row_coins(ds, idx).astype(str), "y": np.log(fut / close), "ret": fut / close - 1.0,
            "own": own, "rel": rel, "ret_1d": trailing_return(ds, idx, 6), "asym": asym}


def markets(r, top10, btc):
    """Per-row market value of three definitions: equal weight, the 10 longest-listed coins, BTC alone."""
    s = pd.Series(r["y"])
    eq = by_time(r["y"], r["t"])
    top = s.where(np.isin(r["coin"], top10)).groupby(r["t"]).transform("mean").to_numpy()
    b = s.where(r["coin"] == btc).groupby(r["t"]).transform("mean").to_numpy()
    return {"eq": eq, "top10": np.where(np.isnan(top), eq, top), "btc": np.where(np.isnan(b), eq, b)}


def fit_coin_stats(tr, mk):
    """Per coin, on TRAIN rows: beta of y on each market, and the sd of the equal-weight residual."""
    df = pd.DataFrame({"coin": tr["coin"], "y": tr["y"], **{f"m_{k}": v for k, v in mk.items()}})
    out = {}
    for coin, g in df.groupby("coin"):
        betas = {k: float(np.cov(g["y"], g[f"m_{k}"])[0, 1] / max(np.var(g[f"m_{k}"], ddof=1), 1e-12)) for k in mk}
        out[coin] = {**betas, "sigma": float(np.std(g["y"] - betas["eq"] * g["m_eq"])) or 1.0}
    return out


def variants(r, mk, stats):
    get = lambda k: np.array([stats[c][k] for c in r["coin"]])  # noqa: E731
    resid = r["y"] - get("eq") * mk["eq"]
    rank = pd.Series(r["y"]).groupby(r["t"]).rank(pct=True).to_numpy() - 0.5
    return {"raw": r["y"], "minus_mean": r["y"] - mk["eq"], "minus_beta_market": resid,
            "beta_resid_over_sigma": resid / get("sigma"), "cs_rank": rank,
            "minus_beta_top10": r["y"] - get("top10") * mk["top10"], "minus_beta_btc": r["y"] - get("btc") * mk["btc"]}


def ridge(f_tr, y_tr, f_va, lam=1e-2):
    mu, sd = f_tr.mean(0), f_tr.std(0) + 1e-12
    a, b = (f_tr - mu) / sd, (f_va - mu) / sd
    w = np.linalg.solve(a.T @ a + lam * len(a) * np.eye(a.shape[1]), a.T @ (y_tr - y_tr.mean()))
    return b @ w + y_tr.mean()


def frame(r, pred):
    return pd.DataFrame({"time": pd.to_datetime(r["t"]), "coin": r["coin"], "mu": pred, "sigma": 1.0, "y_ret": r["ret"],
                         "ret_1d": r["ret_1d"], "asym": r["asym"]})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("artifact"), p.add_argument("out")
    p.add_argument("--top-coins", type=int, default=100), p.add_argument("--max-rows", type=int, default=None)
    a = p.parse_args()
    src = load_dataset(LoaderSettings(name=a.artifact, local_dir="", mmap=True))
    ds = src.dataset
    (tr_i, va_i, _), bounds = run_common.make_plan(ds, SplitSettings(cross_section=True, top_coins=a.top_coins,
                                                                     max_rows=a.max_rows))
    tr, va = rows_of(ds, tr_i), rows_of(ds, va_i)
    counts = pd.Series(tr["coin"]).value_counts()
    top10, btc = list(counts.index[:10]), "BTCUSDT" if "BTCUSDT" in counts.index else counts.index[0]
    mk_tr, mk_va = markets(tr, top10, btc), markets(va, top10, btc)
    stats = fit_coin_stats(tr, mk_tr)
    for c in set(va["coin"]) - set(stats):                       # a coin without train rows: market beta 1
        stats[c] = {"eq": 1.0, "top10": 1.0, "btc": 1.0, "sigma": float(np.std(tr["y"]))}
    v_tr, v_va = variants(tr, mk_tr, stats), variants(va, mk_va, stats)
    rel_va = va["y"] - mk_va["eq"]
    feats = {"own": (tr["own"], va["own"]), "own+relative": (np.hstack([tr["own"], tr["rel"]]),
                                                               np.hstack([va["own"], va["rel"]]))}
    rows = []
    for name in VARIANTS:
        rho, eff = cross_corr(v_va[name], va["t"])
        row = {"variant": name, "cross_coin_corr": rho, "effective_coins": eff}
        for fs, (f_tr, f_va) in feats.items():
            pred = ridge(f_tr, v_tr[name], f_va)
            rep = day_metrics.day_report(frame(va, pred), None)
            row[fs] = {"rank_ic_own_target": metrics.rank_ic(v_va[name], pred, va["t"]),
                       "rank_ic_relative_move": metrics.rank_ic(rel_va, pred, va["t"]),
                       "pearson_pooled_own_target": metrics.pearson(v_va[name], pred),
                       "absolute": rep["absolute"]["model"], "neutral": rep["neutral"]["model"]}
        rows.append(row)
    base = day_metrics.day_report(frame(va, np.zeros(len(va["t"]))), None)
    betas = np.array([stats[c]["eq"] for c in counts.index])
    res = {"artifact": src.manifest.data_fingerprint, "source_commit": src.manifest.source_commit,
           "rows": {"train": int(len(tr_i)), "val": int(len(va_i))}, "split": [str(b) for b in bounds],
           "coins": int(len(counts)), "coins_per_timestamp_val": float(pd.Series(va["t"]).value_counts().mean()),
           "beta": {"min": float(betas.min()), "median": float(np.median(betas)), "max": float(betas.max()),
                    "sd": float(betas.std())},
           "market_share_of_variance_val": float(np.var(np.array([stats[c]["eq"] for c in va["coin"]]) * mk_va["eq"])
                                                 / np.var(va["y"])),
           "variants": rows,
           "baselines": {m: {k: v["gross"] for k, v in base[m].items() if k != "model"}
                         for m in ("absolute", "neutral")},
           "baseline_relative_rank_ic": {k: v for k, v in base["relative_rank_ic"].items() if k != "model"}}
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "normalisation_look.json").write_text(json.dumps(res, indent=1, default=float))
    pct = lambda d: (f"{d['gross']['win_share'] * 100:.1f} % [{d['gross']['win_lo'] * 100:.0f}, "  # noqa: E731
                     f"{d['gross']['win_hi'] * 100:.0f}]" if d["gross"].get("n_days") else "n/a")
    lines = [f"Artifact `{res['artifact'][:12]}`, {res['coins']} coins, train {res['rows']['train']} / val "
             f"{res['rows']['val']} rows, split {res['split']}; beta median {res['beta']['median']:.2f} "
             f"({res['beta']['min']:.2f} .. {res['beta']['max']:.2f}); market share of the close variance "
             f"{res['market_share_of_variance_val']:.2f}.", "",
             "| target variant | cross-coin corr | effective coins | ridge rank IC, own target (own / +relative "
             "features) | rank IC of the relative move | metric 1 winning days | metric 2 winning days |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        o, b = r["own"], r["own+relative"]
        lines.append(f"| {r['variant']} | {r['cross_coin_corr']:+.3f} | {r['effective_coins']:.1f} | "
                     f"{o['rank_ic_own_target']:+.4f} / {b['rank_ic_own_target']:+.4f} | "
                     f"{o['rank_ic_relative_move']:+.4f} / {b['rank_ic_relative_move']:+.4f} | {pct(b['absolute'])} | "
                     f"{pct(b['neutral'])} |")
    lines += ["", "Baselines (VAL, gross winning days): " + "; ".join(
        f"{m} {k} {v['win_share'] * 100:.1f} % ({v['n_days']} d)" for m in ("absolute", "neutral")
        for k, v in res["baselines"][m].items() if v.get("n_days")) + "."]
    (out / "normalisation_look.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
