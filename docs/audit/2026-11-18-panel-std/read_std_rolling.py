"""Reading of the panel std walk-forward (owner task 2026-11-18): candidate vs reference, paired by fold and seed,
the pooled day books, and the failure reading of rule C7 from the saved TEST / VAL frames.

Usage (on Colab, where the run folders live):
    python docs/audit/2026-11-18-panel-std/read_std_rolling.py \
        --cand /content/drive/MyDrive/nig_timenet_runs/panel-rolling-std-001/out \
        --ref /content/drive/MyDrive/nig_timenet_runs/panel-rolling-ref-002/out \
        --old /content/drive/MyDrive/nig_timenet_runs/panel-rolling-001/out \
        --out /content/drive/MyDrive/nig_timenet_runs/panel-rolling-std-001/reading
Writes reading.json and reading.md into --out. Exploratory (rule C5).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.eval import day_metrics as dm
from nig_timenet_torch.stats import metrics

COST = 0.0008
HEADS = ("close", "high", "low")
FOLD_KEYS = [f"targets.{h}.{k}" for h in HEADS for k in ("skill.rank_ic_by_timestamp", "sharp.coverage",
                                                           "sharp.share_std_below_move")]


def fold_rows(out: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(out.glob("fold_*_seed_*/fold_row.json")):
        r = json.loads(f.read_text())
        rows.append({"fold": r["fold"], "seed": r["seed"], "best_epoch": r.get("best_epoch"),
                     "epochs_done": r.get("epochs_done"), **{k: r["metrics"].get(k, np.nan) for k in FOLD_KEYS}})
    return pd.DataFrame(rows).set_index(["fold", "seed"]).sort_index()


def frames(out: Path, stage: str, seed: int) -> pd.DataFrame:
    parts = []
    for d in sorted(out.glob(f"fold_*_seed_{seed}")):
        p = d / f"panel_frame_{stage}.parquet"
        if p.is_file():
            f = pd.read_parquet(p)
            f["fold"] = int(d.name.split("_")[1])
            parts.append(f)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def tsid(f: pd.DataFrame) -> np.ndarray:
    return f["time"].to_numpy().astype("datetime64[ns]").astype(np.int64)


def rel_ic(f: pd.DataFrame, score) -> float:
    return dm.relative_rank_ic(f, score) if len(f) else float("nan")


def group_row(f: pd.DataFrame, name: str) -> dict:
    """Score of one group of rows: the close-sign trade (metric 1 rows) and the relative ranking, next to the best
    trivial baseline on the same rows."""
    if len(f) < 30:
        return {"group": name, "rows": int(len(f))}
    mu, y = f["mu"].to_numpy(np.float64), f["y_ret"].to_numpy(np.float64)
    s = np.sign(mu)
    gross = s * y
    days = dm.absolute_days(f, mu, min_rows=1)
    summ = dm.day_summary(days)
    base = {"always_long": np.ones(len(f)), "last_bar_asymmetry": f["asym"].to_numpy(np.float64),
            "momentum_1d": f["ret_1d"].to_numpy(np.float64)}
    base_gross = {k: float(np.mean(np.sign(v) * y)) for k, v in base.items()}
    best = max(base_gross, key=base_gross.get)
    return {"group": name, "rows": int(len(f)), "days": int(summ.get("n_days", 0)),
            "hit": float(np.mean(s == np.sign(y))), "gross": float(gross.mean()),
            "net": float(gross.mean() - 2 * COST), "day_mean_lo": summ.get("mean_lo"), "day_mean_hi": summ.get("mean_hi"),
            "rel_rank_ic": rel_ic(f, mu), "best_base": best, "best_base_gross": base_gross[best],
            "share_long": float(np.mean(s > 0)), "share_up": float(np.mean(y > 0))}


def error_map(f: pd.DataFrame) -> list[dict]:
    out = [group_row(f, "all")]
    z = np.abs(f["mu"]) / f["sigma"]
    for col, vals in (("sigma", f["sigma"]), ("|mu|", f["mu"].abs()), ("|mu|/sigma", z)):
        q = pd.qcut(vals.rank(method="first"), 5, labels=False)
        out += [group_row(f[q == i], f"{col} quintile {i + 1}") for i in range(5)]
    out += [group_row(f[f["mu"] > 0], "side long"), group_row(f[f["mu"] < 0], "side short")]
    t = tsid(f)
    mkt = pd.Series(f["ret_1d"].to_numpy()).groupby(t).transform("mean").to_numpy()
    out += [group_row(f[mkt > 0.01], "market 24h up > 1 %"), group_row(f[mkt < -0.01], "market 24h down < -1 %"),
            group_row(f[np.abs(mkt) <= 0.01], "market 24h flat")]
    disp = pd.Series(f["ret_1d"].to_numpy()).groupby(t).transform("std").to_numpy()
    hi = disp > np.nanmedian(disp)
    out += [group_row(f[hi], "cross-section dispersion high"), group_row(f[~hi], "cross-section dispersion low")]
    out += [group_row(f[f["fold"] == k], f"fold {k}") for k in sorted(f["fold"].unique())]
    yr = f["time"].dt.year
    out += [group_row(f[yr == y], f"year {y}") for y in sorted(yr.unique())]
    return out


def worst_rows(f: pd.DataFrame, n: int = 15) -> list[dict]:
    z = (f["mu"] / f["sigma"]).to_numpy()
    wrong = np.sign(f["mu"]) != np.sign(f["y_ret"])
    g = f[wrong].assign(z=z[wrong.to_numpy()], loss=np.abs(f["y_ret"][wrong]))
    g = g.assign(score=np.abs(g["z"]) * g["loss"]).nlargest(n, "score")
    return [{"time": str(r.time), "coin": r.coin, "fold": int(r.fold), "mu": float(r.mu), "sigma": float(r.sigma),
             "y_ret": float(r.y_ret), "ret_1d": float(r.ret_1d), "asym": float(r.asym)} for r in g.itertuples()]


def worst_days(f: pd.DataFrame, n: int = 10) -> list[dict]:
    d = dm.neutral_days(f, f["mu"].to_numpy(), cost=COST)
    t = tsid(f)
    mk = pd.Series(f["y_ret"].to_numpy()).groupby(f["time"].dt.floor("D").to_numpy()).mean()
    return [{"day": str(k.date()), "neutral_net": float(v), "market_mean_y": float(mk.get(k, np.nan))}
            for k, v in d.nsmallest(n).items()] if len(t) else []


def bias(f: pd.DataFrame) -> dict:
    res = {"share_mu_up": float(np.mean(f["mu"] > 0)), "share_y_up": float(np.mean(f["y_ret"] > 0))}
    res["per_fold"] = {int(k): {"mu_up": float(np.mean(g["mu"] > 0)), "y_up": float(np.mean(g["y_ret"] > 0)),
                                "long_gross": float(g.loc[g["mu"] > 0, "y_ret"].mean()),
                                "short_gross": float(-g.loc[g["mu"] < 0, "y_ret"].mean())}
                       for k, g in f.groupby("fold")}
    return res


def recenter(val: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Bias correction fitted on VAL per fold: subtract the VAL median of mu, then the close-sign trade on TEST."""
    parts = []
    for k, g in test.groupby("fold"):
        m = float(val.loc[val["fold"] == k, "mu"].median())
        parts.append(g.assign(mu=g["mu"] - m))
    t2 = pd.concat(parts)
    a = dm.day_summary(dm.absolute_days(t2, t2["mu"].to_numpy(), cost=COST))
    return {"net_day_mean": a.get("mean"), "lo": a.get("mean_lo"), "hi": a.get("mean_hi"),
            "share_long": float(np.mean(t2["mu"] > 0))}


RULES = {"top |mu|/sigma": lambda f: np.abs(f["mu"]) / f["sigma"], "low sigma": lambda f: -f["sigma"],
         "top |mu|": lambda f: np.abs(f["mu"]),
         "agree with last-bar asymmetry": lambda f: np.where(np.sign(f["mu"]) == np.sign(f["asym"]),
                                                             np.abs(f["mu"]) / f["sigma"], -np.inf)}
KEEP = (0.05, 0.1, 0.2, 0.5)


def selective(val: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    """Every rule x kept share: threshold = the VAL quantile of the rule's score in each fold, applied to that fold's
    TEST rows; the close-sign trade on the kept rows, net per trade and per day (n_variants = len(RULES) x len(KEEP))."""
    out = []
    for name, fn in RULES.items():
        for keep in KEEP:
            kept = []
            for k, g in test.groupby("fold"):
                v = val[val["fold"] == k]
                sv = np.asarray(fn(v), dtype=np.float64)
                sv = sv[np.isfinite(sv)]
                if not len(sv):
                    continue
                thr = np.quantile(sv, 1 - keep)
                kept.append(g[np.asarray(fn(g), dtype=np.float64) >= thr])
            k_ = pd.concat(kept) if kept else test.iloc[:0]
            if len(k_) < 30:
                out.append({"rule": name, "keep_val": keep, "rows": int(len(k_))})
                continue
            per = np.sign(k_["mu"]) * k_["y_ret"] - 2 * COST
            days = dm.day_summary(dm.absolute_days(k_, k_["mu"].to_numpy(), cost=COST, min_rows=1))
            out.append({"rule": name, "keep_val": keep, "rows": int(len(k_)), "share_rows": len(k_) / len(test),
                        "days": days.get("n_days"), "hit": float(np.mean(np.sign(k_["mu"]) == np.sign(k_["y_ret"]))),
                        "net_per_trade": float(per.mean()), "day_mean": days.get("mean"),
                        "day_lo": days.get("mean_lo"), "day_hi": days.get("mean_hi")})
    return out


def pooled(out: Path, seed: int) -> dict:
    p = out / f"pooled_days_seed_{seed}" / "panel_report.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--old", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fc, fr = fold_rows(a.cand), fold_rows(a.ref)
    paired = fc.join(fr, lsuffix="_cand", rsuffix="_ref", how="inner")
    res = {"folds": {"cand": fc.reset_index().to_dict("records"), "ref": fr.reset_index().to_dict("records")}}
    if a.old is not None and a.old.is_dir():
        res["folds"]["old_ref"] = fold_rows(a.old).reset_index().to_dict("records")
    diff = {}
    for h in HEADS:
        k = f"targets.{h}.skill.rank_ic_by_timestamp"
        d = paired[f"{k}_cand"] - paired[f"{k}_ref"]
        diff[h] = {"per_fold_seed": d.reset_index().to_dict("records"), "mean": float(d.mean()),
                   "folds_up_by_seed": {int(s): int((g > 0).sum()) for s, g in d.groupby(level="seed")},
                   "n_by_seed": {int(s): int(len(g)) for s, g in d.groupby(level="seed")},
                   "seed_mean": {int(s): float(g.mean()) for s, g in d.groupby(level="seed")}}
    res["rank_ic_diff"] = diff
    res["seeds"] = {}
    for s in a.seeds:
        tc, tr = frames(a.cand, "test", s), frames(a.ref, "test", s)
        vc = frames(a.cand, "val", s)
        if not len(tc) or not len(tr):
            continue
        key = ["time", "coin"]
        both = tc[key + ["mu", "y_ret"]].merge(tr[key + ["mu"]], on=key, suffixes=("_c", "_r"))
        r = {"rows": {"cand": int(len(tc)), "ref": int(len(tr)), "matched": int(len(both))},
             "mu_corr_cand_ref": float(np.corrcoef(both["mu_c"], both["mu_r"])[0, 1]),
             "mu_rank_corr_within_ts": float(metrics.rank_ic(both["mu_c"].to_numpy(), both["mu_r"].to_numpy(),
                                                              tsid(both))),
             "pooled_cand": pooled(a.cand, s), "pooled_ref": pooled(a.ref, s)}
        for metric, fn in (("absolute", lambda f: dm.absolute_days(f, f["mu"].to_numpy())),
                           ("neutral", lambda f: dm.neutral_days(f, f["mu"].to_numpy()))):
            r[f"paired_{metric}_gross"] = dm.paired_days(fn(tc), fn(tr), seed=s)
        r["rel_rank_ic"] = {"cand": rel_ic(tc, tc["mu"].to_numpy()), "ref": rel_ic(tr, tr["mu"].to_numpy())}
        r["error_map_cand"] = error_map(tc)
        r["error_map_ref"] = [group_row(tr, "all")]
        r["worst_rows_cand"] = worst_rows(tc)
        r["worst_days_cand"] = worst_days(tc)
        r["bias_cand"], r["bias_ref"] = bias(tc), bias(tr)
        if len(vc):
            r["recenter_cand"] = recenter(vc, tc)
            r["selective_cand"] = selective(vc, tc)
            vr = frames(a.ref, "val", s)
            r["selective_ref"] = selective(vr, tr) if len(vr) else []
        res["seeds"][str(s)] = r
    (a.out / "reading.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res["rank_ic_diff"].items()}, default=float)[:3000])


if __name__ == "__main__":
    main()
