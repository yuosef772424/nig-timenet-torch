"""Per-coin concentration of P&L and losses (owner question 2026-11-19: a cap per coin), on the pooled TEST frames."""
import json
from pathlib import Path
import numpy as np, pandas as pd
R = Path("/content/drive/MyDrive/nig_timenet_runs")
COST = 0.0008
out = {}
def frames(job, seed):
    return pd.concat([pd.read_parquet(d / "panel_frame_test.parquet").assign(fold=int(d.name.split("_")[1]))
                      for d in sorted((R / job / "out").glob(f"fold_*_seed_{seed}"))
                      if (d / "panel_frame_test.parquet").is_file()], ignore_index=True)
for job in ("panel-rolling-ref-002", "panel-rolling-std-001"):
    for seed in (0, 1):
        f = frames(job, seed)
        f["pnl"] = np.sign(f["mu"]) * f["y_ret"] - 2 * COST
        # realised volatility of a coin = sd of its label moves over the test era; groups by quintile
        vol = f.groupby("coin")["y_ret"].std()
        f["vol_q"] = f["coin"].map(pd.qcut(vol.rank(method="first"), 5, labels=False))
        by_coin = f.groupby("coin")["pnl"].agg(["sum", "count", "mean"])
        tot = by_coin["sum"].sum()
        losses = by_coin["sum"][by_coin["sum"] < 0].sort_values()
        worst = f.nsmallest(int(len(f) * 0.01), "pnl")
        res = {"rows": len(f), "coins": int(f["coin"].nunique()), "total_pnl_sum": float(tot),
               "max_coin_row_share": float(by_coin["count"].max() / len(f)),
               "top10_loss_coins_share_of_all_losses": float(losses.head(10).sum() / losses.sum()) if len(losses) else None,
               "worst1pct_rows_top10_coin_share": float(worst["coin"].value_counts().head(10).sum() / len(worst)),
               "worst1pct_rows_by_vol_quintile": worst["vol_q"].value_counts(normalize=True).sort_index().round(3).to_dict(),
               "by_vol_quintile": f.groupby("vol_q").agg(rows=("pnl", "size"), hit=("pnl", lambda p: float((p + 2*COST > 0).mean())),
                                                         gross=("pnl", lambda p: float(p.mean() + 2*COST)), net=("pnl", "mean")).round(5).to_dict("index"),
               "drop_top_vol_quintile_net": float(f.loc[f["vol_q"] < 4, "pnl"].mean()),
               "all_net": float(f["pnl"].mean())}
        # a per-coin cap on the day book: weight of one coin in a day's equal-weight book, max over days
        day = f["time"].dt.floor("D")
        w = f.groupby([day, "coin"]).size() / f.groupby(day).size().reindex(f.groupby([day, "coin"]).size().index.get_level_values(0)).values
        res["max_coin_weight_in_a_day_book"] = float(w.max())
        res["median_coin_weight_in_a_day_book"] = float(w.median())
        out[f"{job}_s{seed}"] = res
        print(job, seed, json.dumps({k: v for k, v in res.items() if k != "by_vol_quintile"}, default=float))
        print("  by vol quintile", res["by_vol_quintile"])
ls = []
for p in sorted((R / "panel-rolling-ls-001/out").glob("fold_*_seed_*/fold_row.json")):
    r = json.loads(p.read_text())
    ref = json.loads((R / "panel-rolling-ref-002/out" / p.parent.name / "fold_row.json").read_text())
    k = "targets.close.skill.rank_ic_by_timestamp"
    ls.append({"fold": r["fold"], "seed": r["seed"], "best_epoch": r["best_epoch"], "ref_best_epoch": ref["best_epoch"],
               "close_ic": round(r["metrics"][k], 4), "ref_close_ic": round(ref["metrics"][k], 4)})
print("ls vs ref", ls)
out["ls_vs_ref"] = ls
(R / "memorisation-check-001/coins.json").write_text(json.dumps(out, indent=1, default=float))
