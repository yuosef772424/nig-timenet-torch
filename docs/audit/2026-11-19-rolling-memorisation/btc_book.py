"""BTC-hedged book against the market-neutral book (pre-registered addendum 4, owner 2026-11-19), from the saved TEST
frames of finished walk-forwards, pooled over the folds per seed. Units: one unit long against one unit short per
timestamp; P&L = r_long - r_short; cost = 2 x cost-per-side on each leg that is traded (full turnover every 4 h for
the alt legs: an upper bound shared by both books). Days = UTC days, mean over the day's timestamps."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.eval import day_metrics as dm

R = Path("/content/drive/MyDrive/nig_timenet_runs")
COST = {"a_flat": (0.0008, 0.0008, True), "b_per_instrument": (0.0008, 0.0004, True),
        "c_btc_held": (0.0008, 0.0004, False)}          # (alt per side, BTC per side, BTC leg traded every timestamp)
TOP = 0.2


def frames(job, seed):
    parts = [pd.read_parquet(d / "panel_frame_test.parquet") for d in sorted((R / job / "out").glob(f"fold_*_seed_{seed}"))
             if (d / "panel_frame_test.parquet").is_file()]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def books(f):
    btc = f[f["coin"] == "BTCUSDT"].set_index("time")[["y_ret", "mu"]].rename(columns={"y_ret": "y_btc", "mu": "mu_btc"})
    f = f[f["coin"] != "BTCUSDT"].join(btc, on="time", how="inner")
    f = f[np.isfinite(f["mu"])]
    r = f.groupby("time")["mu"].rank(method="first", pct=True)
    n = f.groupby("time")["mu"].transform("size")
    top, bot = r > 1 - TOP, r <= TOP
    g = f.assign(top_y=np.where(top, f["y_ret"], np.nan), bot_y=np.where(bot, f["y_ret"], np.nan),
                 s=np.sign(f["mu"] - f["mu_btc"]), rel=f["y_ret"] - f["y_btc"])
    g["pair"] = g["s"] * g["rel"]
    t = g.groupby("time").agg(top=("top_y", "mean"), bot=("bot_y", "mean"), btc=("y_btc", "first"),
                              pair=("pair", "mean"), net_btc=("s", "mean"), n=("y_ret", "size"))
    return t[t["n"] >= 10].dropna(subset=["top", "bot", "btc"])


def daily(series):
    return series.groupby(series.index.floor("D")).mean()


out = {}
jobs = sys.argv[1:] or ["panel-rolling-ref-002", "panel-rolling-std-001", "panel-rolling-ls-001"]
for job in jobs:
    for seed in (0, 1):
        f = frames(job, seed)
        if f.empty:
            continue
        t = books(f)
        res = {"timestamps": int(len(t)), "days": int(t.index.floor("D").nunique()),
               "gross_mean_per_timestamp": {"neutral": float((t["top"] - t["bot"]).mean()),
                                            "btc_hedged": float((t["top"] - t["btc"]).mean()),
                                            "btc_pair_sign": float(t["pair"].mean())}}
        for cname, (c_alt, c_btc, btc_traded) in COST.items():
            neutral = daily(t["top"] - t["bot"] - 2 * c_alt - 2 * c_alt)
            hedged = daily(t["top"] - t["btc"] - 2 * c_alt - (2 * c_btc if btc_traded else 0.0))
            pair = daily(t["pair"] - 2 * c_alt - 2 * c_btc * (t["net_btc"].abs() if not btc_traded else 1.0))
            res[cname] = {"neutral": dm.day_summary(neutral, seed=seed), "btc_hedged": dm.day_summary(hedged, seed=seed),
                          "btc_pair_sign": dm.day_summary(pair, seed=seed),
                          "btc_hedged_minus_neutral": dm.paired_days(hedged, neutral, seed=seed)}
        out[f"{job}_s{seed}"] = res
        line = {k: (round(v["neutral"]["mean"] * 100, 4), round(v["btc_hedged"]["mean"] * 100, 4),
                    round(v["btc_hedged_minus_neutral"]["d_mean"] * 100, 4),
                    round(v["btc_hedged_minus_neutral"]["d_mean_lo"] * 100, 4),
                    round(v["btc_hedged_minus_neutral"]["d_mean_hi"] * 100, 4)) for k, v in res.items() if k in COST}
        print(job, seed, "gross/ts %", {k: round(v * 100, 4) for k, v in res["gross_mean_per_timestamp"].items()})
        print("   net day % (neutral, hedged, diff, lo, hi):", line, flush=True)
(R / "memorisation-check-001").mkdir(exist_ok=True)
name = "btc_book_" + ("_".join(j.replace("panel-rolling-", "") for j in jobs)) + ".json"
(R / "memorisation-check-001" / name).write_text(json.dumps(out, indent=1, default=float))
