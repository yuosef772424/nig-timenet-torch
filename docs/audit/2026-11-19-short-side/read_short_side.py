"""Read the short side of `rolling-single-001` (7 expanding folds x 2 seeds) from the saved battery tables.

Usage: python docs/audit/2026-11-19-short-side/read_short_side.py <folder with test_f{k}_s{s}.parquet [val_...]>

The battery tables come from Drive `nig_timenet_runs/rolling-single-001/fold_00k_seed_s/out/battery_table_*.parquet`.
VAL tables above 10 MB could not be fetched through the Drive connector, so the thresholds of fold k come from its own
VAL table when present (folds 0 and 1) and else from the TEST table of fold k-1 of the same seed: an earlier period,
never the rows being judged (with expanding folds it is the latest part of fold k's own VAL era).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.eval import battery_scorecard as sc
from nig_timenet_torch.eval.battery_lib import round_trip

COST = 0.0008
RT = round_trip(COST)


def load(folder: Path, seed: int):
    out = []
    for k in range(7):
        test = pd.read_parquet(folder / f"test_f{k}_s{seed}.parquet")
        vp = folder / f"val_f{k}_s{seed}.parquet"
        val = pd.read_parquet(vp) if vp.is_file() else pd.read_parquet(folder / f"test_f{k - 1}_s{seed}.parquet")
        out.append((k, val, test))
    return out


def pooled_masks(folds):
    names, masks, tests = None, {}, []
    for k, val, test in folds:
        m = sc.selections(val, test)
        names = list(m)
        for n in names:
            masks.setdefault(n, []).append(m[n])
        tests.append(test.assign(fold=k))
    return pd.concat(tests, ignore_index=True), {n: np.concatenate(masks[n]) for n in names}


def table_asym(test, masks):
    side = np.where(test["p_dir"].to_numpy() > 0.5, 1.0, -1.0)
    y, mu, sig = (test[c].to_numpy() for c in ("y_ret", "mu", "sigma"))
    print(f"  {'selection':<32}{'n':>7}{'win%':>7}{'avgW%':>8}{'avgL%':>8}{'L/W':>6}{'log%':>9}{'exp%':>9}"
          f"{'pred exp%':>10}{'net%':>9}{'sd%':>7}{'worst1%':>8}")
    for name in ("all rows", "top 5% |mu|", "top 10% |mu|", "top 20% |mu|", "top 20% confidence",
                 "low 20% sigma", "top 20% |mu|/sigma"):
        for label, sg in (("LONG", 1.0), ("SHORT", -1.0)):
            idx = np.where(masks[name] & (side == sg))[0]
            if len(idx) < sc.MIN_TRADES:
                continue
            line, _ = sc._asym_line(f"{label} {name}", sg, y[idx], sig[idx], mu[idx], RT)
            print(line)


def table_days(test, masks, n_rows):
    side = np.where(test["p_dir"].to_numpy() > 0.5, 1.0, -1.0)
    y = test["y_ret"].to_numpy()
    days = pd.to_datetime(test["end_time"]).to_numpy().astype("datetime64[D]")
    level = 1.0 - 0.05 / n_rows
    print(f"  CI level {level:.4f} (Bonferroni over {n_rows} rows)")
    print(f"  {'selection':<30}{'side':<6}{'n':>8}{'days':>6}{'win%':>7}{'avg%':>9}{'win d%':>8}{'mean d% [CI]':>28}"
          f"{'worst d%':>10}{'CVaR%':>8}{'tail%':>7}{'maxDD%':>8}{'w/o best%':>10}{'busy d%':>9}")
    for name, mask in masks.items():
        for label, sg in (("LONG", 1.0), ("SHORT", -1.0), ("BOTH", 0.0)):
            idx = np.where(mask & (side == sg))[0] if sg else np.where(mask)[0]
            if len(idx) < sc.MIN_TRADES:
                continue
            pnl = side[idx] * y[idx] - RT
            p = sc.day_profile(pnl, days[idx], level)
            print(f"  {name:<30}{label:<6}{len(idx):>8,}{p['days']:>6}{(pnl > 0).mean() * 100:>6.1f}%"
                  f"{pnl.mean() * 100:>+8.3f}%{p['win_days'] * 100:>7.1f}%"
                  f"{p['mean_day'] * 100:>+9.3f} [{p['ci_lo'] * 100:+.3f}, {p['ci_hi'] * 100:+.3f}]"
                  f"{p['worst_day'] * 100:>+10.2f}{p['cvar'] * 100:>+8.2f}{p['tail_loss_share'] * 100:>6.0f}%"
                  f"{p['max_dd'] * 100:>8.1f}{p['mean_wo_best'] * 100:>+10.3f}{p['mean_busy'] * 100:>+9.3f}")


def per_fold(folds):
    print(f"  {'fold':<5}{'test from':<12}{'P(short)':>9}{'P(y<0)':>8}  SHORT top10%: {'n':>6}{'win%':>6}{'log%':>8}"
          f"{'exp%':>8}{'net%':>8}  LONG top10%: {'n':>6}{'log%':>8}{'exp%':>8}{'net%':>8}")
    for k, val, test in folds:
        m = sc.selections(val, test)["top 10% |mu|"]
        side = np.where(test["p_dir"].to_numpy() > 0.5, 1.0, -1.0)
        y = test["y_ret"].to_numpy()
        cells = []
        for sg in (-1.0, 1.0):
            idx = np.where(m & (side == sg))[0]
            g, lp = sg * y[idx], sg * np.log1p(y[idx])
            cells.append((len(idx), (g - RT > 0).mean(), lp.mean(), (g - lp).mean(), (g - RT).mean()))
        s, lg = cells
        print(f"  {k:<5}{str(test['end_time'].min().date()):<12}{(side < 0).mean():>9.3f}{(y < 0).mean():>8.3f}"
              f"  {'':>14}{s[0]:>6}{s[1] * 100:>6.1f}{s[2] * 100:>+8.3f}{s[3] * 100:>+8.3f}{s[4] * 100:>+8.3f}"
              f"  {'':>13}{lg[0]:>6}{lg[2] * 100:>+8.3f}{lg[3] * 100:>+8.3f}{lg[4] * 100:>+8.3f}")


def worst_shorts(test, masks, n=15):
    side = np.where(test["p_dir"].to_numpy() > 0.5, 1.0, -1.0)
    d = test[masks["top 10% |mu|"] & (side < 0)].copy()
    d["net"] = -d["y_ret"] - RT
    d["vol_rank"] = d["sigma"].rank(pct=True)
    cols = ["fold", "coin", "end_time", "mu", "sigma", "p_dir", "y_ret", "net", "vol_rank"]
    print(d.nsmallest(n, "net")[cols].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    loss = d["net"][d["net"] < 0]
    worst = loss.nsmallest(max(1, len(d) // 100)).sum() / loss.sum()
    print(f"  short top10% losing trades {len(loss)}; worst 1% of trades hold {worst:.0%} of all losses; "
          f"trades below -10%: {(d['net'] < -0.10).sum()} ({(d['net'] < -0.10).mean():.2%})")
    by = d.groupby("coin")["net"].agg(["count", "sum"]).sort_values("sum").head(8)
    print("  coins that hold the short loss (sum of net):\n" + by.to_string(float_format=lambda v: f"{v:+.3f}"))


def market_days(test, masks):
    """Short top10% day P&L against the market's equal-weight day move."""
    side = np.where(test["p_dir"].to_numpy() > 0.5, 1.0, -1.0)
    days = pd.to_datetime(test["end_time"]).dt.floor("D")
    mkt = test.groupby(days)["y_ret"].mean()
    m = masks["top 10% |mu|"] & (side < 0)
    sday = (-test["y_ret"][m] - RT).groupby(days[m]).mean()
    j = pd.concat({"short": sday, "mkt": mkt}, axis=1, sort=True).dropna()
    q = pd.qcut(j["mkt"], 5, labels=["mkt q1 (down)", "q2", "q3", "q4", "q5 (up)"])
    agg = j.groupby(q, observed=True)["short"].agg(["count", "mean"])
    print(agg.to_string(float_format=lambda v: f"{v * 100:+.3f}%"))
    print(f"  corr(short day, market day) {j['short'].corr(j['mkt']):+.2f}")


def main():
    folder = Path(sys.argv[1])
    for seed in (0, 1):
        folds = load(folder, seed)
        test, masks = pooled_masks(folds)
        print(f"\n######## seed {seed}: {len(test):,} TEST rows, {test['end_time'].dt.floor('D').nunique()} days, "
              f"{test['coin'].nunique()} coins; skew(y) {test['y_ret'].skew():+.2f}, skew(log y) "
              f"{np.log1p(test['y_ret']).skew():+.2f}")
        print("\n== payoff split (pooled)")
        table_asym(test, masks)
        print("\n== per fold")
        per_fold(folds)
        print("\n== days (pooled)")
        table_days(test, masks, len(masks) * 3)
        print("\n== worst short trades (top 10% |mu|)")
        worst_shorts(test, masks)
        print("\n== short top10% by market day")
        market_days(test, masks)


if __name__ == "__main__":
    main()
