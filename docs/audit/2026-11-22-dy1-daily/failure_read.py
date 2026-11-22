"""DY1 failure reading (rule C7, docs/PHILOSOPHY.md section 3) from the saved TEST / VAL tables of one arm and seed.

    python docs/audit/2026-11-22-dy1-daily/failure_read.py LABEL=GLOB [LABEL=GLOB ...] > OUT.txt
GLOB = the fold folders of one arm and seed (battery_table_{val,test}.parquet).

Prints per group: (1) what was learned (rank IC per day of mu / pred_high / pred_low against y_ret / fut_high / fut_low,
next to the trivial volatility baseline atr for high and low, and sigma against |y_ret|); (2) the error map of the
long book (picks = mu >= the fold's VAL q90): excess over the equal-weight basket of the day per group of rows
(sigma tercile, coin volatility tercile by atr, market regime = the basket's past 7-day move tercile, year, fold),
with the day-block CI; (3) the 12 worst long picks; (4) bias: share mu > 0 against share y_ret > 0, per year;
(5) selective use: rules written here before TEST is read, cut on VAL, run through the account (fee 0.08 % per side).
"""
import glob
import re
import sys

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.account import account_section
from nig_timenet_torch.stats.metrics import block_bootstrap_ci
from nig_timenet_torch.workflow.battery_run import _tables

RULES = {"mu q90 (the book)": ("mu", 0.90), "mu q95": ("mu", 0.95), "mu q80": ("mu", 0.80),
         "mu / sigma q90": ("z", 0.90), "mu q90 and sigma below its VAL median": ("mu_lowsig", 0.90)}


def _key(p):
    m = re.search(r"fold_(\d+)", p)
    return int(m.group(1)) if m else 0


def daily_ic(df, a, b):
    ic = df.groupby("end_time").apply(lambda d: d[a].rank().corr(d[b].rank()) if len(d) > 4 else np.nan,
                                      include_groups=False).dropna()
    return ic.mean(), ic.mean() / ic.std() * np.sqrt(len(ic) / 2)


def past_move(df):
    """The basket's mean past 7-day close change at each timestamp (from last_close; no future)."""
    p = df.pivot_table(index="end_time", columns="coin", values="last_close").sort_index()
    return (p / p.shift(7) - 1).mean(axis=1)


def add_cols(val, test):
    out = []
    for d in (val, test):
        d = d.loc[np.isfinite(d["y_ret"]) & np.isfinite(d["mu"])].copy()
        d["z"] = d["mu"] / d["sigma"]
        d["excess"] = d["y_ret"] - d.groupby("end_time")["y_ret"].transform("mean")
        out.append(d)
    return out


def score_rule(val, test, rule):
    col, q = RULES[rule]
    if col == "mu_lowsig":
        thr, smed = np.quantile(val["mu"], q), np.median(val["sigma"])
        return (test["mu"] >= thr) & (test["sigma"] <= smed)
    return test[col] >= np.quantile(val[col], q)


def ci_line(x, days):
    m, lo, hi = block_bootstrap_ci(x.to_numpy(), days, level=0.975, n_boot=1000)
    return f"{m * 100:+.3f}% [{lo * 100:+.3f}, {hi * 100:+.3f}]"


def main():
    for label, pattern in (a.split("=", 1) for a in sys.argv[1:]):
        folders = sorted(glob.glob(pattern), key=_key)
        folds = [add_cols(*_tables(f)) for f in folders]
        test = pd.concat([t.assign(fold=k) for k, (_, t) in enumerate(folds)], ignore_index=True)
        test["long"] = np.concatenate([(t["mu"] >= np.quantile(v["mu"], 0.9)).to_numpy() for v, t in folds])
        print(f"\n=== {label}: {len(folders)} folds, {len(test):,} TEST rows, {test['end_time'].nunique()} days")
        print("## 1 learned (rank IC per day, t ~ with overlapping labels)")
        for a_, b_ in (("mu", "y_ret"), ("pred_high", "fut_high"), ("atr", "fut_high"), ("pred_low", "fut_low"),
                       ("atr", "fut_low")):
            if a_ in test and b_ in test:
                m, t = daily_ic(test, a_, b_)
                print(f"  {a_:>9} vs {b_:<8} IC {m:+.4f} (t~{t:.1f})")
        test["abs_y"] = test["y_ret"].abs()
        m, t = daily_ic(test, "sigma", "abs_y")
        print(f"  {'sigma':>9} vs |y_ret|  IC {m:+.4f} (t~{t:.1f})   atr vs |y_ret| IC {daily_ic(test, 'atr', 'abs_y')[0]:+.4f}")
        regime = past_move(test)
        test["regime"] = pd.qcut(test["end_time"].map(regime), 3, labels=["down", "flat", "up"])
        test["sig_t"] = test.groupby("end_time")["sigma"].transform(lambda s: pd.qcut(s.rank(method="first"), 3,
                                                                                         labels=False))
        test["vol_t"] = test.groupby("end_time")["atr"].transform(lambda s: pd.qcut(s.rank(method="first"), 3,
                                                                                       labels=False))
        test["year"] = test["end_time"].dt.year
        days = test["end_time"].to_numpy().astype("datetime64[D]").astype(np.int64)
        lg = test[test["long"]]
        ld = days[test["long"].to_numpy()]
        print(f"## 2 error map of the long book (excess over the day's basket, 2-day, CI 97.5 %); all picks "
              f"{len(lg):,}: {ci_line(lg['excess'], ld)}; hit (excess > 0) {(lg['excess'] > 0).mean():.3f}")
        for g in ("sig_t", "vol_t", "regime", "year", "fold"):
            for v, sub in lg.groupby(g, observed=True):
                dd = ld[(lg[g] == v).to_numpy()]
                print(f"  {g:<6} {str(v):<5} rows {len(sub):>6} days {len(np.unique(dd)):>5} excess "
                      f"{ci_line(sub['excess'], dd)} raw y {sub['y_ret'].mean() * 100:+.3f}% hit "
                      f"{(sub['excess'] > 0).mean():.3f}")
        print("## 3 the 12 worst long picks")
        cols = ["end_time", "coin", "mu", "sigma", "y_ret", "excess"]
        print(lg.nsmallest(12, "y_ret")[cols].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
        print("## 4 bias: share mu > 0 / share y_ret > 0 per year")
        for y, sub in test.groupby("year"):
            print(f"  {y}: {(sub['mu'] > 0).mean():.3f} / {(sub['y_ret'] > 0).mean():.3f}  mean mu {sub['mu'].mean():+.4f}"
                  f" mean y {sub['y_ret'].mean():+.4f}")
        print("## 5 selective use (rules fixed before TEST; cut on VAL; account at 0.08 % per side, 2 sub-books)")
        for rule in RULES:
            parts = []
            for v, t in folds:
                t2 = t.copy()
                t2["mu"] = score_rule(v, t, rule).astype(float)    # 1 = picked; the account keeps mu >= VAL q of 1
                parts.append((pd.DataFrame({"mu": [0.0] * 9 + [1.0]}), t2))
            st = account_section(parts, 0.0008, n_books=2, n_placebo=20)["stats"]
            m, p = st["model"], st["placebo"]
            print(f"  {rule:<40} CAGR {m['cagr'] * 100:+6.1f}% Sharpe {m['sharpe']:5.2f} DD {m['max_dd'] * 100:5.0f}% "
                  f"expo {m['exposure']:.3f} | placebo p95 {p['cagr_p95'] * 100:+.1f}% | always long "
                  f"{st['always long']['cagr'] * 100:+.1f}%")


if __name__ == "__main__":
    main()
