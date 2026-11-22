"""Is the account's profit market TIMING skill, or a market move the book happened to sit in?

Usage (Colab): . /content/nig_env.sh; python timing.py <runs root>
Per run x seed (fold folders as in scan.py), on the TEST rows of all folds:
  1. timing vs its own placebos: the basket (equal weight, all coins) held at the book's breadth, against the same
     breadth series circularly SHIFTED in time by a random offset (keeps its distribution and autocorrelation, breaks
     its link with the market; 50 draws) and against a CONSTANT exposure equal to its mean;
  2. timing IC: Spearman of breadth(t) with the basket's next-horizon return, per calendar year;
  3. concentration: the share of the log growth made by the best 5 % of days, the CAGR with the best 5 % of days
     removed and with the best AND worst 5 % removed, positive months / folds; the same for always long and BTC;
  4. the book split into exposure x basket (the same exposure on all coins) and the rest (coin choice), gross of cost.
"""
import glob
import sys

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.account import account_curve, account_frame, account_stats, book_weights, row_step
from nig_timenet_torch.stats.metrics import spearman
from nig_timenet_torch.workflow.battery_run import _tables

COST, N_SHIFT, BOOKS = 0.0008, 50, 2
RUNS = [("rolling-dw1-r-001/out_s0", ""), ("rolling-dw1-r-001/out_s1", ""),
        ("panel-rolling-std-001/out", "seed_0"), ("panel-rolling-std-001/out", "seed_1"),
        ("panel-rolling-ls-001/out", "seed_0"), ("panel-rolling-ls-001/out", "seed_1"),
        ("snapshot-rolling-base-001/out", "seed_0"), ("snapshot-rolling-base-001/out", "seed_1")]


def daily(eq):
    return eq.resample("D").last().dropna().pct_change().dropna()


def trimmed_cagr(d, lo, hi):
    """CAGR of a daily return series with the days outside [lo, hi] quantiles set to 0 (held in cash)."""
    q_lo, q_hi = d.quantile(lo), d.quantile(hi)
    kept = d.where((d >= q_lo) & (d <= q_hi), 0.0)
    years = (d.index[-1] - d.index[0]).days / 365.25
    return float((1 + kept).prod() ** (1 / years) - 1)


def concentration(eq):
    d = daily(eq)
    lg = np.log1p(d)
    top = lg[lg >= lg.quantile(0.95)].sum() / lg.sum() if lg.sum() > 0 else float("nan")
    m = eq.resample("ME").last().pct_change().dropna()
    return {"top5_share": float(top), "wo_best5": trimmed_cagr(d, 0.0, 0.95), "sym5": trimmed_cagr(d, 0.05, 0.95),
            "pos_months": float((m > 0).mean())}


def fold_cagr(eq, frame):
    out = []
    for _, g in frame.groupby("fold"):
        e = eq.loc[g["end_time"].min():g["end_time"].max()]
        if len(e) > 10:
            yrs = (e.index[-1] - e.index[0]).days / 365.25
            out.append((e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1)
    return out


def run(root, rel, seed, rng):
    tables = sorted(glob.glob(f"{root}/{rel}/fold_*{seed}/battery_table_test.parquet"))
    folders = [f.rsplit("/", 1)[0] for f in tables]
    frame = account_frame([_tables(f) for f in folders])
    step = row_step(frame["end_time"])
    r = frame.pivot(index="end_time", columns="coin", values="y_ret").sort_index()
    present = r.notna()
    picks = frame.pivot(index="end_time", columns="coin", values="long").reindex_like(r).fillna(False).astype(bool)
    basket = r.mean(axis=1).to_frame("basket")
    breadth = (picks.sum(axis=1) / present.sum(axis=1)).clip(0, 1)

    def acc(expo, cost=COST):
        return account_curve(expo.to_frame("basket"), basket, cost, BOOKS, step)

    timing = acc(breadth)
    s_t = account_stats(timing, step)
    const = account_stats(acc(breadth * 0 + breadth.mean()), step)
    shifts = []
    n = len(breadth)
    for _ in range(N_SHIFT):
        k = int(rng.integers(n // 10, n - n // 10))
        shifts.append(account_stats(acc(pd.Series(np.roll(breadth.to_numpy(), k), index=breadth.index)), step)["cagr"])
    shifts = np.array(shifts)
    yr = breadth.index.year
    ic = {str(y): spearman(breadth[yr == y].to_numpy(), basket["basket"][yr == y].to_numpy()) for y in sorted(set(yr))}
    slots = r.shape[1]
    w = book_weights(picks, slots)
    model = account_curve(w, r, COST, BOOKS, step)
    gross_model = account_stats(account_curve(w, r, 0.0, BOOKS, step), step)["cagr"]
    gross_expo = account_stats(acc(w.sum(axis=1), 0.0), step)["cagr"]
    always = account_curve(book_weights(present, slots), r, COST, BOOKS, step)
    hold = present.copy()
    hold.loc[:, hold.columns != "BTCUSDT"] = False
    btc = account_curve(book_weights(hold, 1), r, COST, BOOKS, step)
    print(f"\n== {rel} {seed or ''}  ({len(folders)} folds, {len(frame):,} rows)")
    print(f"  timing (basket x breadth): CAGR {s_t['cagr']*100:+.1f}%  Sharpe {s_t['sharpe']:.2f}  | constant exposure "
          f"{breadth.mean():.2f}: {const['cagr']*100:+.1f}%  | time-shifted breadth: mean {shifts.mean()*100:+.1f}% "
          f"p95 {np.quantile(shifts, .95)*100:+.1f}%  "
          f"timing beats {np.mean(s_t['cagr'] > shifts)*100:.0f}% of {N_SHIFT}")
    ic_line = "  ".join(f"{y} {v:+.3f}" for y, v in ic.items())
    print(f"  timing IC (breadth vs next basket move) per year: {ic_line}")
    for name, eq in (("timing", timing), ("model book", model), ("always long", always), ("BTC hold", btc)):
        c, f = concentration(eq), fold_cagr(eq, frame)
        print(f"  {name:<12} best 5% days hold {c['top5_share']*100:4.0f}% of the log growth | CAGR w/o best 5% days "
              f"{c['wo_best5']*100:+6.1f}%  w/o best+worst 5% {c['sym5']*100:+6.1f}%  | months > 0 "
              f"{c['pos_months']*100:3.0f}% | folds > 0 {sum(x > 0 for x in f)}/{len(f)}")
    print(f"  book gross of cost: CAGR {gross_model*100:+.1f}% = same exposure on the basket {gross_expo*100:+.1f}% "
          f"+ coin choice {(gross_model - gross_expo)*100:+.1f} points")


if __name__ == "__main__":
    root, rng = sys.argv[1], np.random.default_rng(0)
    for rel, seed in RUNS:
        run(root, rel, seed, rng)
