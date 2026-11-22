"""DY1 reading: the pre-registered account of each arm and seed, plus the secondary rows of PREREGISTERED.md.

    python docs/audit/2026-11-22-dy1-daily/read_dy1.py OUT_JSON LABEL=GLOB [LABEL=GLOB ...]
GLOB names the fold folders of ONE arm and seed (each with battery_table_{val,test}.parquet), e.g.
``S0=/home/user/dy1/runs/dy1-single-001/out/fold_*_seed_0``; the folders are read in fold order.

Per group: the ACCOUNT section (eval/account.py, 2 sub-books, 0.08 % per side; the accept rule reads its model /
always long / placebo rows), the rank IC per TEST day, the long-short book (long the VAL q90 picks, short the VAL q10
picks, fixed slots of 1 / coins each side, the same fee), the CAGR over the TEST days present (the split leaves the
first window + embargo of every TEST window empty, see the results note) and a day-block bootstrap CI of the mean
period return of the account (blocks of 7 days, level 1 - 0.05 / 2).
"""
import glob
import json
import re
import sys

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.account import account_curve, account_section, book_weights, row_step
from nig_timenet_torch.stats.metrics import block_bootstrap_ci
from nig_timenet_torch.workflow.battery_run import _tables

COST, BOOKS, LEVEL = 0.0008, 2, 1 - 0.05 / 2


def _fold_key(p: str) -> int:
    m = re.search(r"fold_(\d+)", p)
    return int(m.group(1)) if m else 0


def rank_ic_days(test: pd.DataFrame) -> np.ndarray:
    g = test.groupby("end_time")
    ic = g.apply(lambda d: d["mu"].rank().corr(d["y_ret"].rank()) if len(d) > 4 else np.nan, include_groups=False)
    return ic.dropna().to_numpy()


def long_short(folds) -> tuple[pd.Series, pd.Timedelta]:
    parts = []
    for val, test in folds:
        hi, lo = np.quantile(val["mu"], 0.9), np.quantile(val["mu"], 0.1)
        t = test.loc[np.isfinite(test["y_ret"]) & np.isfinite(test["mu"])].copy()
        t["side"] = np.where(t["mu"] >= hi, 1.0, np.where(t["mu"] <= lo, -1.0, 0.0))
        parts.append(t)
    f = pd.concat(parts, ignore_index=True)
    r = f.pivot(index="end_time", columns="coin", values="y_ret").sort_index()
    side = f.pivot(index="end_time", columns="coin", values="side").reindex_like(r).fillna(0.0)
    slots = r.shape[1]
    w = side / slots
    step = row_step(f["end_time"])
    return account_curve(w, r, COST, BOOKS, step), step


def model_curve(folds) -> tuple[pd.Series, pd.Timedelta]:
    from nig_timenet_torch.eval.account import _books, account_frame
    frame = account_frame(folds)
    r, _present, picks, slots, _ = _books(frame, None, "BTCUSDT")
    step = row_step(frame["end_time"])
    return account_curve(book_weights(picks, slots), r, COST, BOOKS, step), step


def present_cagr(eq: pd.Series) -> float:
    return float(eq.iloc[-1] ** (365.25 / len(eq)) - 1.0)


def main() -> None:
    out, groups = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
    res = {}
    for label, pattern in groups:
        folders = sorted(glob.glob(pattern), key=_fold_key)
        folds = [_tables(f) for f in folders]
        sec = account_section(folds, COST, n_books=BOOKS)
        st = sec["stats"]
        test = pd.concat([t for _, t in folds])
        ic = rank_ic_days(test)
        eq, _ = model_curve(folds)
        per = eq.pct_change().dropna()
        days = per.index.to_numpy().astype("datetime64[D]").astype(np.int64)
        ci = block_bootstrap_ci(per.to_numpy(), days // 7, level=LEVEL, n_boot=2000)
        ls, _ = long_short(folds)
        ls_per = ls.pct_change().dropna()
        ls_ci = block_bootstrap_ci(ls_per.to_numpy(), ls_per.index.to_numpy().astype("datetime64[D]").astype(
            np.int64) // 7, level=LEVEL, n_boot=2000)
        res[label] = {"folders": folders, "rows": sec["rows"], "note": sec["note"], "stats": st,
                      "ic_mean": float(ic.mean()), "ic_t": float(ic.mean() / ic.std() * np.sqrt(len(ic) / 2)),
                      "test_days": int(len(eq)), "model_cagr_present": present_cagr(eq),
                      "model_period_ci": ci, "ls_cagr_present": present_cagr(ls), "ls_final": float(ls.iloc[-1]),
                      "ls_max_dd": float((ls / ls.cummax() - 1).min()), "ls_period_ci": ls_ci}
        m, p, al = st["model"], st["placebo"], st["always long"]
        ok = m["cagr"] > 0 and m["cagr"] > al["cagr"] and m["cagr"] > p["cagr_p95"]
        print(f"{label:<6} folds {len(folders)} IC {res[label]['ic_mean']:+.4f} (t~{res[label]['ic_t']:.1f}) | "
              f"CAGR {m['cagr'] * 100:+.1f}% (present days {res[label]['model_cagr_present'] * 100:+.1f}%) Sharpe "
              f"{m['sharpe']:.2f} DD {m['max_dd'] * 100:.0f}% expo {m['exposure']:.2f} | always long "
              f"{al['cagr'] * 100:+.1f}% placebo p95 {p['cagr_p95'] * 100:+.1f}% BTC "
              f"{st.get('buy and hold', {}).get('cagr', float('nan')) * 100:+.1f}% | accept {ok} | period CI "
              f"[{ci[1] * 1e4:+.2f}, {ci[2] * 1e4:+.2f}] bp | long-short CAGR {res[label]['ls_cagr_present'] * 100:+.1f}% "
              f"DD {res[label]['ls_max_dd'] * 100:.0f}% CI [{ls_ci[1] * 1e4:+.2f}, {ls_ci[2] * 1e4:+.2f}] bp", flush=True)
        for row in sec["rows"]:
            print("   " + row.strip())
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
