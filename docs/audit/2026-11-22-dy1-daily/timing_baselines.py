"""DY1 look (NOT pre-registered; written after the dy1-vm long books were read as timing): does a trivial timing
rule on the equal-weight basket do what the model's breadth does?

The basket = the equal-weight mean 2-day return of the coins present at a timestamp, run in the same account (2
sub-books, 0.08 % per side, compounded) at a fixed exposure E when a rule is on. Rules (thresholds fixed on the
rows ending before 2023-01-01, the TRAIN era of every fold): basket past k-day move (k = 1, 3, 7, 14) below its
33rd percentile (reversal) or above its 67th (momentum); always on. E = 0.09 (about the model books' exposure).
Only the TEST days of the dy1-vm folds are used (the days present in a model table, read from FOLDS).

    python docs/audit/2026-11-22-dy1-daily/timing_baselines.py ART_VM "FOLD_GLOB"
"""
import glob
import sys

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.account import account_curve, account_stats
from nig_timenet_torch.workflow.battery_run import table_path
from own_returns import own_frame  # noqa: E402  (same folder)

E, CUT = 0.09, pd.Timestamp("2023-01-01")


def main(art, pattern):
    own = own_frame(art)
    days = sorted(set(pd.concat([pd.read_parquet(table_path(f, "test"))["end_time"] for f in glob.glob(pattern)])))
    y = own.pivot_table(index="end_time", columns="coin", values="y_own")
    basket = y.mean(axis=1)                                      # 2-day forward basket return at each end_time
    from nig_timenet_torch.core.schema import LAST_COLUMNS, TS_COL
    from nig_timenet_torch.data_io import npy_dir
    ds = npy_dir.load(art, mmap=True)
    lc = np.asarray(ds["last_candles"])
    one = pd.DataFrame({"end_time": own["end_time"], "coin": own["coin"],
                        "move": np.asarray(ds[f"X_{ds['base_tf']}"])[:, -1, 2]})   # the last bar's close log move
    past = one.pivot_table(index="end_time", columns="coin", values="move").mean(axis=1)  # known at end_time
    del lc, TS_COL, LAST_COLUMNS
    rows = []
    for k in (1, 3, 7, 14):
        mv = past.rolling(k).sum()
        lo, hi = mv[mv.index < CUT].quantile([1 / 3, 2 / 3])
        rows += [(f"reversal {k}d", mv < lo), (f"momentum {k}d", mv > hi)]
    rows.append(("always on", basket.notna()))
    r = basket.reindex(days).to_frame("basket")
    step = pd.Timedelta("1D")
    for name, on in rows:
        w = (on.reindex(days).fillna(False).astype(float) * E).to_frame("basket")
        st = account_stats(account_curve(w, r.fillna(0.0), 0.0008, 2, step), step, float(w.mean().iloc[0]))
        print(f"{name:<14} CAGR {st['cagr'] * 100:+6.1f}% Sharpe {st['sharpe']:5.2f} DD {st['max_dd'] * 100:5.1f}% "
              f"expo {st['exposure']:.3f} years " + " ".join(f"{v * 100:+.0f}%" for v in st["years"].values()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
