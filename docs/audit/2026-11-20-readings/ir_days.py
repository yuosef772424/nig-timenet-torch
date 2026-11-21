"""Why does IR's long book pay per DAY but not per TRADE? Day net against the number of trades of the day.

    python ir_days.py OUT_DIR [OUT_DIR ...]

Same book as ``2026-11-19-arch-indicators/read_arms.py`` (VAL 90th percentile of mu, net = y_ret - 0.0016). Prints per
out dir: day mean and trade mean, the day mean by tercile of trades per day, the share of trades in the busiest third
of days, the correlation of trades per day with the day's equal-weight market move, and the day mean of a book that
caps each day at its 10 highest-mu trades. A study script (rule G6): reads files, no training.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COST = 0.0016


def book(out_dir: Path) -> pd.DataFrame:
    parts = []
    for d in sorted(out_dir.glob("fold_*_seed_*")):
        if not re.match(r"fold_\d+_seed_\d+", d.name):
            continue
        val = pd.read_parquet(d / "battery_table_val.parquet")
        st = pd.read_parquet(d / "battery_table_test.parquet")
        st["mkt"] = st.groupby("end_time")["y_ret"].transform("mean")
        st["day"] = pd.to_datetime(st["end_time"]).dt.floor("D")
        parts.append(st[st["mu"] >= np.quantile(val["mu"], 0.9)])
    return pd.concat(parts, ignore_index=True)


def main(argv=None) -> int:
    for path in (argv or sys.argv[1:]):
        lg = book(Path(path))
        lg["net"] = lg["y_ret"] - COST
        days = lg.groupby("day").agg(net=("net", "mean"), trades=("net", "size"), mkt=("mkt", "mean"))
        terc = pd.qcut(days["trades"].rank(method="first"), 3, labels=["few", "mid", "many"])
        by = days.groupby(terc, observed=True).agg(day_net=("net", "mean"), trades=("trades", "median"),
                                                   days=("net", "size"), mkt=("mkt", "mean"))
        busy = days.loc[terc == "many", "trades"].sum() / days["trades"].sum()
        top10 = lg.sort_values("mu", ascending=False).groupby("day").head(10)
        cap = top10.groupby("day")["net"].mean()
        print(f"\n## {path}\nday mean {days['net'].mean() * 100:+.3f} %, trade mean {lg['net'].mean() * 100:+.3f} %, "
              f"{len(days)} days, {len(lg)} trades; busiest third of days hold {busy:.0%} of the trades")
        print("corr(trades per day, market move of the day)", f"{days['trades'].corr(days['mkt']):+.3f}")
        with pd.option_context("display.float_format", "{:+.5f}".format):
            print(by.to_string())
        print(f"capped at 10 trades a day: day mean {cap.mean() * 100:+.3f} %, "
              f"trade mean {top10['net'].mean() * 100:+.3f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
