"""Short top 10% |mu|: day P&L by the number of shorts that day (usage: <tables folder>)."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "docs/audit/2026-11-19-short-side")
from pathlib import Path

from read_short_side import RT, load, pooled_masks

for seed in (0, 1):
    test, masks = pooled_masks(load(Path(sys.argv[1]), seed))
    side = np.where(test.p_dir > .5, 1., -1.)
    m = masks["top 10% |mu|"] & (side < 0)
    d = pd.DataFrame({"day": test.end_time.dt.floor("D")[m], "net": -test.y_ret[m] - RT})
    g = d.groupby("day")["net"].agg(["count", "mean", "sum"])
    q = pd.qcut(g["count"].rank(method="first"), 4, labels=["fewest", "q2", "q3", "most"])
    print(f"seed {seed}: short top10% days by number of shorts that day")
    print(g.groupby(q, observed=True).agg(days=("count", "size"), trades=("count", "sum"), mean_day=("mean", "mean")).to_string(float_format=lambda v: f"{v:+.4f}"))
    print(f"  share of trades on the 'most' quarter of days: {g['count'][q == 'most'].sum() / g['count'].sum():.2f}")
