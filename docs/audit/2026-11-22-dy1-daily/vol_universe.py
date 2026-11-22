"""DY1 amendment 1 (owner 2026-11-22): a universe of coins with similar volatility, chosen on the TRAIN era only.

Per coin of the daily artifact: the sd of its daily close log moves over the rows that END before 2023-01-01 (the
first TEST window starts 2023-01-03, so no TEST day chooses a coin); coins with < 180 such days are left out (their
volatility cannot be measured without the TEST era). Kept: the coins between the 25th and the 95th percentile of that
sd (drops the low movers: BTC, ETH, BNB, XMR, TRX, LTC, the BTCDOM index, and the most extreme tail).

    python docs/audit/2026-11-22-dy1-daily/vol_universe.py ARTIFACT OUT_TXT
"""
import sys

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.data_io import npy_dir

CUT, MIN_DAYS, LO, HI = "2023-01-01", 180, 0.25, 0.95


def main(src: str, out: str) -> None:
    ds = npy_dir.load(src, mmap=True)
    x, t = ds[f"X_{ds['base_tf']}"], ds["last_candles"][:, TS_COL].astype(np.int64)
    cut, vols = pd.Timestamp(CUT).value, {}
    for b in ds["asset_bounds"]:
        m = t[b["start"]:b["end"]] < cut
        if m.sum() >= MIN_DAYS:
            vols[b["name"]] = float(np.std(x[b["start"]:b["end"]][m][:, -1, 2]))
    v = pd.Series(vols)
    lo, hi = v.quantile([LO, HI])
    keep = v[(v >= lo) & (v <= hi)].sort_index()
    with open(out, "w") as f:
        f.write(",".join(keep.index))
    print(f"{len(v)} measurable coins, kept {len(keep)} with daily sd in [{lo:.4f}, {hi:.4f}]; dropped low: "
          f"{', '.join(v[v < lo].sort_values().index)}; dropped high: {', '.join(v[v > hi].index)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
