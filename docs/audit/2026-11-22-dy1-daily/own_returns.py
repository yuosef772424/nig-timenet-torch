"""DY1 amendment 1: copies of an M3 arm's fold tables with y_ret replaced by the coin's OWN 2-day return.

An M3 table's y_ret is the residual return (coin minus beta x market). The accept rule of the M3 arms reads the
picks on the coins' own returns: this joins every row (coin, end_time) to the own-return artifact (``art_vm``:
future_close / last_close - 1 of the row whose last bar closes at end_time) and writes the tables again.

    python docs/audit/2026-11-22-dy1-daily/own_returns.py ART_VM OUT_DIR FOLD_FOLDER [FOLD_FOLDER ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import LAST_COLUMNS, TS_COL
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.workflow.battery_run import table_path


def own_frame(art) -> pd.DataFrame:
    ds = npy_dir.load(art, mmap=True)
    lc = np.asarray(ds["last_candles"])
    coins = np.empty(len(lc), dtype=object)
    for b in ds["asset_bounds"]:
        coins[b["start"]:b["end"]] = b["name"]
    end = pd.to_datetime(lc[:, TS_COL].astype(np.int64) + pd.Timedelta(ds["base_tf"].upper()).value)
    y = lc[:, LAST_COLUMNS.index("future_close")] / lc[:, LAST_COLUMNS.index("last_close")] - 1.0
    return pd.DataFrame({"coin": coins, "end_time": end, "y_own": y})


def main(art, out, folders):
    own = own_frame(art)
    for f in folders:
        dst = Path(out) / Path(f).name
        dst.mkdir(parents=True, exist_ok=True)
        for st in ("val", "test"):
            t = pd.read_parquet(table_path(f, st))
            m = t.merge(own, on=["coin", "end_time"], how="left", validate="many_to_one")
            miss = int(m["y_own"].isna().sum())
            if miss > 0.001 * len(m):
                raise ValueError(f"{f} {st}: {miss} of {len(m)} rows have no own return (end_time convention?)")
            m["y_ret"] = m.pop("y_own")
            m.to_parquet(table_path(dst, st))
        print(f"{dst}: own returns joined", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
