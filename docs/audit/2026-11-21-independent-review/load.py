"""Shared loader: the 7 TEST folds of one seed of rolling-dw1-r-001 (arm R), each with its VAL 90th pct of mu."""
from pathlib import Path
import numpy as np, pandas as pd
DATA = Path("/home/user/workspace/data_r")
COST = 2 * 0.0008
Q = 0.90

def load(seed: int) -> pd.DataFrame:
    parts = []
    for k in range(7):
        t = pd.read_parquet(DATA / f"out_s{seed}__fold_{k:03d}_seed_{seed}__battery_table_test.parquet")
        v = pd.read_parquet(DATA / f"out_s{seed}__fold_{k:03d}_seed_{seed}__battery_table_val.parquet",
                            columns=["mu", "sigma", "p_dir"])
        t["fold"] = k
        t["thr"] = float(np.quantile(v["mu"], Q))
        t["thr_snr"] = float(np.quantile(v["mu"] / v["sigma"], Q))
        t["long"] = t["mu"] >= t["thr"]
        t["day"] = t["end_time"].dt.floor("D")
        parts.append(t)
    d = pd.concat(parts, ignore_index=True)
    d = d[np.isfinite(d["y_ret"]) & np.isfinite(d["mu"])]
    return d
