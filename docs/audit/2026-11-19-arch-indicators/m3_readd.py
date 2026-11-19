"""M3 amendment 2: the market put back on M's coin selection (V1, V2) against arm A's long book (V0), long only.

    python m3_readd.py RES_ARTIFACT A_OUT_S0 A_OUT_S1 M_OUT_S0 M_OUT_S1

A study script (rule G6): reads the folds' battery tables and the residual artifact's beta channel.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.stats.metrics import block_bootstrap_ci

COST = 0.0016


def beta_table(res_art: str) -> pd.Series:
    d = npy_dir.load(res_art, mmap=True)
    ts = pd.to_datetime(np.asarray(d["last_candles"])[:, LAST_COLUMN_INDEX["timestamp"]].astype(np.int64))
    coin = np.empty(len(ts), dtype=object)
    for b in d["asset_bounds"]:
        coin[int(b["start"]):int(b["end"])] = b["name"]
    beta = np.asarray(d["X_4h"][:, -1, list(d["feature_order"]).index("beta")], dtype=float)
    # battery end_time = prediction time = the last bar's close (open + one bar)
    end = ts + pd.Timedelta(d["base_tf"])
    return pd.Series(beta, index=pd.MultiIndex.from_arrays([coin, end], names=["coin", "end_time"]))


def stage(a_dir: Path, m_dir: Path, fold: str, name: str, beta: pd.Series) -> pd.DataFrame:
    a = pd.read_parquet(a_dir / fold / f"battery_table_{name}.parquet")
    m = pd.read_parquet(m_dir / fold / f"battery_table_{name}.parquet")[["coin", "end_time", "mu"]]
    t = a.merge(m.rename(columns={"mu": "mu_m"}), on=["coin", "end_time"])
    t["beta"] = t.set_index(["coin", "end_time"]).index.map(beta).to_numpy(dtype=float)
    t["beta"] = t["beta"].fillna(1.0)
    t["mkt_fc"] = t.groupby("end_time")["mu"].transform("mean")
    t["score"] = t["mu_m"] + t["beta"] * t["mkt_fc"]
    t["rank_m"] = t.groupby("end_time")["mu_m"].rank(pct=True)
    return t


def book(rows: pd.DataFrame) -> dict:
    net = rows["y_ret"] - COST
    days = net.groupby(pd.to_datetime(rows["end_time"]).dt.floor("D")).mean()
    m, lo, hi = block_bootstrap_ci(days.to_numpy(), np.arange(len(days)), level=0.95, n_boot=2000)
    return {"trades": len(rows), "net/trade %": net.mean() * 100, "day %": m * 100, "lo": lo * 100, "hi": hi * 100,
            "w/o best5 %": days[days < days.quantile(0.95)].mean() * 100, "win days": (days > 0).mean()}


def main(argv=None) -> int:
    res, a0, a1, m0, m1 = (argv or sys.argv[1:])
    beta = beta_table(res)
    out = []
    for seed, a_dir, m_dir in ((0, Path(a0), Path(m0)), (1, Path(a1), Path(m1))):
        parts = {k: [] for k in ("V0", "V1", "V2", "always")}
        for fold in sorted(p.name for p in a_dir.glob("fold_*")):
            va, te = stage(a_dir, m_dir, fold, "val", beta), stage(a_dir, m_dir, fold, "test", beta)
            parts["V0"].append(te[te["mu"] >= np.quantile(va["mu"], 0.9)])
            parts["V1"].append(te[te["score"] >= np.quantile(va["score"], 0.9)])
            lvl = np.quantile(va.groupby("end_time")["mkt_fc"].first(), 0.8)
            parts["V2"].append(te[(te["mkt_fc"] >= lvl) & (te["rank_m"] >= 0.9)])
            parts["always"].append(te)
        for k, v in parts.items():
            out.append({"seed": seed, "variant": k, **book(pd.concat(v))})
    with pd.option_context("display.width", 200, "display.float_format", "{:+.3f}".format):
        print(pd.DataFrame(out).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
