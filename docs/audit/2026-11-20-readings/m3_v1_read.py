"""Sceptic reading of V1 (amendment 1 of PREREGISTERED_M3_FULL.md): is the market-put-back book coin choice or timing?

    python m3_v1_read.py RES_ARTIFACT A_OUT_S0 A_OUT_S1 M_OUT_S0 M_OUT_S1

V1 as in ``2026-11-19-arch-indicators/m3_readd.py`` (score = M's mu + beta x mean of A's mu at the timestamp; buy rows
with score >= the VAL 90th percentile; y_ret = A's table = the coin's own simple return). Prints per seed: day mean per
fold and year, minus the equal-weight market, minus BTC with two costs, beta to BTC, days traded, trades per day, and
the timing-only book (all rows of the timestamps V1 trades). A study script (rule G6): reads files, no training.
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
    end = ts + pd.Timedelta(d["base_tf"])
    return pd.Series(beta, index=pd.MultiIndex.from_arrays([coin, end], names=["coin", "end_time"]))


def stage(a_dir: Path, m_dir: Path, fold: str, name: str, beta: pd.Series) -> pd.DataFrame:
    a = pd.read_parquet(a_dir / fold / f"battery_table_{name}.parquet")
    m = pd.read_parquet(m_dir / fold / f"battery_table_{name}.parquet")[["coin", "end_time", "mu"]]
    t = a.merge(m.rename(columns={"mu": "mu_m"}), on=["coin", "end_time"])
    t["beta"] = t.set_index(["coin", "end_time"]).index.map(beta).to_numpy(dtype=float)
    t["beta"] = t["beta"].fillna(1.0)
    t["score"] = t["mu_m"] + t["beta"] * t.groupby("end_time")["mu"].transform("mean")
    t["mkt"] = t.groupby("end_time")["y_ret"].transform("mean")
    btc = t.loc[t["coin"] == "BTCUSDT"].drop_duplicates("end_time").set_index("end_time")["y_ret"]
    t["btc"] = t["end_time"].map(btc)
    t["day"] = pd.to_datetime(t["end_time"]).dt.floor("D")
    t["fold"] = fold
    return t


def day_stats(v: pd.Series) -> str:
    v = v.dropna()
    m, lo, hi = block_bootstrap_ci(v.to_numpy(), np.arange(len(v)), level=0.95, n_boot=2000)
    cut = v[v < v.quantile(0.95)].mean()
    return f"{m * 100:+.3f} [{lo * 100:+.3f}, {hi * 100:+.3f}] w/o best5 {cut * 100:+.3f} ({len(v)} days)"


def main(argv=None) -> int:
    res, a0, a1, m0, m1 = (argv or sys.argv[1:])
    beta = beta_table(res)
    for seed, a_dir, m_dir in ((0, Path(a0), Path(m0)), (1, Path(a1), Path(m1))):
        parts, alls = [], []
        for fold in sorted(p.name for p in a_dir.glob("fold_*")):
            va, te = stage(a_dir, m_dir, fold, "val", beta), stage(a_dir, m_dir, fold, "test", beta)
            parts.append(te[te["score"] >= np.quantile(va["score"], 0.9)])
            alls.append(te)
        lg, al = pd.concat(parts), pd.concat(alls)
        net = lg["y_ret"] - COST
        print(f"\n## seed {seed}: V1 trades {len(lg)}, days with a trade {lg['day'].nunique()} of "
              f"{al['day'].nunique()}, trades per traded day median {lg.groupby('day').size().median():.0f}")
        print("V1 own net            ", day_stats(net.groupby(lg["day"]).mean()))
        print("V1 minus market (1 c) ", day_stats((net - lg["mkt"]).groupby(lg["day"]).mean()))
        ok = lg["btc"].notna()
        print("V1 minus BTC (2 c)    ", day_stats((net - lg["btc"] - COST)[ok].groupby(lg.loc[ok, "day"]).mean()))
        b = np.polyfit(lg.loc[ok, "btc"], lg.loc[ok, "y_ret"], 1)[0]
        print(f"beta to BTC {b:.2f}; per-trade net {net.mean() * 100:+.3f} %")
        times = set(lg["end_time"])
        tim = al[al["end_time"].isin(times)]
        print("timing only (1 c)     ", day_stats((tim["y_ret"] - COST).groupby(tim["day"]).mean()))
        print("per fold day mean     ", " ".join(f"{f[5:8]}:{v * 100:+.3f}" for f, v in
                                                net.groupby([lg["fold"], lg["day"]]).mean().groupby(level=0).mean()
                                                .items()))
        yr = pd.to_datetime(lg["day"]).dt.year
        print("per year day mean     ", " ".join(f"{y}:{v * 100:+.3f}" for y, v in
                                                net.groupby([yr, lg["day"]]).mean().groupby(level=0).mean().items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
