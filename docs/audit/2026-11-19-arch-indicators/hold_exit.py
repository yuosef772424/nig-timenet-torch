"""Amendment 3 of PREREGISTERED_M3.md: enter in the top 10 % of mu, exit only below a lower VAL quantile (hysteresis).

    python hold_exit.py --arm NAME=OUT_DIR [--arm ...] [--exits 0.9,0.75,0.7]

OUT_DIR holds ``fold_*_seed_*/battery_table_{val,test}.parquet`` (rows = consecutive horizon periods per coin). A study
script (rule G6).
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.stats.metrics import block_bootstrap_ci

SIDE = 0.0008
ENTRY_Q = 0.90


def simulate(te: pd.DataFrame, enter: float, exit_: float) -> tuple[pd.DataFrame, int]:
    """Per-period position rows (end_time, net) of the hysteresis book on one fold's TEST table; and entries."""
    rows, entries = [], 0
    step = te.groupby("coin")["end_time"].apply(lambda s: s.sort_values().diff().median()).median()
    for _, g in te.sort_values("end_time").groupby("coin"):
        t, mu, y = g["end_time"].to_numpy(), g["mu"].to_numpy(), g["y_ret"].to_numpy()
        held = False
        for i in range(len(g)):
            gap = i > 0 and (t[i] - t[i - 1]) != step
            if held and (gap or mu[i] < exit_):
                rows[-1][1] -= SIDE                      # exit cost on the last held period
                held = False
            if not held and mu[i] >= enter:
                held, entries = True, entries + 1
                rows.append([t[i], y[i] - SIDE])         # entry cost on the first period
            elif held:
                rows.append([t[i], y[i]])
        if held:
            rows[-1][1] -= SIDE
    return pd.DataFrame(rows, columns=["end_time", "net"]), entries


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--exits", default="0.9,0.75,0.7")
    a = ap.parse_args(argv)
    exits = [float(v) for v in a.exits.split(",")]
    out = []
    for spec in a.arm:
        name, path = spec.split("=", 1)
        per = {}
        for d in sorted(Path(path).glob("fold_*_seed_*")):
            seed = int(re.match(r"fold_\d+_seed_(\d+)", d.name).group(1))
            va = pd.read_parquet(d / "battery_table_val.parquet")
            te = pd.read_parquet(d / "battery_table_test.parquet")
            for q in exits:
                pos, n = simulate(te, float(np.quantile(va["mu"], ENTRY_Q)), float(np.quantile(va["mu"], q)))
                per.setdefault((seed, q), []).append((pos, n))
        for (seed, q), parts in sorted(per.items()):
            pos = pd.concat([p for p, _ in parts])
            n = sum(k for _, k in parts)
            book = pos.groupby("end_time")["net"].mean()
            days = book.groupby(pd.to_datetime(book.index).floor("D")).mean()
            m, lo, hi = block_bootstrap_ci(days.to_numpy(), np.arange(len(days)), level=0.95, n_boot=2000)
            out.append({"arm": name, "seed": seed, "exit_q": q, "entries": n, "periods/entry": len(pos) / max(n, 1),
                        "day %": m * 100, "lo": lo * 100, "hi": hi * 100,
                        "w/o best5 %": days[days < days.quantile(0.95)].mean() * 100,
                        "win days": (days > 0).mean(), "days": len(days)})
    with pd.option_context("display.width", 200, "display.float_format", "{:+.3f}".format):
        print(pd.DataFrame(out).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
