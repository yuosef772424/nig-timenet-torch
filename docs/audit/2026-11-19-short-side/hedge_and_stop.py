"""Top 10% |mu|: market-hedged trades and a stop at k x sigma on the realised high (usage: <tables folder>)."""
import sys

import numpy as np

sys.path.insert(0, "docs/audit/2026-11-19-short-side")
from pathlib import Path

from read_short_side import RT, load

from nig_timenet_torch.eval import battery_scorecard as sc

folder = Path(sys.argv[1])
for seed in (0, 1):
    out = {}
    for k, val, test in load(folder, seed):
        m = sc.selections(val, test)["top 10% |mu|"]
        mk = test.groupby("end_time")["y_ret"].transform("mean").to_numpy()
        side = np.where(test.p_dir.to_numpy() > .5, 1., -1.)
        y, sig, fh = test.y_ret.to_numpy(), test.sigma.to_numpy(), test.fut_high.to_numpy()
        i = np.where(m & (side < 0))[0]
        out.setdefault("short plain", []).append(-y[i] - RT)
        out.setdefault("short hedged (+mkt)", []).append(-(y[i] - mk[i]) - 2 * RT)
        for c in (1.0, 2.0, 3.0):
            stop = c * sig[i]
            hit = fh[i] >= stop
            out.setdefault(f"short stop {c:.0f}x sigma", []).append(np.where(hit, -stop, -y[i]) - RT)
        j = np.where(m & (side > 0))[0]
        out.setdefault("long plain", []).append(y[j] - RT)
        out.setdefault("long hedged (-mkt)", []).append((y[j] - mk[j]) - 2 * RT)
    print(f"### seed {seed} (top 10% |mu|, pooled 7 folds)")
    for nm, v in out.items():
        v = np.concatenate(v); print(f"  {nm:<22} n={len(v):>7,} win={np.mean(v>0):.3f} net={v.mean()*100:+.3f}%  worst1%={np.sort(v)[:len(v)//100].mean()*100:+.2f}%")
