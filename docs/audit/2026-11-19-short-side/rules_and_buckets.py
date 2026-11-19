"""Mean-corrected side rule and sigma / confidence quintiles, pooled over the 7 folds (usage: <tables folder>)."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "docs/audit/2026-11-19-short-side")
from pathlib import Path

from read_short_side import RT, load

from nig_timenet_torch.eval import battery_scorecard as sc

folder = Path(sys.argv[1])
for seed in (0, 1):
    folds = load(folder, seed)
    rows = {}
    bk = {}
    for k, val, test in folds:
        t = test[test.dir_true != 0]
        mu, sig, y = t.mu.to_numpy(), t.sigma.to_numpy(), t.y_ret.to_numpy()
        m = np.log1p(mu); drag = 0.5*np.log1p(sig/(1+mu))**2
        side = np.where(t.p_dir.to_numpy() > .5, 1., -1.)
        for nm, sg, r in (("LONG mean-rule", 1., m+drag > RT), ("SHORT mean-rule", -1., -m-drag > RT),
                          ("SHORT naive mu<-cost", -1., -m > RT), ("LONG naive mu>cost", 1., m > RT)):
            rows.setdefault(nm, []).append(sg*y[r]-RT)
        e = sc._bucket_edges(val.sigma.to_numpy()); b = np.searchsorted(e, sig, side="right")
        ec = sc._bucket_edges(np.abs(val.p_dir.to_numpy()-.5)); bc = np.searchsorted(ec, np.abs(t.p_dir.to_numpy()-.5), side="right")
        for i in range(5):
            for nm, bb in (("sigma", b), ("conf", bc)):
                mm = bb == i
                bk.setdefault((nm, i), []).append(pd.DataFrame({"g": side[mm]*y[mm], "s": side[mm], "y": y[mm]}))
    print(f"### seed {seed}")
    for nm, v in rows.items():
        v = np.concatenate(v); print(f"  {nm:<24} n={len(v):>7,} win={np.mean(v>0):.3f} net={v.mean()*100:+.3f}%")
    for (nm, i), parts in sorted(bk.items()):
        d = pd.concat(parts); L = d[d.s > 0]; Sh = d[d.s < 0]
        print(f"  {nm:<6} q{i+1} n={len(d):>7,} hit={np.mean(d.g>0):.3f} mean|y|={d.y.abs().mean()*100:.2f}% "
              f"LONG net={(L.y-RT).mean()*100:+.3f}% SHORT net={(-Sh.y-RT).mean()*100:+.3f}% "
              f"SHORT exp={(np.log1p(Sh.y)-Sh.y).mean()*100:+.3f}% LONG exp={(L.y-np.log1p(L.y)).mean()*100:+.3f}%")
