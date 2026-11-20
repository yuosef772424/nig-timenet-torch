import warnings; warnings.filterwarnings("ignore")
import numpy as np
import a4_sizing as A
from load import load
rng = np.random.default_rng(7)
for seed in (0, 1):
    d = load(seed)
    cnt = d.groupby("end_time")["long"].transform("sum")
    print(f"== seed {seed}")
    for i in range(3):
        rk = d.assign(u=rng.random(len(d))).groupby("end_time")["u"].rank(method="first")
        f, c, sh, dd, yr = A.run(d, rk <= cnt, 101)
        print(f"  placebo {i} (same count/timestamp, random coins, slots 101): CAGR {c*100:+.1f}% Sharpe {sh:.2f} maxDD {dd*100:.1f}%")
    for side in (0.0002, 0.0005, 0.0008, 0.0010, 0.0015):
        A.SIDE = side
        f, c, sh, dd, yr = A.run(d, d.long, 101)
        print(f"  fee {side*100:.2f}%/side: model long slots 101 CAGR {c*100:+.1f}% Sharpe {sh:.2f}")
    A.SIDE = 0.0008
