"""Is 'mean day without the best 5 % of days' a fair test? Apply it to benchmarks and to a placebo (random books)."""
import numpy as np, pandas as pd
from load import load, COST
rng = np.random.default_rng(0)

def stats(day):
    day = day.dropna()
    q95, q05 = day.quantile(.95), day.quantile(.05)
    return dict(mean=day.mean(), wo_best5=day[day < q95].mean(), sym_trim=day[(day > q05) & (day < q95)].mean(),
                median=day.median(), top5_share=day[day >= q95].sum() / day.sum() if day.sum() > 0 else np.nan)

def fmt(name, s):
    print(f"  {name:<34} mean {s['mean']*100:+.3f}  w/o best5 {s['wo_best5']*100:+.3f}  sym-trim5 {s['sym_trim']*100:+.3f}"
          f"  median {s['median']*100:+.3f}")

for seed in (0, 1):
    d = load(seed)
    print(f"== seed {seed}")
    lg = d[d.long]
    model = (lg.y_ret - COST).groupby(lg.day).mean()
    fmt("MODEL long book (net)", stats(model))
    fmt("always long, all coins (net)", stats(d.groupby("day").y_ret.mean() - COST))
    btc = d[d.coin == "BTCUSDT"]
    fmt("BTC long every row (net)", stats(btc.groupby("day").y_ret.mean() - COST))
    # placebo: same number of long trades per TIMESTAMP, random coins of that timestamp
    n_ts = lg.groupby("end_time").size()
    grp = d.groupby("end_time").indices
    P = []
    yv = d.y_ret.to_numpy(); dayv = d.day.to_numpy()
    for it in range(200):
        idx = np.concatenate([rng.choice(grp[t], size=min(n, len(grp[t])), replace=False) for t, n in n_ts.items()])
        P.append(stats(pd.Series(yv[idx] - COST).groupby(dayv[idx]).mean()))
    P = pd.DataFrame(P)
    m = stats(model)
    for k in ("mean", "wo_best5", "sym_trim", "median"):
        print(f"  placebo (same timing, random coins) {k:<9} mean {P[k].mean()*100:+.3f}  p95 {P[k].quantile(.95)*100:+.3f}"
              f"  model {m[k]*100:+.3f}  model beats {np.mean(m[k] > P[k])*100:.1f}% of placebos")
