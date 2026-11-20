"""Fixed-slot sizing (each selected coin gets 1/S of the sub-book; the rest stays in cash), fee on turnover,
compounded. S = number of coins (exposure = breadth share), and a breadth gate fitted on VAL only."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from load import load, DATA, Q
SIDE = 0.0008

def run(d, sel, slots):
    res = []
    for phase in (0, 4):
        x = d[(d.end_time.dt.hour % 8 == phase)]
        prev, eq, curve = {}, 1.0, []
        for t, g in x.groupby("end_time", sort=True):
            s = g[sel.loc[g.index]]
            w = {c: 1.0 / slots for c in s.coin}
            if sum(w.values()) > 1: w = {c: 1.0 / len(s) for c in s.coin}
            turn = sum(abs(w.get(c, 0) - prev.get(c, 0)) for c in set(w) | set(prev))
            r = float(sum(w[c] * y for c, y in zip(s.coin, s.y_ret))) - SIDE * turn
            eq *= 1 + r; curve.append((t, eq)); prev = w
        res.append(pd.Series(dict(curve)))
    eq = pd.concat(res, axis=1, sort=True).ffill().fillna(1.0).mean(axis=1)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    r = eq.pct_change().dropna()
    yearly = eq.resample("YE").last(); yearly = pd.concat([pd.Series([1.0]), yearly.reset_index(drop=True)]).pct_change().dropna()
    return eq.iloc[-1], eq.iloc[-1] ** (1 / yrs) - 1, r.mean() / r.std() * np.sqrt(6 * 365), (eq / eq.cummax() - 1).min(), yearly.values

def val_breadth_gate(seed):
    """Per fold: on VAL, the breadth (share of coins with mu >= thr) above which the VAL long book earned > cost."""
    gates = {}
    for k in range(7):
        v = pd.read_parquet(DATA / f"out_s{seed}__fold_{k:03d}_seed_{seed}__battery_table_val.parquet")
        thr = float(np.quantile(v["mu"], Q))
        v["long"] = v.mu >= thr
        b = v.groupby("end_time").agg(br=("long", "mean"), r=("y_ret", lambda s: 0))
        lr = v[v.long].groupby("end_time").y_ret.mean()
        b = b.join(lr.rename("lr")).dropna()
        best, bestv = 1.0, -1e9
        for g in np.arange(0.0, 0.8, 0.05):
            m = b[b.br >= g]
            if len(m) < 100: break
            val = ((m.lr - 2 * SIDE) * m.br).sum()   # money of slot sizing above the gate
            if val > bestv: best, bestv = g, val
        gates[k] = best
    return gates

def main():
    for seed in (0, 1):
        d = load(seed)
        br = d.groupby("end_time")["long"].transform("mean")
        gates = val_breadth_gate(seed)
        gate_mask = d.long & (br >= d.fold.map(gates))
        print(f"== seed {seed}  VAL breadth gates per fold: {gates}")
        for name, sel, S in (("model long, slots = all coins (1/101)", d.long, 101),
                             ("model long, slots = 20", d.long, 20),
                             ("model long + VAL breadth gate, slots 101", gate_mask, 101),
                             ("always long, slots = all coins", pd.Series(True, index=d.index), 101),
                             ("BTC hold", d.coin == "BTCUSDT", 1)):
            f, cagr, sh, dd, yr = run(d, sel, S)
            print(f"  {name:<42} x{f:5.2f} CAGR {cagr*100:+6.1f}% Sharpe {sh:5.2f} maxDD {dd*100:6.1f}%  years " + " ".join(f"{v*100:+.0f}%" for v in yr))


if __name__ == "__main__":
    main()
