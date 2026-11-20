"""Real money view: two 8 h sub-books (phases 0/8/16 and 4/12/20), each holds half the capital, equal weight on the
selected coins (cash otherwise). Fees = 0.08 % per side on the actual weight TURNOVER (a coin kept from one 8 h period
to the next pays nothing). Compounded. Compared with: per-trade fee (what the battery charges), the timing placebo,
always long, BTC."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from load import load
SIDE = 0.0008
rng = np.random.default_rng(1)

def subbook(d, sel_col, fee_mode):
    out = {}
    for phase in (0, 4):
        x = d[d.end_time.dt.hour % 8 == phase]
        prev = {}
        rets = []
        for t, g in x.groupby("end_time", sort=True):
            s = g[g[sel_col]]
            w = {c: 1.0 / len(s) for c in s.coin} if len(s) else {}
            if fee_mode == "turnover":
                turn = sum(abs(w.get(c, 0) - prev.get(c, 0)) for c in set(w) | set(prev))
            else:  # every trade a full round trip
                turn = 2.0 * sum(w.values())
            r = float((s.y_ret * (1.0 / len(s))).sum()) if len(s) else 0.0
            # drift weights after the period (approximately: keep target weights, cost on target change only)
            rets.append((t, r - SIDE * turn))
            prev = w
        out[phase] = pd.Series(dict(rets))
    # half capital each: per 4 h step, portfolio return ~ mean of the two sub-books' 8 h returns / spread over time
    a, b = out[0], out[4]
    return a, b

def summarize(name, a, b):
    # each sub-book compounds on its own 8 h grid, half the capital each
    wa, wb = (1 + a).cumprod(), (1 + b).cumprod()
    years = (a.index.max() - a.index.min()).days / 365.25
    final = 0.5 * wa.iloc[-1] + 0.5 * wb.iloc[-1]
    eq = pd.concat([wa, wb], axis=1).ffill().fillna(1.0).mean(axis=1)
    dd = (eq / eq.cummax() - 1).min()
    r = pd.concat([a, b]).sort_index()
    sharpe = r.mean() / r.std() * np.sqrt(3 * 365) if r.std() > 0 else np.nan
    yr = eq.resample("YE").last().pct_change().fillna(eq.resample("YE").last().iloc[0] - 1)
    print(f"  {name:<44} x{final:6.2f}  CAGR {(final ** (1 / years) - 1) * 100:+7.1f}%  Sharpe {sharpe:5.2f}  maxDD {dd * 100:6.1f}%"
          f"  years " + " ".join(f"{v * 100:+.0f}%" for v in yr.values))

for seed in (0, 1):
    d = load(seed)
    d["all"] = True
    d["btc"] = d.coin == "BTCUSDT"
    # placebo: same count per timestamp, random coins
    cnt = d.groupby("end_time")["long"].transform("sum")
    rk = d.assign(u=rng.random(len(d))).groupby("end_time")["u"].rank(method="first")
    d["placebo"] = rk <= cnt
    # timing only: equal-weight ALL coins at timestamps where the model has any long
    d["timing_all"] = cnt > 0
    print(f"== seed {seed}")
    for name, col, fm in (("MODEL long book, fee per trade (battery)", "long", "trade"),
                          ("MODEL long book, fee on turnover", "long", "turnover"),
                          ("placebo same timing random coins, turnover", "placebo", "turnover"),
                          ("timing only: all coins when model>0, turnover", "timing_all", "turnover"),
                          ("always long all coins, turnover", "all", "turnover"),
                          ("BTC buy and hold", "btc", "turnover")):
        summarize(name, *subbook(d, col, fm))
