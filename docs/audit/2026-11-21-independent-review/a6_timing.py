"""The model's breadth (share of coins with mu >= VAL q90) used as the EXPOSURE of one instrument: BTC, or the
equal-weight basket. Fee 0.08 %/side on exposure changes. Two 8 h sub-books, compounded."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from load import load
SIDE = 0.0008

def run(ts, expo, ret):
    curves = []
    for phase in (0, 4):
        idx = ts[ts.hour % 8 == phase]
        e = expo.loc[idx].clip(0, 1); r = ret.loc[idx]
        turn = e.diff().abs().fillna(e.iloc[0])
        curves.append((1 + e * r - SIDE * turn).cumprod())
    eq = pd.concat(curves, axis=1, sort=True).ffill().fillna(1).mean(axis=1)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    rr = eq.pct_change().dropna()
    y = eq.resample("YE").last(); y = pd.concat([pd.Series([1.0]), y.reset_index(drop=True)]).pct_change().dropna()
    return f"CAGR {(eq.iloc[-1]**(1/yrs)-1)*100:+6.1f}% Sharpe {rr.mean()/rr.std()*np.sqrt(6*365):5.2f} maxDD {(eq/eq.cummax()-1).min()*100:6.1f}% avg expo {expo.loc[ts].mean():.2f} years " + " ".join(f"{v*100:+.0f}%" for v in y)

for seed in (0, 1):
    d = load(seed)
    g = d.groupby("end_time")
    br = g["long"].mean()
    ew = g["y_ret"].mean()
    btc = d[d.coin == "BTCUSDT"].set_index("end_time").y_ret.reindex(br.index).fillna(0)
    ts = br.index
    print(f"== seed {seed}")
    for name, e, r in (("basket, exposure = breadth", br, ew), ("basket, exposure = min(1, 3 x breadth)", (3*br).clip(0,1), ew),
                       ("BTC, exposure = breadth", br, btc), ("BTC, exposure = min(1, 3 x breadth)", (3*br).clip(0,1), btc),
                       ("BTC, always 1", br*0+1, btc), ("basket, always 1", br*0+1, ew)):
        print(f"  {name:<40} {run(ts, e, r)}")
