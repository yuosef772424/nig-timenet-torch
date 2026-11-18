"""Model-free look (owner 2026-11-19): does the growth of USDT supply (DefiLlama, daily, free) predict the market's
next 8 h / 24 h beyond the market's own 24 h move? TRAIN era of the 4h artifact; no lookahead: a row on day D uses
the supply of day D-1 (the DefiLlama daily point is stamped 00:00 UTC; we never use the same day's point)."""
import json
import numpy as np, pandas as pd
from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.stats import metrics

raw = json.load(open("/tmp/usdt.json"))["tokens"]
s = pd.Series({pd.Timestamp(int(r["date"]), unit="s", tz="UTC"): r["circulating"]["peggedUSD"] for r in raw}).sort_index()
s.to_frame("usdt_supply").to_csv("/content/drive/MyDrive/crypto_model/external_usdt_supply_defillama.csv")
s = s.asfreq("D").ffill()
g1, g7 = np.log(s).diff(1), np.log(s).diff(7)
ds = npy_dir.load("/content/art1", mmap=True)
lc = np.asarray(ds["last_candles"])
ts = pd.to_datetime(lc[:, TS_COL].astype("int64"), utc=True)
keep = np.flatnonzero(ts < pd.Timestamp("2026-02-24 04:00", tz="UTC"))
f = pd.DataFrame({"t": ts[keep], "d1": np.asarray(ds["X_4h"][keep, -6:, 2], float).sum(axis=1),
                  "y": np.asarray(ds["y_close_reg"][keep], float)})
T = f.groupby("t").agg(n=("y", "size"), mkt_d1=("d1", "mean"), mkt_y=("y", "mean"))
T = T[T["n"] >= 20]
T["mkt_y24"] = T["mkt_y"].rolling(3).sum().shift(-2)      # three consecutive 8h-ahead labels ~ next 24h (overlapping)
prev_day = (T.index.floor("D") - pd.Timedelta("1D"))
T["g1"], T["g7"] = g1.reindex(prev_day).to_numpy(), g7.reindex(prev_day).to_numpy()
T = T.dropna()
day = T.index.floor("D").astype("int64") // (86400 * 10**9)
blocks = ((day - day.min()) // 5).to_numpy()

def c(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    z = (a - a.mean()) / a.std() * (b - b.mean()) / b.std()
    m, lo, hi = metrics.block_bootstrap_ci(z, blocks, level=0.95, seed=0)
    return [round(float(m), 4), round(float(lo), 4), round(float(hi), 4)]

def resid(y, x):
    x1 = np.column_stack([np.ones(len(x)), x])
    return y - x1 @ np.linalg.lstsq(x1, y, rcond=None)[0]

out = {"timestamps": int(len(T)), "span": [str(T.index.min()), str(T.index.max())],
       "usdt_days": int(len(s)), "usdt_first": str(s.index.min()), "usdt_last": str(s.index.max())}
for k in ("g1", "g7"):
    out[f"{k} (USDT supply growth, prev day) -> market next 8h"] = c(T[k], T["mkt_y"])
    out[f"{k} -> market next 24h"] = c(T[k], T["mkt_y24"])
    # partial: beyond the market's own 24h move
    out[f"{k} -> market next 8h, partial on market 24h"] = c(resid(T[k].to_numpy(), T["mkt_d1"].to_numpy()),
                                                          resid(T["mkt_y"].to_numpy(), T["mkt_d1"].to_numpy()))
out["market 24h -> market next 8h (reference)"] = c(T["mkt_d1"], T["mkt_y"])
out["corr g7 with market 24h (is it the same information?)"] = c(T["g7"], T["mkt_d1"])
print(json.dumps(out, indent=0))
json.dump(out, open("/content/drive/MyDrive/nig_timenet_runs/memorisation-check-001/usdt_supply_look.json", "w"), indent=1)
