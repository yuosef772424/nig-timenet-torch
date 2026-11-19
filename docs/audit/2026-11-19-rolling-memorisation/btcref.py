"""Model-free look (owner 2026-11-19): BTC alone as the reference, dispersion over all coins except BTC and ETH."""
import json
import numpy as np, pandas as pd
from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.stats import metrics

ds = npy_dir.load("/content/art1", mmap=True)
lc = np.asarray(ds["last_candles"])
ts = pd.to_datetime(lc[:, TS_COL].astype("int64"), utc=True)
keep = np.flatnonzero(ts < pd.Timestamp("2026-02-24 04:00", tz="UTC"))
coin = np.empty(len(ts), dtype=object)
for b in ds["asset_bounds"]:
    coin[b["start"]:b["end"]] = b["name"]
X = ds["X_4h"]
f = pd.DataFrame({"t": ts[keep], "coin": coin[keep], "last": np.asarray(X[keep, -1, 2], float),
                  "d1": np.asarray(X[keep, -6:, 2], float).sum(axis=1), "y": np.asarray(ds["y_close_reg"][keep], float)})
alts = f[~f["coin"].isin(["BTCUSDT", "ETHUSDT"])]
g, ga = f.groupby("t"), alts.groupby("t")
T = pd.DataFrame({"n": g.size(), "mkt_d1": g["d1"].mean(), "mkt_last": g["last"].mean(), "mkt_y": g["y"].mean(),
                  "disp_all_d1": g["d1"].std(), "disp_alt_d1": ga["d1"].std(), "disp_all_last": g["last"].std(),
                  "disp_alt_last": ga["last"].std(), "disp_all_y": g["y"].std(), "disp_alt_y": ga["y"].std(),
                  "absrel_y": (f["y"] - g["y"].transform("mean")).abs().groupby(f["t"]).mean()})
b = f[f["coin"] == "BTCUSDT"].set_index("t")
T["btc_d1"], T["btc_last"], T["btc_y"] = b["d1"], b["last"], b["y"]
T = T.dropna()
T = T[T["n"] >= 20]
day = T.index.floor("D").astype("int64") // (86400 * 10**9)
blocks = ((day - day.min()) // 5).to_numpy()

def c(a, bb):
    a, bb = np.asarray(a, float), np.asarray(bb, float)
    z = (a - a.mean()) / a.std() * (bb - bb.mean()) / bb.std()
    m, lo, hi = metrics.block_bootstrap_ci(z, blocks, level=0.95, seed=0)
    return [round(float(m), 4), round(float(lo), 4), round(float(hi), 4)]

out = {"timestamps": int(len(T))}
out["BTC vs market mean, 24h move"] = c(T["btc_d1"], T["mkt_d1"])
out["BTC vs market mean, last 4h bar"] = c(T["btc_last"], T["mkt_last"])
out["BTC vs market mean, next 8h"] = c(T["btc_y"], T["mkt_y"])
out["dispersion: alts only vs all coins (24h)"] = c(T["disp_alt_d1"], T["disp_all_d1"])
out["dispersion: alts only vs all coins (last bar)"] = c(T["disp_alt_last"], T["disp_all_last"])
out["BTC 24h -> market next 8h"] = c(T["btc_d1"], T["mkt_y"])
out["market 24h -> market next 8h"] = c(T["mkt_d1"], T["mkt_y"])
out["BTC 24h -> BTC next 8h"] = c(T["btc_d1"], T["btc_y"])
out["dispersion alts 24h -> dispersion next 8h"] = c(T["disp_alt_d1"], T["disp_alt_y"])
out["dispersion all 24h -> dispersion next 8h"] = c(T["disp_all_d1"], T["disp_all_y"])
out["dispersion alts last bar -> mean |relative move| next 8h"] = c(T["disp_alt_last"], T["absrel_y"])
out["|BTC 24h| -> dispersion next 8h"] = c(T["btc_d1"].abs(), T["disp_alt_y"])
# ranking: (own - BTC) and (own - BTC) / dispersion give the same order inside a timestamp as (own - market)
f = f[f["t"].isin(T.index)].copy()
f["rel_y"] = f["y"] - f["t"].map(T["mkt_y"])
tid = f["t"].astype("int64").to_numpy()
for name, s in (("own - market", f["d1"] - f["t"].map(T["mkt_d1"])), ("own - BTC", f["d1"] - f["t"].map(T["btc_d1"])),
                ("(own - BTC) / dispersion alts", (f["d1"] - f["t"].map(T["btc_d1"])) / f["t"].map(T["disp_alt_d1"]))):
    out[f"rank IC {name} -> next relative move"] = round(float(metrics.rank_ic(f["rel_y"].to_numpy(), s.to_numpy(), tid)), 5)
print(json.dumps(out, indent=0))
json.dump(out, open("/content/drive/MyDrive/nig_timenet_runs/memorisation-check-001/btc_ref_look.json", "w"), indent=1)
