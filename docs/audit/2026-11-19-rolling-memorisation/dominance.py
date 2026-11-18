"""Model-free look (owner 2026-11-19): USDT.D / BTC.D proxies and a BTC + ETH reference instead of the cross-section
mean, on the TRAIN era of the 4h artifact (rows before the default train_end). Every number with a day-block CI."""
import json
import numpy as np, pandas as pd
from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.stats import metrics

ds = npy_dir.load("/content/art1", mmap=True)
TRAIN_END = pd.Timestamp("2026-02-24 04:00", tz="UTC")
lc = np.asarray(ds["last_candles"])
ts = pd.to_datetime(lc[:, TS_COL].astype("int64"), utc=True)
keep = np.flatnonzero(ts < TRAIN_END)
coin = np.empty(len(ts), dtype=object)
for b in ds["asset_bounds"]:
    coin[b["start"]:b["end"]] = b["name"]
X = ds["X_4h"]
last = np.asarray(X[keep, -1, 2], dtype=np.float64)                     # last 4h close move
d1 = np.asarray(X[keep, -6:, 2], dtype=np.float64).sum(axis=1)          # last 24h close move
y = np.asarray(ds["y_close_reg"][keep], dtype=np.float64)              # next 8h close move (label)
f = pd.DataFrame({"t": ts[keep], "coin": coin[keep], "last": last, "d1": d1, "y": y})
g = f.groupby("t")
T = pd.DataFrame({"n": g.size(), "mkt_last": g["last"].mean(), "mkt_d1": g["d1"].mean(), "mkt_y": g["y"].mean(),
                  "disp_d1": g["d1"].std(), "disp_y": g["y"].std()})
for c, k in (("BTCUSDT", "btc"), ("ETHUSDT", "eth")):
    s = f[f["coin"] == c].set_index("t")
    T[f"{k}_last"], T[f"{k}_d1"], T[f"{k}_y"] = s["last"], s["d1"], s["y"]
T = T.dropna()
T = T[T["n"] >= 20]
T["be_d1"] = (T["btc_d1"] + T["eth_d1"]) / 2
T["be_y"] = (T["btc_y"] + T["eth_y"]) / 2
T["btcd_proxy_d1"] = T["btc_d1"] - T["mkt_d1"]            # BTC.D change ~ BTC minus the market
T["btcd_proxy_y"] = T["btc_y"] - T["mkt_y"]
T["usdtd_proxy_d1"] = -(0.75 * T["btc_d1"] + 0.25 * T["eth_d1"])   # USDT.D change ~ minus the cap-weighted market
day = T.index.floor("D").astype("int64") // (86400 * 10**9)
blocks = (day - day.min()) // 5

def corr_ci(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    za, zb = (a - a.mean()) / a.std(), (b - b.mean()) / b.std()
    m, lo, hi = metrics.block_bootstrap_ci(za * zb, blocks.to_numpy(), level=0.95, seed=0)
    return {"corr": round(float(m), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}

out = {"timestamps": int(len(T)), "span": [str(T.index.min()), str(T.index.max())], "median_coins": float(T["n"].median())}
# 1. how well does BTC + ETH stand for the cross-section mean (the shared window's market state)?
out["btc_eth_vs_market_d1"] = corr_ci(T["be_d1"], T["mkt_d1"])
out["btc_eth_vs_market_last"] = corr_ci((T["btc_last"] + T["eth_last"]) / 2, T["mkt_last"])
out["btc_eth_vs_market_future"] = corr_ci(T["be_y"], T["mkt_y"])
# 2. do the proxies predict the market's next 8 h (the absolute book) or BTC-vs-alts (rotation)?
for name in ("mkt_d1", "be_d1", "usdtd_proxy_d1", "btcd_proxy_d1", "disp_d1"):
    out[f"{name} -> mkt_y"] = corr_ci(T[name], T["mkt_y"])
out["btcd_proxy_d1 -> btcd_proxy_y (rotation persists?)"] = corr_ci(T["btcd_proxy_d1"], T["btcd_proxy_y"])
out["btcd_proxy_d1 -> disp_y"] = corr_ci(T["btcd_proxy_d1"], T["disp_y"])
# 3. relative ranking: a per-timestamp reference does not change ranks; the beta x market interaction does matter:
#    does the sign of the reference's last day decide which coins (high or low beta) lead next?
f = f[f["t"].isin(T.index)].copy()
f["rel_d1_mkt"] = f["d1"] - f["t"].map(T["mkt_d1"])
f["rel_d1_be"] = f["d1"] - f["t"].map(T["be_d1"])
f["rel_y"] = f["y"] - f["t"].map(T["mkt_y"])
tid = f["t"].astype("int64").to_numpy()
out["rank_ic rel_d1 (market ref) -> rel_y"] = float(metrics.rank_ic(f["rel_y"].to_numpy(), f["rel_d1_mkt"].to_numpy(), tid))
out["rank_ic rel_d1 (BTC+ETH ref) -> rel_y"] = float(metrics.rank_ic(f["rel_y"].to_numpy(), f["rel_d1_be"].to_numpy(), tid))
# beta of each coin to BTC+ETH (train era), then: rank IC of beta with the next relative move, split by the sign of
# the BTC.D proxy and of the market's last day (does a dominance regime say whether high-beta alts lead?)
fb = f.join(T[["be_y"]], on="t")
beta = fb.groupby("coin").apply(lambda s: np.cov(s["y"], s["be_y"])[0, 1] / np.var(s["be_y"]) if len(s) > 200 else np.nan)
f["beta"] = f["coin"].map(beta)
f = f.dropna(subset=["beta"])
for name, cond in (("BTC.D proxy up (BTC beat the market, 24h)", f["t"].map(T["btcd_proxy_d1"]) > 0),
                   ("BTC.D proxy down", f["t"].map(T["btcd_proxy_d1"]) <= 0),
                   ("market 24h up", f["t"].map(T["mkt_d1"]) > 0), ("market 24h down", f["t"].map(T["mkt_d1"]) <= 0)):
    s = f[cond]
    out[f"rank_ic beta -> rel_y | {name}"] = round(float(metrics.rank_ic(s["rel_y"].to_numpy(), s["beta"].to_numpy(),
                                                                        s["t"].astype("int64").to_numpy())), 4)
print(json.dumps(out, indent=1, default=str))
json.dump(out, open("/content/drive/MyDrive/nig_timenet_runs/memorisation-check-001/dominance_look.json", "w"), indent=1, default=str)
