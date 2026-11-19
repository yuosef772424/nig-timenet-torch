# ==== nig_backtest on the SINGLE-COIN model (4h bars, horizon 8h), traded on 15m candles ====
# One Colab cell. The model's outputs are the run's battery tables (battery_table_{val,test}.parquet): mu is a decoded
# RETURN over the horizon (kind "return"), sigma is in return units, p_dir = p(up), y_ret = the realised return.
# A finished run folder gives val + test; a walk-forward folder (rolling-*/out with fold_NNN_seed_S) is pooled over
# its folds (test of every fold = one out-of-sample series).
import os, sys, glob, re, subprocess, base64
from concurrent.futures import ThreadPoolExecutor

RUNS = "/content/drive/MyDrive/nig_timenet_runs"
CACHE = "/content/drive/MyDrive/nig_backtest_cache"
RUN_FOLDER = None            # a run folder or a walk-forward out folder; None = the newest one found under RUNS
SKIP_WORDS = ("panel",)      # folders whose path holds one of these words are not single-coin runs
SEED = 0                     # walk-forward folders only: which seed's folds to pool
OUT = "/content/bt_single_15m"
INTERVAL = "15m"
HORIZON = "8h"               # 2 bars of the 4h artifact
MAX_COINS = None             # 20 for a quick try
REPO_DIR = "/content/nig-timenet-torch"
BRANCH = "claude/legacy-tests-normalization-3ufedz"     # holds nig_backtest.app (the web page)

from google.colab import drive
drive.mount("/content/drive")
tok = open("/content/drive/MyDrive/crypto/secrets/github_token.txt").read().strip()
auth = base64.b64encode(f"x-access-token:{tok}".encode()).decode()
env = {**os.environ, "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
       "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {auth}"}
URL = "https://github.com/yuosef772424/nig-timenet-torch.git"
if not os.path.isdir(f"{REPO_DIR}/.git"):
    subprocess.run(["git", "init", "-q", REPO_DIR], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "remote", "add", "origin", URL], check=True)
subprocess.run(["git", "-C", REPO_DIR, "fetch", "-q", "--depth", "1", "origin", BRANCH], check=True, env=env)
subprocess.run(["git", "-C", REPO_DIR, "checkout", "-q", "-f", "-B", BRANCH, "FETCH_HEAD"], check=True)
print(subprocess.run(["git", "-C", REPO_DIR, "log", "--oneline", "-1"], capture_output=True, text=True).stdout)
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "gradio"], check=True)
sys.path.insert(0, f"{REPO_DIR}/src")

import numpy as np, pandas as pd
import nig_backtest as bt
import nig_backtest.app as app
from nig_backtest.bridges import nig_timenet_torch as bridge     # no torch import: it reads the battery tables only
from nig_backtest.prices import vision_klines
from IPython.display import display, Image

# 1) which run: a folder with battery tables, or a walk-forward folder with fold_NNN_seed_S inside
def roots_under(runs):
    hits = glob.glob(f"{runs}/**/battery_table_test.parquet", recursive=True)
    roots = {(os.path.dirname(os.path.dirname(h)) if re.fullmatch(r"fold_\d+_seed_\d+", os.path.basename(os.path.dirname(h)))
              else os.path.dirname(h)) for h in hits}
    roots = [r for r in roots if not any(w in r.lower() for w in SKIP_WORDS)]
    return sorted(roots, key=lambda r: os.path.getmtime(r))

if RUN_FOLDER is None:
    found = roots_under(RUNS)
    assert found, f"no battery_table_test.parquet under {RUNS} (was the battery switched off in the run?)"
    print("runs found (oldest -> newest):", *found[-8:], sep="\n  ")
    RUN_FOLDER = found[-1]
print("using:", RUN_FOLDER)

def load_signals(folder, seed):
    if os.path.isfile(f"{folder}/battery_table_test.parquet"):
        return bridge.signals_from_folder(folder), "one run"
    folds = sorted(glob.glob(f"{folder}/fold_*_seed_{seed}"))
    assert folds, f"{folder} holds neither battery tables nor fold_*_seed_{seed} folders"
    parts = []
    for i, f in enumerate(folds):
        try:
            s = bridge.signals_from_folder(f)
        except FileNotFoundError:
            continue
        s["x_fold"] = float(i)
        parts.append(s)
    s = pd.concat(parts, ignore_index=True)
    val = s[s["stage"] == "val"].drop_duplicates(["coin", "time"], keep="last")   # expanding folds re-use the same val bars
    test = s[s["stage"] == "test"]
    return pd.concat([val, test], ignore_index=True), f"walk-forward, {len(parts)} folds, seed {seed}"

signals, how = load_signals(RUN_FOLDER, SEED)
signals = signals.dropna(subset=["score"])
if MAX_COINS:
    signals = signals[signals["coin"].isin(sorted(signals["coin"].unique())[:MAX_COINS])]
coins = sorted(signals["coin"].unique())
t0, t1 = pd.to_datetime(signals["time"]).min(), pd.to_datetime(signals["time"]).max()
print(f"{how}: {len(signals):,} signals, {len(coins)} coins, {t0} -> {t1}", signals.groupby("stage").size().to_dict())
print("mu (decoded return): mean %+.5f, sd %.5f | sigma median %.5f" % (
    signals["score"].mean(), signals["score"].std(), signals["sigma"].median() if "sigma" in signals else np.nan))

# 2) 15m candles, in parallel, cached on Drive; a cached file that does not cover [start, end) is fetched again
start, end = (t0 - pd.Timedelta(days=3)).floor("D"), (t1 + pd.Timedelta(days=2)).ceil("D")
have_end = min(end, pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")) - pd.Timedelta(days=3)
cache_dir = f"{CACHE}/{INTERVAL}"; os.makedirs(cache_dir, exist_ok=True)
def covered(path):
    try:
        ix = pd.read_parquet(path, columns=["close"]).index
        return len(ix) > 0 and ix.min() <= start + pd.Timedelta(days=1) and ix.max() >= have_end
    except Exception:
        return False
def fetch(c):
    path = f"{cache_dir}/{c}.parquet"
    if os.path.exists(path) and covered(path): return c, "cache"
    try:
        f = vision_klines(c, INTERVAL, start, end)
        if len(f): f.to_parquet(path)
        return c, len(f)
    except Exception as e:
        return c, f"error {type(e).__name__}"
with ThreadPoolExecutor(8) as ex:
    got = list(ex.map(fetch, coins))
missing = [c for c, n in got if n == 0 or str(n).startswith("error")]
print(f"{INTERVAL} candles: {len(coins)-len(missing)} ready, missing: {missing[:20]}")
px = bt.ArchivePrices(INTERVAL, start=start, end=end, cache_dir=CACHE)

# 3) the 24h return known at each signal's time, from the same candles (the battery table has no ret_1d)
def trailing_return(signals, window="1D"):
    out = np.full(len(signals), np.nan)
    bar = pd.Timedelta(INTERVAL)
    for c, grp in signals.groupby("coin", sort=False):
        path = f"{cache_dir}/{c}.parquet"
        if not os.path.exists(path): continue
        b = pd.read_parquet(path, columns=["close"])["close"]
        known = b.index.to_numpy() + bar.to_timedelta64()           # a candle's close is known at open + bar
        t = pd.to_datetime(grp["time"]).to_numpy()
        now = np.searchsorted(known, t, side="right") - 1
        past = np.searchsorted(known, t - pd.Timedelta(window).to_timedelta64(), side="right") - 1
        ok = (now >= 0) & (past >= 0)
        v = b.to_numpy()
        r = np.full(len(grp), np.nan)
        r[ok] = v[now[ok]] / v[past[ok]] - 1.0
        out[signals.index.get_indexer(grp.index)] = r
    return out
signals = signals.reset_index(drop=True)
signals["x_ret_1d"] = trailing_return(signals)
sig = bt.SignalSet(frame=signals, kind="return", horizon=HORIZON)     # absolute kind: exits in `move` units are allowed

# 4) a temporary rule: no short on a coin that rose more than 10% in the last day
def no_squeeze_short(rows, p):
    if "x_ret_1d" not in rows: return np.ones(len(rows), bool)
    r = rows["x_ret_1d"].fillna(0).to_numpy()
    return ~((rows["dir"].to_numpy() < 0) & (r > float(p.get("max_ret_1d", 0.10))))
if "no_squeeze_short" not in bt.ENTRY_FILTERS.names():
    bt.ENTRY_FILTERS.register("no_squeeze_short", no_squeeze_short, needs=())

base = bt.strategy_from_dict({"name": "base", "entry": {"top_share": 0.10},
    "exit": {"hold": "horizon", "atr_timeframe": "4h"},
    "costs": {"mode": "fixed", "fee_per_side": 0.0008}, "book": {"per_coin": 1, "risk_per_trade": 0.02}})
steps = [("tp 1.5 atr", {"exit.take_profit": 1.5}),
         ("stop 2 atr", {"exit.stop_loss": 2.0}),
         ("3 per side per time", {"entry.per_time_top_k": 3}),
         ("no squeeze shorts", {"entry.filters": ["no_squeeze_short"],
                                "entry.filter_params": {"no_squeeze_short": {"max_ret_1d": 0.10}}}),
         ("max 10 open, strongest", {"book.max_open": 10, "book.admit": "strongest"})]

# 5) each rule's effect under the protocol (choice on VAL, TEST shown for the chosen variant only)
table, runs = bt.ladder(sig, px, base, steps, bt.Protocol(enabled=True, metric="final_equity"))
print(table.attrs["protocol"], "| variants:", table.attrs["variants_tried"])
display(table[["variant", "added", "period", "trades", "final_equity", "max_drawdown", "win_rate",
               "mean_net", "profit_factor", "chosen"]])

# 6) the chosen variant's report, every period apart, charts per batch of coins
chosen = table.attrs["chosen"]; run = runs[chosen]
periods = bt.write_report(run, OUT, prices=px, gallery=6, batch_coins=30)
print("chosen on VAL:", chosen)
for p, s in periods.items():
    pt = s["per_trade"]
    print(f"  {p:5s} trades {s['trades']:5d} | equity {s['final_equity']:.4f} | max drawdown {100*s['max_drawdown']:.2f}% "
          f"| win {pt.get('win_rate', float('nan')):.3f} | mean net {100*pt.get('mean', float('nan')):+.3f}%")
print("rule trace:", " -> ".join(f"{k} {v}" for k, v in run.rule_trace))
print("skipped:", {k: v for k, v in run.skipped.items() if k != "coins_without_prices"})
book = bt.TradeBook(run.trades)
display(book.where(period="val").by("exit_reason"))
display(book.where(period="test").worst(5)[["coin", "entry_time", "side", "exit_reason", "net"]])
for png in glob.glob(f"{OUT}/batch_000/equity.png") + sorted(glob.glob(f"{OUT}/batch_000/worst/*.png"))[:3]:
    display(Image(png))
print(f"report: {OUT}/summary.md | trades: {OUT}/trades.parquet | charts per batch: {OUT}/batch_NNN/")

# 7) the web page (a share link is public to whoever holds it; TEST stays hidden until the page's switch)
page = app.build_app(sig, px, title=f"single-coin model ({os.path.basename(RUN_FOLDER)}, 15m candles)")
page.launch(share=True)
