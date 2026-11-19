"""PURPOSE: Write a BTC-relative copy of a contract artifact: each bar's log high / low / close moves minus BTC's of the
    same bar, RSI and stochastic of the coin / BTC price ratio, the coin's relative volume and BTC's own moves and RSI.
TAGS: btc relative, market neutral inputs, ratio price, alt btc pair, rsi, stochastic, derived artifact, btc features
PITFALLS: replaces the base frame's channels (rows, labels, coins, splits unchanged; labels stay the coin's own
    return); BTC's own rows get 0 in the relative channels; the ratio stochastic uses ratio CLOSES only (an ALT/BTC
    pair's intrabar high / low cannot be rebuilt from two USDT candles); a row with a missing coin or BTC bar or too
    little warm-up gets zeros and is counted. Owner task 2026-11-19.

Channels (``CHANNELS``): ``rel_high`` / ``rel_low`` / ``rel_close`` = the artifact's log(x / previous close) minus
BTC's for the same bar; ``ratio_rsi_14``, ``ratio_stoch_k_14_3``, ``ratio_stoch_d_3`` of close_coin / close_BTC,
scaled (x - 50) / 50; ``volume_rel_30`` of the coin (as data_io/indicator_channels.py); ``btc_high`` / ``btc_low`` /
``btc_close`` (BTC's log moves) and ``btc_rsi_14``.

    python -m nig_timenet_torch.data_io.btc_relative SRC DST --cache DIR [--workers 16]
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import TARGET_COLUMNS, TS_COL
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.data_io.artifact import read_manifest
from nig_timenet_torch.data_io.indicator_channels import (
    RSI_LEN,
    STOCH_LEN,
    STOCH_SMOOTH,
    VOL_CLIP,
    VOL_LEN,
    WARMUP_BARS,
    _coin_candles,
)

__all__ = ["CHANNELS", "BTC", "rsi", "stoch_close", "add_btc_relative", "DERIVATION_JSON", "main"]

DERIVATION_JSON = "derivation_btc_relative.json"
BTC = "BTCUSDT"
CHANNELS = ("rel_high", "rel_low", "rel_close", "ratio_rsi_14", "ratio_stoch_k_14_3", "ratio_stoch_d_3",
            "volume_rel_30", "btc_high", "btc_low", "btc_close", "btc_rsi_14")


def rsi(close: pd.Series) -> np.ndarray:
    """Wilder RSI(14) of ``close`` scaled to (x - 50) / 50 (bar i uses bars <= i)."""
    d = close.diff()
    a = 1.0 / RSI_LEN
    au = d.clip(lower=0.0).ewm(alpha=a, adjust=False).mean()
    ad = (-d).clip(lower=0.0).ewm(alpha=a, adjust=False).mean()
    tot = au + ad
    r = np.where(tot > 0, 100.0 * au / tot.where(tot > 0, 1.0), 50.0)
    return (np.asarray(r, np.float64) - 50.0) / 50.0


def stoch_close(close: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Slow stochastic on closes only: raw %K over 14 closes, %K = its 3-bar mean, %D = 3-bar mean of %K, scaled."""
    hh, ll = close.rolling(STOCH_LEN).max(), close.rolling(STOCH_LEN).min()
    rng = hh - ll
    raw = (100.0 * (close - ll) / rng.where(rng > 0, 1.0)).where(rng > 0, 50.0).where(rng.notna())
    k = raw.rolling(STOCH_SMOOTH).mean()
    return (k.to_numpy() - 50.0) / 50.0, (k.rolling(STOCH_SMOOTH).mean().to_numpy() - 50.0) / 50.0


def _on_grid(candles: pd.DataFrame, first: int, step: int, size: int) -> pd.DataFrame:
    idx = pd.DatetimeIndex(candles.index)
    idx = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    ns = idx.as_unit("ns").asi8
    pos = (ns - first) // step
    ok = ((ns - first) % step == 0) & (pos >= 0) & (pos < size)
    out = pd.DataFrame(np.nan, index=np.arange(size), columns=list(candles.columns))
    out.iloc[pos[ok]] = candles.to_numpy(np.float64)[ok]
    return out


def _moves(g: pd.DataFrame) -> np.ndarray:
    prev = g["close"].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([np.log(g[c] / prev).to_numpy() for c in TARGET_COLUMNS], axis=1)


def coin_channels(coin: pd.DataFrame, btc: pd.DataFrame, end_times: np.ndarray, base_tf: str, window: int,
                  x_coin: np.ndarray, is_btc: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(values (n, window, 11) float32, valid (n,) bool) for the windows ending at ``end_times`` (last bar OPEN time,
    int64 ns). ``x_coin`` (n, window, 3): the artifact's own log high / low / close moves of the coin."""
    n = len(end_times)
    out = np.zeros((n, window, len(CHANNELS)), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    if n == 0 or coin.empty or btc.empty:
        return out, valid
    step = pd.Timedelta(base_tf).value
    c_ns = pd.DatetimeIndex(coin.index).as_unit("ns").asi8
    first = int(c_ns.min())
    size = int((int(np.max(end_times)) - first) // step) + 1
    if size <= 0:
        return out, valid
    gc, gb = _on_grid(coin, first, step, size), _on_grid(btc, first, step, size)
    both = gc["close"].notna().to_numpy() & gb["close"].notna().to_numpy()
    ratio = (gc["close"] / gb["close"]).ffill()
    k, d = stoch_close(ratio)
    bmov = _moves(gb)
    lv = np.log1p(gc["quote_volume"].fillna(0.0).clip(lower=0.0))
    ref = np.log1p(gc["quote_volume"].fillna(0.0).clip(lower=0.0).rolling(VOL_LEN).mean().shift(1))
    per_bar = np.column_stack([rsi(ratio), k, d, (lv - ref).clip(-VOL_CLIP, VOL_CLIP).to_numpy(), bmov,
                               rsi(gb["close"].ffill())])                     # (size, 8)
    bar_ok = both & np.isfinite(per_bar).all(axis=1) & (np.arange(size) >= max(RSI_LEN, VOL_LEN + 1))
    last = np.asarray(end_times, dtype=np.int64) - first
    e = last // step
    ok = (last % step == 0) & (e - window + 1 >= 0) & (e < size)
    rows = np.flatnonzero(ok)
    if len(rows):
        take = e[rows][:, None] + np.arange(-window + 1, 1)[None, :]
        good = bar_ok[take].all(axis=1)
        r = rows[good]
        pb = per_bar[take[good]]                                                # (m, window, 8)
        rel = np.zeros((len(r), window, 3)) if is_btc else np.asarray(x_coin, np.float64)[r] - pb[..., 4:7]
        out[r] = np.concatenate([rel, pb[..., :4], pb[..., 4:7], pb[..., 7:8]], axis=-1).astype(np.float32)
        valid[r] = True
    return out, valid


def add_btc_relative(src, dst, *, cache=None, workers: int = 16, fetch=None, overwrite: bool = False) -> dict:
    """Write ``dst``: the one-frame contract artifact ``src`` with its base-frame channels replaced by ``CHANNELS``
    (rows, labels, coins and splits unchanged). Returns the derivation record (also in ``dst``)."""
    from nig_timenet_torch.data.artifact_writer import write_artifact
    from nig_timenet_torch.data.fetch import vision_klines

    fetch = fetch or vision_klines
    src, dst = Path(src), Path(dst)
    man = read_manifest(src)
    ds = npy_dir.load(src, mmap=False)
    if len(ds["timeframes"]) != 1 or list(ds["feature_order"])[:3] != list(TARGET_COLUMNS):
        raise ValueError(f"{src}: needs one frame whose first channels are {TARGET_COLUMNS}")
    base_tf = ds["base_tf"]
    key = f"X_{base_tf}"
    x = np.asarray(ds[key])[..., :3]
    window = x.shape[1]
    cache = None if cache is None else Path(cache)
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    ts = np.asarray(ds["last_candles"])[:, TS_COL].astype(np.int64)
    bar = pd.Timedelta(base_tf)
    lo = pd.Timestamp(int(ts.min()), tz="UTC") - bar * (window + WARMUP_BARS)
    hi = pd.Timestamp(int(ts.max()), tz="UTC") + bar
    btc_all = _coin_candles(BTC, base_tf, lo, hi, None, fetch)
    new = np.zeros((len(ts), window, len(CHANNELS)), dtype=np.float32)

    def one(b: dict) -> tuple[str, int, int]:
        s, e = int(b["start"]), int(b["end"])
        t = ts[s:e]
        start = pd.Timestamp(int(t.min()), tz="UTC") - bar * (window + WARMUP_BARS)
        end = pd.Timestamp(int(t.max()), tz="UTC") + bar
        coin = _coin_candles(b["name"], base_tf, start, end, cache, fetch)
        w, ok = coin_channels(coin, btc_all, t, base_tf, window, x[s:e], is_btc=b["name"] == BTC)
        new[s:e] = w
        return b["name"], int(e - s), int((~ok).sum())

    t0 = time.monotonic()
    per_coin = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, (name, rows, missing) in enumerate(pool.map(one, ds["asset_bounds"])):
            per_coin[name] = {"rows": rows, "missing": missing}
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(ds['asset_bounds'])} coins, {time.monotonic() - t0:.0f} s", flush=True)
    ds = {**ds, key: new, "feature_order": list(CHANNELS)}
    manifest = write_artifact(ds, dst, source_commit=man.source_commit, overwrite=overwrite,
                              producer=f"nig_timenet_torch.data_io.btc_relative (on {man.data_fingerprint})")
    n_missing = sum(v["missing"] for v in per_coin.values())
    record = {"source_artifact": str(src), "source_fingerprint": man.data_fingerprint,
              "fingerprint": manifest["export"]["data_sha256"], "channels": list(CHANNELS),
              "btc_in_artifact": BTC in per_coin, "rows": int(len(ts)), "missing_rows": n_missing,
              "missing_share": n_missing / max(1, len(ts)), "seconds": time.monotonic() - t0, "per_coin": per_coin}
    (dst / DERIVATION_JSON).write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nig_timenet_torch.data_io.btc_relative", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    r = add_btc_relative(a.src, a.dst, cache=a.cache, workers=a.workers, overwrite=a.overwrite)
    print(json.dumps({k: v for k, v in r.items() if k != "per_coin"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
