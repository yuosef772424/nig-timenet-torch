"""PURPOSE: Add indicator channels (RSI 14, stochastic %K / %D 14-3-3, relative quote volume) to every bar of the base
    frame's windows of a contract artifact, from the coin's own base-frame candles with full warm-up history.
TAGS: indicator channels, rsi, stochastic, volume, relative volume, derived artifact, binance vision, no lookahead
PITFALLS: a bar's values use only that bar and earlier bars (closed at the window's end, rule D2); the candles come
    from the futures archive, so every row is checked against the artifact's own close channel (``close_mismatch``
    in derivation.json); a row whose window has a missing bar, too little warm-up or a close mismatch gets zeros in
    the new channels and is counted (never dropped: rows, labels, coins and splits stay the source's). Owner task
    2026-11-19 (RSI, stochastic, volume).

Channels (each scaled to about [-1, 1]):

* ``rsi_14``: (Wilder RSI(14) of the close - 50) / 50;
* ``stoch_k_14_3`` / ``stoch_d_3``: the slow stochastic (raw %K over 14 bars of high / low, smoothed over 3 bars; %D
  its 3-bar mean), (x - 50) / 50 (a bar whose 14-bar range is 0 counts as 50);
* ``volume_rel_30``: log(1 + quote volume) - log(1 + mean quote volume of the 30 PREVIOUS bars), clipped at +-5.

Usage on Colab (candles cached per coin under ``--cache``, so a cut run resumes)::

    python -m nig_timenet_torch.data_io.indicator_channels SRC DST --cache DIR [--workers 16]
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.data_io.artifact import read_manifest

__all__ = ["CHANNELS", "bar_indicators", "indicator_windows", "add_indicator_channels", "DERIVATION_JSON", "main"]

DERIVATION_JSON = "derivation_indicator_channels.json"
CHANNELS = ("rsi_14", "stoch_k_14_3", "stoch_d_3", "volume_rel_30")
RSI_LEN, STOCH_LEN, STOCH_SMOOTH, VOL_LEN = 14, 14, 3, 30
WARMUP_BARS = 100            # history fetched before a coin's first window (RSI converges in ~ 5 x 14 bars)
VOL_CLIP = 5.0
CLOSE_TOL = 2e-4             # |log close move (candles) - X close channel| above this = a mismatched bar
_CANDLE_COLS = ("high", "low", "close", "quote_volume")


def _grid(candles: pd.DataFrame, step: int) -> tuple[int, pd.DataFrame]:
    """(first open ns, the candles on a regular grid of ``step`` ns: missing bars are NaN rows)."""
    idx = pd.DatetimeIndex(candles.index)
    idx = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    open_ns = idx.as_unit("ns").asi8
    first = int(open_ns.min())
    pos = (open_ns - first) // step
    on = (open_ns - first) % step == 0
    size = int(pos.max()) + 1
    out = pd.DataFrame(np.nan, index=np.arange(size), columns=list(_CANDLE_COLS))
    for c in _CANDLE_COLS:
        out.loc[pos[on], c] = candles[c].to_numpy(np.float64)[on]
    return first, out


def bar_indicators(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(channels (size, 4), bar valid (size,), log close move (size,)) on a regular grid ``g`` (high / low / close /
    quote_volume, NaN = missing bar). Every value at bar i uses bars <= i only."""
    present = g["close"].notna().to_numpy()
    c = g["close"].ffill()
    h, lo = g["high"].fillna(c), g["low"].fillna(c)
    d = c.diff()
    up, dn = d.clip(lower=0.0), (-d).clip(lower=0.0)
    a = 1.0 / RSI_LEN
    au, ad = up.ewm(alpha=a, adjust=False).mean(), dn.ewm(alpha=a, adjust=False).mean()
    rsi = np.where(au + ad > 0, 100.0 * au / (au + ad).where(au + ad > 0, 1.0), 50.0)
    hh, ll = h.rolling(STOCH_LEN).max(), lo.rolling(STOCH_LEN).min()
    rng = hh - ll
    raw_k = (100.0 * (c - ll) / rng.where(rng > 0, 1.0)).where(rng > 0, 50.0).where(rng.notna())
    k = raw_k.rolling(STOCH_SMOOTH).mean()
    dd = k.rolling(STOCH_SMOOTH).mean()
    lv = np.log1p(g["quote_volume"].fillna(0.0).clip(lower=0.0))
    ref = np.log1p(g["quote_volume"].fillna(0.0).clip(lower=0.0).rolling(VOL_LEN).mean().shift(1))
    vol = (lv - ref).clip(-VOL_CLIP, VOL_CLIP)
    ch = np.stack([(np.asarray(rsi, np.float64) - 50.0) / 50.0, (k.to_numpy() - 50.0) / 50.0,
                   (dd.to_numpy() - 50.0) / 50.0, vol.to_numpy()], axis=1)
    idx = np.arange(len(g))
    valid = present & np.isfinite(ch).all(axis=1) & (idx >= max(RSI_LEN, VOL_LEN + 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        move = np.log(g["close"].to_numpy() / np.concatenate([[np.nan], g["close"].to_numpy()[:-1]]))
    return ch, valid, move


def indicator_windows(candles: pd.DataFrame, end_times: np.ndarray, base_tf: str, window: int,
                      x_close: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(values (n, window, 4) float32, valid (n,) bool, close mismatch (n,) bool): for each window whose last bar OPENS
    at ``end_times`` (int64 ns, UTC), the channels of its ``window`` bars. ``x_close`` (n, window): the artifact's own
    log close moves; a row whose candles disagree with it is a mismatch and invalid. Invalid rows are zeros."""
    n = len(end_times)
    out = np.zeros((n, window, len(CHANNELS)), dtype=np.float32)
    valid, mismatch = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    if n == 0 or candles.empty:
        return out, valid, mismatch
    step = pd.Timedelta(base_tf).value
    first, g = _grid(candles, step)
    ch, bar_ok, move = bar_indicators(g)
    last = np.asarray(end_times, dtype=np.int64) - first
    ok = (last % step == 0) & (last >= 0)
    e = last // step
    ok &= (e - window + 1 >= 0) & (e < len(g))
    rows = np.flatnonzero(ok)
    if len(rows):
        take = e[rows][:, None] + np.arange(-window + 1, 1)[None, :]
        good = bar_ok[take].all(axis=1)
        if x_close is not None:
            diff = np.abs(move[take] - np.asarray(x_close, np.float64)[rows])
            bad = (np.nan_to_num(diff, nan=0.0) > CLOSE_TOL).any(axis=1)     # NaN = no previous close: not checked
            mismatch[rows[bad]] = True
            good &= ~bad
        out[rows[good]] = ch[take[good]].astype(np.float32)
        valid[rows[good]] = True
    return out, valid, mismatch


def _coin_candles(symbol: str, base_tf: str, start, end, cache: Path | None, fetch) -> pd.DataFrame:
    path = None if cache is None else cache / f"{symbol}_{base_tf}_hlcqv.parquet"
    if path is not None and path.is_file():
        return pd.read_parquet(path)
    frame = fetch(symbol, base_tf, start, end)[list(_CANDLE_COLS)]
    if path is not None:
        tmp = path.with_suffix(".tmp")
        frame.to_parquet(tmp)
        tmp.rename(path)
    return frame


def add_indicator_channels(src, dst, *, cache=None, workers: int = 16, fetch=None, overwrite: bool = False) -> dict:
    """Write ``dst``: the one-frame contract artifact ``src`` with the ``CHANNELS`` appended to its base frame's
    windows (rows, labels, coins and splits unchanged). Returns the derivation record (also in ``dst``)."""
    from nig_timenet_torch.data.artifact_writer import write_artifact
    from nig_timenet_torch.data.fetch import vision_klines

    fetch = fetch or vision_klines
    src, dst = Path(src), Path(dst)
    man = read_manifest(src)
    ds = npy_dir.load(src, mmap=False)
    if set(CHANNELS) & set(ds["feature_order"]):
        raise ValueError(f"{src} already has indicator channels")
    if len(ds["timeframes"]) != 1:
        raise ValueError(f"{src} holds {list(ds['timeframes'])}: add the channels to a one-frame artifact")
    base_tf = ds["base_tf"]
    key = f"X_{base_tf}"
    x = np.asarray(ds[key])
    close_i = list(ds["feature_order"]).index("close")
    window = x.shape[1]
    cache = None if cache is None else Path(cache)
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    ts = np.asarray(ds["last_candles"])[:, TS_COL].astype(np.int64)
    new = np.zeros((len(ts), window, len(CHANNELS)), dtype=np.float32)
    bar = pd.Timedelta(base_tf)

    def one(b: dict) -> tuple[str, int, int, int]:
        s, e = int(b["start"]), int(b["end"])
        t = ts[s:e]
        start = pd.Timestamp(int(t.min()), tz="UTC") - bar * (window + WARMUP_BARS)
        end = pd.Timestamp(int(t.max()), tz="UTC") + bar
        candles = _coin_candles(b["name"], base_tf, start, end, cache, fetch)
        w, ok, mis = indicator_windows(candles, t, base_tf, window, x[s:e, :, close_i])
        new[s:e] = w
        return b["name"], int(e - s), int((~ok).sum()), int(mis.sum())

    t0 = time.monotonic()
    per_coin = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, (name, rows, missing, mis) in enumerate(pool.map(one, ds["asset_bounds"])):
            per_coin[name] = {"rows": rows, "missing": missing, "close_mismatch": mis}
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(ds['asset_bounds'])} coins, {time.monotonic() - t0:.0f} s", flush=True)
    ds = {**ds, key: np.concatenate([x, new.astype(x.dtype)], axis=-1),
          "feature_order": [*ds["feature_order"], *CHANNELS]}
    manifest = write_artifact(ds, dst, source_commit=man.source_commit, overwrite=overwrite,
                              producer=f"nig_timenet_torch.data_io.indicator_channels (+{len(CHANNELS)} channels on "
                                       f"{man.data_fingerprint})")
    n_missing = sum(v["missing"] for v in per_coin.values())
    n_mis = sum(v["close_mismatch"] for v in per_coin.values())
    record = {"source_artifact": str(src), "source_fingerprint": man.data_fingerprint,
              "fingerprint": manifest["export"]["data_sha256"], "added_channels": list(CHANNELS),
              "rules": {"rsi_14": "(Wilder RSI 14 of close - 50) / 50",
                        "stoch_k_14_3": "(SMA3 of raw %K over 14 bars - 50) / 50",
                        "stoch_d_3": "(SMA3 of %K - 50) / 50",
                        "volume_rel_30": "log1p(quote volume) - log1p(mean quote volume of the 30 previous bars), "
                                         "clip +-5"},
              "candles": "Binance USD-M futures archive", "warmup_bars": WARMUP_BARS, "close_tol": CLOSE_TOL,
              "rows": int(len(ts)), "missing_rows": n_missing, "missing_share": n_missing / max(1, len(ts)),
              "close_mismatch_rows": n_mis, "seconds": time.monotonic() - t0, "per_coin": per_coin}
    (dst / DERIVATION_JSON).write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nig_timenet_torch.data_io.indicator_channels", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    r = add_indicator_channels(a.src, a.dst, cache=a.cache, workers=a.workers, overwrite=a.overwrite)
    print(json.dumps({k: v for k, v in r.items() if k != "per_coin"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
