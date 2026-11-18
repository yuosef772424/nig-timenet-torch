"""PURPOSE: Add a finer timeframe's input window (X_<tf>, e.g. the last 32 1h bars) to an existing contract artifact,
    row by row, from Binance archive candles, as a new contract artifact with the same rows, labels and coins.
TAGS: lower timeframe, multi timeframe, 1h window, derived artifact, log moves, binance vision, alignment, x_1h
PITFALLS: the channels copy the artifact's own (log high / low / close over the previous close; checked on BTCUSDT
    2021-05-20 against the archive, owner task 2026-11-18); a row's fine window ends where its base bar closes
    (last_candles timestamp = base bar OPEN time + one base bar); a row whose fine bars are incomplete gets zeros and is
    counted in derivation.json (``missing_share``), never dropped, so the row set stays the source's. The fine
    candles come from the futures archive (the source artifact's coins are futures symbols).

Usage on Colab (the fine candles are cached per coin under ``--cache``, so a cut run resumes)::

    python -m nig_timenet_torch.data_io.lower_timeframe SRC DST --tf 1h --window 32 --cache DIR [--workers 16]
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

__all__ = ["log_move_windows", "add_timeframe", "DERIVATION_JSON", "main"]

DERIVATION_JSON = "derivation.json"
#: the channels of the source artifact, in order (its feature_order: the price targets high / low / close)
CHANNELS = tuple(TARGET_COLUMNS)


def log_move_windows(candles: pd.DataFrame, end_times: np.ndarray, tf: str, window: int,
                     base_tf: str) -> tuple[np.ndarray, np.ndarray]:
    """(X (n, window, 3) float32, valid (n,) bool): for each base bar OPEN time in ``end_times`` (int64 ns, UTC), the
    ``window`` bars of ``tf`` that close at or before the base bar's close, as log(high / previous close),
    log(low / previous close), log(close / previous close). ``candles`` is indexed by bar open time (UTC) with
    high / low / close. A row with a missing bar (or without the previous close) is invalid and all zeros."""
    n = len(end_times)
    out = np.zeros((n, window, len(CHANNELS)), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    if n == 0 or candles.empty:
        return out, valid
    step = pd.Timedelta(tf).value
    base = pd.Timedelta(base_tf).value
    idx = pd.DatetimeIndex(candles.index)
    idx = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    open_ns = idx.as_unit("ns").asi8
    first = int(open_ns.min())
    pos = (open_ns - first) // step
    on_grid = (open_ns - first) % step == 0
    size = int(pos.max()) + 1
    h, lo, c = (np.full(size, np.nan) for _ in range(3))
    h[pos[on_grid]] = candles["high"].to_numpy(np.float64)[on_grid]
    lo[pos[on_grid]] = candles["low"].to_numpy(np.float64)[on_grid]
    c[pos[on_grid]] = candles["close"].to_numpy(np.float64)[on_grid]
    prev = np.concatenate([[np.nan], c[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        ch = np.stack([np.log(h / prev), np.log(lo / prev), np.log(c / prev)], axis=1)   # (size, 3)
    last = (np.asarray(end_times, dtype=np.int64) + base - step - first)
    ok = (last % step == 0) & (last >= 0)
    e = last // step
    ok &= (e - window + 1 >= 0) & (e < size)
    rows = np.flatnonzero(ok)
    if len(rows):
        take = e[rows][:, None] + np.arange(-window + 1, 1)[None, :]          # (m, window)
        w = ch[take]                                                           # (m, window, 3)
        good = np.isfinite(w).all(axis=(1, 2))
        out[rows[good]] = w[good].astype(np.float32)
        valid[rows[good]] = True
    return out, valid


def _coin_candles(symbol: str, tf: str, start: pd.Timestamp, end: pd.Timestamp, cache: Path | None,
                  fetch) -> pd.DataFrame:
    path = None if cache is None else cache / f"{symbol}_{tf}.parquet"
    if path is not None and path.is_file():
        return pd.read_parquet(path)
    frame = fetch(symbol, tf, start, end)[list(CHANNELS)]
    if path is not None:
        tmp = path.with_suffix(".tmp")
        frame.to_parquet(tmp)
        tmp.rename(path)
    return frame


def add_timeframe(src, dst, *, tf: str = "1h", window: int = 32, cache=None, workers: int = 16, fetch=None,
                  overwrite: bool = False) -> dict:
    """Write ``dst``: the contract artifact ``src`` plus ``X_<tf>`` (its rows, labels, coins and splits unchanged;
    ``timeframes`` = source frames + ``tf``). Returns the derivation record (also ``dst/derivation.json``)."""
    from nig_timenet_torch.data.artifact_writer import write_artifact
    from nig_timenet_torch.data.fetch import vision_klines

    fetch = fetch or vision_klines
    src, dst = Path(src), Path(dst)
    man = read_manifest(src)
    ds = npy_dir.load(src, mmap=False)
    if tf in ds["timeframes"]:
        raise ValueError(f"{src} already holds the timeframe {tf!r}")
    base_tf = ds["base_tf"]
    if pd.Timedelta(tf) >= pd.Timedelta(base_tf) or pd.Timedelta(base_tf) % pd.Timedelta(tf):
        raise ValueError(f"{tf!r} must be a finer frame that divides the base frame {base_tf!r}")
    if list(ds["feature_order"]) != list(CHANNELS):
        raise ValueError(f"the source channels are {ds['feature_order']}, this tool writes {CHANNELS}")
    cache = None if cache is None else Path(cache)
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    ts = np.asarray(ds["last_candles"])[:, TS_COL].astype(np.int64)
    x = np.zeros((len(ts), window, len(CHANNELS)), dtype=np.float32)
    span = pd.Timedelta(tf) * (window + 1)

    def one(b: dict) -> tuple[str, int, int]:
        s, e = int(b["start"]), int(b["end"])
        t = ts[s:e]
        start = pd.Timestamp(int(t.min()), tz="UTC") + pd.Timedelta(base_tf) - span
        end = pd.Timestamp(int(t.max()), tz="UTC") + pd.Timedelta(base_tf)
        candles = _coin_candles(b["name"], tf, start, end, cache, fetch)
        w, ok = log_move_windows(candles, t, tf, window, base_tf)
        x[s:e] = w
        return b["name"], int(e - s), int((~ok).sum())

    t0 = time.monotonic()
    per_coin = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, (name, rows, missing) in enumerate(pool.map(one, ds["asset_bounds"])):
            per_coin[name] = {"rows": rows, "missing": missing}
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(ds['asset_bounds'])} coins, {time.monotonic() - t0:.0f} s", flush=True)
    ds = {**ds, f"X_{tf}": x, "timeframes": [*ds["timeframes"], tf],
          "window_sizes": {**dict(ds["window_sizes"]), tf: window}}
    manifest = write_artifact(ds, dst, source_commit=man.source_commit, overwrite=overwrite,
                              producer=f"nig_timenet_torch.data_io.lower_timeframe (+X_{tf} on {man.data_fingerprint})")
    n_missing = sum(v["missing"] for v in per_coin.values())
    record = {"source_artifact": str(src), "source_fingerprint": man.data_fingerprint,
              "fingerprint": manifest["export"]["data_sha256"], "added": f"X_{tf}", "window": window,
              "channels": [f"log({c} / previous close)" for c in CHANNELS], "candles": "Binance USD-M futures archive",
              "rows": int(len(ts)), "missing_rows": n_missing, "missing_share": n_missing / max(1, len(ts)),
              "seconds": time.monotonic() - t0, "per_coin": per_coin}
    (dst / DERIVATION_JSON).write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nig_timenet_torch.data_io.lower_timeframe", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    r = add_timeframe(a.src, a.dst, tf=a.tf, window=a.window, cache=a.cache, workers=a.workers,
                      overwrite=a.overwrite)
    print(json.dumps({k: v for k, v in r.items() if k != "per_coin"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
