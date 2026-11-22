"""PURPOSE: Add a finer timeframe's input window (X_<tf>, e.g. the last 32 1h bars) to an existing contract artifact,
    row by row, from Binance archive candles, as a new contract artifact with the same rows, labels and coins.
TAGS: lower timeframe, multi timeframe, 1h window, derived artifact, log moves, binance vision, alignment, x_1h
PITFALLS: the channels copy the artifact's own (log high / low / close over the previous close; checked on BTCUSDT
    2021-05-20 against the archive, owner task 2026-11-18); a row's fine window ends where its base bar closes
    (last_candles timestamp = base bar OPEN time + one base bar); a row whose fine bars are incomplete gets zeros and is
    counted in derivation.json (``missing_share``), never dropped, so the row set stays the source's. The fine
    candles come from the futures archive (the source artifact's coins are futures symbols).

Usage on Colab (the fine candles are cached per coin under ``--cache``, so a cut run resumes)::

    python -m nig_timenet_torch.data.lower_timeframe SRC DST --tf 1h --window 32 --cache DIR [--workers 16]
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import TARGET_COLUMNS, TS_COL
from nig_timenet_torch.data.channel_math import to_grid, window_take
from nig_timenet_torch.data.derive import (
    cache_dir,
    coin_candles,
    load_source,
    make_parser,
    per_coin_map,
    print_record,
    write_derived,
)

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
    first, g = to_grid(candles, step, CHANNELS)
    h, lo, c = (g[k].to_numpy() for k in CHANNELS)
    size = len(g)
    prev = np.concatenate([[np.nan], c[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        ch = np.stack([np.log(h / prev), np.log(lo / prev), np.log(c / prev)], axis=1)   # (size, 3)
    last = (np.asarray(end_times, dtype=np.int64) + base - step - first)
    ok = (last % step == 0) & (last >= 0)
    e = last // step
    ok &= (e - window + 1 >= 0) & (e < size)
    rows = np.flatnonzero(ok)
    if len(rows):
        take = window_take(e[rows], window)                                    # (m, window)
        w = ch[take]                                                           # (m, window, 3)
        good = np.isfinite(w).all(axis=(1, 2))
        out[rows[good]] = w[good].astype(np.float32)
        valid[rows[good]] = True
    return out, valid


def add_timeframe(src, dst, *, tf: str = "1h", window: int = 32, cache=None, workers: int = 16, fetch=None,
                  overwrite: bool = False) -> dict:
    """Write ``dst``: the contract artifact ``src`` plus ``X_<tf>`` (its rows, labels, coins and splits unchanged;
    ``timeframes`` = source frames + ``tf``). Returns the derivation record (also ``dst/derivation.json``)."""
    from nig_timenet_torch.data.fetch import vision_klines

    fetch = fetch or vision_klines
    src, dst = Path(src), Path(dst)
    man, ds = load_source(src)
    if tf in ds["timeframes"]:
        raise ValueError(f"{src} already holds the timeframe {tf!r}")
    base_tf = ds["base_tf"]
    if pd.Timedelta(tf) >= pd.Timedelta(base_tf) or pd.Timedelta(base_tf) % pd.Timedelta(tf):
        raise ValueError(f"{tf!r} must be a finer frame that divides the base frame {base_tf!r}")
    if list(ds["feature_order"]) != list(CHANNELS):
        raise ValueError(f"the source channels are {ds['feature_order']}, this tool writes {CHANNELS}")
    cache = cache_dir(cache)
    ts = np.asarray(ds["last_candles"])[:, TS_COL].astype(np.int64)
    x = np.zeros((len(ts), window, len(CHANNELS)), dtype=np.float32)
    span = pd.Timedelta(tf) * (window + 1)

    def one(b: dict) -> tuple[str, int, int]:
        s, e = int(b["start"]), int(b["end"])
        t = ts[s:e]
        start = pd.Timestamp(int(t.min()), tz="UTC") + pd.Timedelta(base_tf) - span
        end = pd.Timestamp(int(t.max()), tz="UTC") + pd.Timedelta(base_tf)
        candles = coin_candles(b["name"], tf, start, end, cache, fetch, CHANNELS)
        w, ok = log_move_windows(candles, t, tf, window, base_tf)
        x[s:e] = w
        return b["name"], int(e - s), int((~ok).sum())

    results, t0 = per_coin_map(ds, one, workers=workers)
    per_coin = {name: {"rows": rows, "missing": missing} for name, rows, missing in results}
    ds = {**ds, f"X_{tf}": x, "timeframes": [*ds["timeframes"], tf],
          "window_sizes": {**dict(ds["window_sizes"]), tf: window}}
    n_missing = sum(v["missing"] for v in per_coin.values())
    return write_derived(
        src, dst, man, ds, overwrite=overwrite, derivation_json=DERIVATION_JSON,
        producer=f"nig_timenet_torch.data_io.lower_timeframe (+X_{tf} on {man.data_fingerprint})",
        record={"added": f"X_{tf}", "window": window,
                "channels": [f"log({c} / previous close)" for c in CHANNELS],
                "candles": "Binance USD-M futures archive",
                "rows": int(len(ts)), "missing_rows": n_missing, "missing_share": n_missing / max(1, len(ts)),
                "seconds": time.monotonic() - t0, "per_coin": per_coin})


def main(argv: list[str] | None = None) -> int:
    ap = make_parser("python -m nig_timenet_torch.data.lower_timeframe", __doc__)
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--window", type=int, default=32)
    a = ap.parse_args(argv)
    print_record(add_timeframe(a.src, a.dst, tf=a.tf, window=a.window, cache=a.cache, workers=a.workers,
                               overwrite=a.overwrite))
    return 0


if __name__ == "__main__":
    sys.exit(main())
