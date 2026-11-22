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

    python -m nig_timenet_torch.data.indicator_channels SRC DST --cache DIR [--workers 16]
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import QUOTE_VOLUME_COL, TS_COL
from nig_timenet_torch.data.channel_math import (
    RSI_LEN,
    VOL_LEN,
    WARMUP_BARS,
    rsi_scaled,
    stoch_slow,
    to_grid,
    volume_rel,
    window_take,
)
from nig_timenet_torch.data.derive import (
    CANDLE_COLS,
    cache_dir,
    coin_candles,
    load_source,
    make_parser,
    per_coin_map,
    print_record,
    require_one_frame,
    write_derived,
)
from nig_timenet_torch.data.fetch import vision_klines

__all__ = ["CHANNELS", "bar_indicators", "indicator_windows", "add_indicator_channels", "DERIVATION_JSON", "main"]

DERIVATION_JSON = "derivation_indicator_channels.json"
CHANNELS = ("rsi_14", "stoch_k_14_3", "stoch_d_3", "volume_rel_30")
CLOSE_TOL = 2e-4             # |log close move (candles) - X close channel| above this = a mismatched bar


def bar_indicators(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(channels (size, 4), bar valid (size,), log close move (size,)) on a regular grid ``g`` (high / low / close /
    quote_volume, NaN = missing bar). Every value at bar i uses bars <= i only."""
    present = g["close"].notna().to_numpy()
    c = g["close"].ffill()
    h, lo = g["high"].fillna(c), g["low"].fillna(c)
    k, dd = stoch_slow(c, h, lo)
    ch = np.stack([rsi_scaled(c), k, dd, volume_rel(g[QUOTE_VOLUME_COL]).to_numpy()], axis=1)
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
    first, g = to_grid(candles, step, CANDLE_COLS)
    ch, bar_ok, move = bar_indicators(g)
    last = np.asarray(end_times, dtype=np.int64) - first
    ok = (last % step == 0) & (last >= 0)
    e = last // step
    ok &= (e - window + 1 >= 0) & (e < len(g))
    rows = np.flatnonzero(ok)
    if len(rows):
        take = window_take(e[rows], window)
        good = bar_ok[take].all(axis=1)
        if x_close is not None:
            diff = np.abs(move[take] - np.asarray(x_close, np.float64)[rows])
            bad = (np.nan_to_num(diff, nan=0.0) > CLOSE_TOL).any(axis=1)     # NaN = no previous close: not checked
            mismatch[rows[bad]] = True
            good &= ~bad
        out[rows[good]] = ch[take[good]].astype(np.float32)
        valid[rows[good]] = True
    return out, valid, mismatch


def add_indicator_channels(src, dst, *, cache=None, workers: int = 16, fetch=None, overwrite: bool = False) -> dict:
    """Write ``dst``: the one-frame contract artifact ``src`` with the ``CHANNELS`` appended to its base frame's
    windows (rows, labels, coins and splits unchanged). Returns the derivation record (also in ``dst``)."""
    fetch = fetch or vision_klines
    src, dst = Path(src), Path(dst)
    man, ds = load_source(src)
    if set(CHANNELS) & set(ds["feature_order"]):
        raise ValueError(f"{src} already has indicator channels")
    require_one_frame(ds, src, "add the channels to a one-frame artifact")
    base_tf = ds["base_tf"]
    key = f"X_{base_tf}"
    x = np.asarray(ds[key])
    close_i = list(ds["feature_order"]).index("close")
    window = x.shape[1]
    cache = cache_dir(cache)
    ts = np.asarray(ds["last_candles"])[:, TS_COL].astype(np.int64)
    new = np.zeros((len(ts), window, len(CHANNELS)), dtype=np.float32)
    bar = pd.Timedelta(base_tf)

    def one(b: dict) -> tuple[str, int, int, int]:
        s, e = int(b["start"]), int(b["end"])
        t = ts[s:e]
        start = pd.Timestamp(int(t.min()), tz="UTC") - bar * (window + WARMUP_BARS)
        end = pd.Timestamp(int(t.max()), tz="UTC") + bar
        candles = coin_candles(b["name"], base_tf, start, end, cache, fetch, CANDLE_COLS)
        w, ok, mis = indicator_windows(candles, t, base_tf, window, x[s:e, :, close_i])
        new[s:e] = w
        return b["name"], int(e - s), int((~ok).sum()), int(mis.sum())

    results, t0 = per_coin_map(ds, one, workers=workers)
    per_coin = {name: {"rows": rows, "missing": missing, "close_mismatch": mis}
                for name, rows, missing, mis in results}
    ds = {**ds, key: np.concatenate([x, new.astype(x.dtype)], axis=-1),
          "feature_order": [*ds["feature_order"], *CHANNELS]}
    n_missing = sum(v["missing"] for v in per_coin.values())
    n_mis = sum(v["close_mismatch"] for v in per_coin.values())
    return write_derived(
        src, dst, man, ds, overwrite=overwrite, derivation_json=DERIVATION_JSON,
        producer=f"nig_timenet_torch.data_io.indicator_channels (+{len(CHANNELS)} channels on {man.data_fingerprint})",
        record={"added_channels": list(CHANNELS),
                "rules": {"rsi_14": "(Wilder RSI 14 of close - 50) / 50",
                          "stoch_k_14_3": "(SMA3 of raw %K over 14 bars - 50) / 50",
                          "stoch_d_3": "(SMA3 of %K - 50) / 50",
                          "volume_rel_30": "log1p(quote volume) - log1p(mean quote volume of the 30 previous "
                                           "bars), clip +-5"},
                "candles": "Binance USD-M futures archive", "warmup_bars": WARMUP_BARS, "close_tol": CLOSE_TOL,
                "rows": int(len(ts)), "missing_rows": n_missing, "missing_share": n_missing / max(1, len(ts)),
                "close_mismatch_rows": n_mis, "seconds": time.monotonic() - t0, "per_coin": per_coin})


def main(argv: list[str] | None = None) -> int:
    ap = make_parser("python -m nig_timenet_torch.data.indicator_channels", __doc__)
    a = ap.parse_args(argv)
    print_record(add_indicator_channels(a.src, a.dst, cache=a.cache, workers=a.workers, overwrite=a.overwrite))
    return 0


if __name__ == "__main__":
    sys.exit(main())
