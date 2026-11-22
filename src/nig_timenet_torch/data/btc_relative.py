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

    python -m nig_timenet_torch.data.btc_relative SRC DST --cache DIR [--workers 16]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import (
    LAST_COLUMN_INDEX,
    QUOTE_VOLUME_COL,
    TARGET_COLUMNS,
    TS_COL,
    class_label_key,
    reg_label_key,
)
from nig_timenet_torch.data.channel_math import (
    RSI_LEN,
    VOL_LEN,
    WARMUP_BARS,
    rsi_scaled,
    stoch_slow,
    to_grid,
    to_ns_utc,
    volume_rel,
    window_take,
)
from nig_timenet_torch.data.derive import (
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

__all__ = ["CHANNELS", "BTC", "add_btc_relative", "btc_moves_over_horizon", "relative_close_labels",
           "DERIVATION_JSON", "LABEL_JSON", "main"]

DERIVATION_JSON = "derivation_btc_relative.json"
LABEL_JSON = "derivation_btc_relative_label.json"
BTC = "BTCUSDT"
CHANNELS = ("rel_high", "rel_low", "rel_close", "ratio_rsi_14", "ratio_stoch_k_14_3", "ratio_stoch_d_3",
            "volume_rel_30", "btc_high", "btc_low", "btc_close", "btc_rsi_14")


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
    c_ns = to_ns_utc(coin.index)
    first = int(c_ns.min())
    size = int((int(np.max(end_times)) - first) // step) + 1
    if size <= 0:
        return out, valid
    gc = to_grid(coin, step, first=first, size=size)[1]
    gb = to_grid(btc, step, first=first, size=size)[1]
    both = gc["close"].notna().to_numpy() & gb["close"].notna().to_numpy()
    ratio = (gc["close"] / gb["close"]).ffill()
    k, d = stoch_slow(ratio)
    bmov = _moves(gb)
    per_bar = np.column_stack([rsi_scaled(ratio), k, d, volume_rel(gc[QUOTE_VOLUME_COL]).to_numpy(), bmov,
                               rsi_scaled(gb["close"].ffill())])              # (size, 8)
    bar_ok = both & np.isfinite(per_bar).all(axis=1) & (np.arange(size) >= max(RSI_LEN, VOL_LEN + 1))
    last = np.asarray(end_times, dtype=np.int64) - first
    e = last // step
    ok = (last % step == 0) & (e - window + 1 >= 0) & (e < size)
    rows = np.flatnonzero(ok)
    if len(rows):
        take = window_take(e[rows], window)
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
    fetch = fetch or vision_klines
    src, dst = Path(src), Path(dst)
    man, ds = load_source(src)
    require_one_frame(ds, src, f"needs one frame whose first channels are {TARGET_COLUMNS}")
    if list(ds["feature_order"])[:3] != list(TARGET_COLUMNS):
        raise ValueError(f"{src}: needs one frame whose first channels are {TARGET_COLUMNS}")
    base_tf = ds["base_tf"]
    key = f"X_{base_tf}"
    x = np.asarray(ds[key])[..., :3]
    window = x.shape[1]
    cache = cache_dir(cache)
    ts = np.asarray(ds["last_candles"])[:, TS_COL].astype(np.int64)
    bar = pd.Timedelta(base_tf)
    lo = pd.Timestamp(int(ts.min()), tz="UTC") - bar * (window + WARMUP_BARS)
    hi = pd.Timestamp(int(ts.max()), tz="UTC") + bar
    btc_all = coin_candles(BTC, base_tf, lo, hi, None, fetch)
    new = np.zeros((len(ts), window, len(CHANNELS)), dtype=np.float32)

    def one(b: dict) -> tuple[str, int, int]:
        s, e = int(b["start"]), int(b["end"])
        t = ts[s:e]
        start = pd.Timestamp(int(t.min()), tz="UTC") - bar * (window + WARMUP_BARS)
        end = pd.Timestamp(int(t.max()), tz="UTC") + bar
        coin = coin_candles(b["name"], base_tf, start, end, cache, fetch)
        w, ok = coin_channels(coin, btc_all, t, base_tf, window, x[s:e], is_btc=b["name"] == BTC)
        new[s:e] = w
        return b["name"], int(e - s), int((~ok).sum())

    results, t0 = per_coin_map(ds, one, workers=workers, progress_every=50)
    per_coin = {name: {"rows": rows, "missing": missing} for name, rows, missing in results}
    ds = {**ds, key: new, "feature_order": list(CHANNELS)}
    n_missing = sum(v["missing"] for v in per_coin.values())
    return write_derived(
        src, dst, man, ds, overwrite=overwrite, derivation_json=DERIVATION_JSON,
        producer=f"nig_timenet_torch.data_io.btc_relative (on {man.data_fingerprint})",
        record={"channels": list(CHANNELS), "btc_in_artifact": BTC in per_coin, "rows": int(len(ts)),
                "missing_rows": n_missing, "missing_share": n_missing / max(1, len(ts)),
                "seconds": time.monotonic() - t0, "per_coin": per_coin})


def btc_moves_over_horizon(btc: pd.DataFrame, ts: np.ndarray, base_tf: str, horizon: int) -> np.ndarray:
    """log(BTC close of the bar opening at ts + horizon bars / BTC close of the bar opening at ts) per row (the close
    label's own span: label_end = ts + (horizon + 1) bars); NaN where a BTC bar is missing."""
    close = pd.Series(btc["close"].to_numpy(np.float64), index=to_ns_utc(btc.index))
    close = close[~close.index.duplicated()]
    t = np.asarray(ts, dtype=np.int64)
    now = close.reindex(t).to_numpy()
    later = close.reindex(t + horizon * pd.Timedelta(base_tf).value).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(later / now)


def relative_close_labels(src, dst, *, fetch=None, overwrite: bool = False) -> dict:
    """Write ``dst``: ``src`` with the CLOSE label made relative to BTC: y_close_reg = clip(log(coin future / last
    close) - log(BTC future / BTC last close), +-scale), y_close_class = (it > 0), and last_candles future_close =
    last_close * exp(label) so the codec decodes it (rule B3). High / low labels, inputs and rows are unchanged; rows
    without a BTC bar keep their own label and are counted."""
    fetch = fetch or vision_klines
    src, dst = Path(src), Path(dst)
    man, ds = load_source(src)
    lc = np.array(ds["last_candles"], dtype=np.float64)
    ts = lc[:, TS_COL].astype(np.int64)
    h, bar = int(ds["forecast_horizon"]), pd.Timedelta(ds["base_tf"])
    btc = coin_candles(BTC, ds["base_tf"], pd.Timestamp(int(ts.min()), tz="UTC") - bar,
                       pd.Timestamp(int(ts.max()), tz="UTC") + bar * (h + 2), None, fetch)
    m = btc_moves_over_horizon(btc, ts, ds["base_tf"], h)
    ok = np.isfinite(m)
    scale = float(ds.get("reg_target_scale", 1.0))
    reg_key = reg_label_key("close")
    cls_key = class_label_key("close")
    y = np.asarray(ds[reg_key], dtype=np.float64)
    rel = np.where(ok, np.clip(y - np.where(ok, m, 0.0), -scale, scale), y)
    lc[:, LAST_COLUMN_INDEX["future_close"]] = lc[:, LAST_COLUMN_INDEX["last_close"]] * np.exp(rel)
    ds = {**ds, reg_key: rel.astype(np.asarray(ds[reg_key]).dtype),
          cls_key: (rel > 0).astype(np.asarray(ds[cls_key]).dtype), "last_candles": lc}
    return write_derived(
        src, dst, man, ds, overwrite=overwrite, derivation_json=LABEL_JSON,
        producer=f"nig_timenet_torch.data_io.btc_relative close label (on {man.data_fingerprint})",
        record={"rule": "y_close_reg = clip(own close label - log BTC move over the same span, +-scale); "
                        "last_candles future_close = last_close * exp(label)",
                "rows": int(len(ts)), "rows_without_btc": int((~ok).sum()),
                "share_up_before": float((y > 0).mean()), "share_up_after": float((rel > 0).mean())})


def main(argv: list[str] | None = None) -> int:
    ap = make_parser("python -m nig_timenet_torch.data.btc_relative", __doc__)
    ap.add_argument("--label-only", action="store_true",
                    help="only make the close label relative to BTC (relative_close_labels); channels unchanged")
    a = ap.parse_args(argv)
    if a.label_only:
        print(json.dumps(relative_close_labels(a.src, a.dst, overwrite=a.overwrite), indent=1))
        return 0
    print_record(add_btc_relative(a.src, a.dst, cache=a.cache, workers=a.workers, overwrite=a.overwrite))
    return 0


if __name__ == "__main__":
    sys.exit(main())
