"""PURPOSE: Write a market-residual copy of a contract artifact: each bar's coin moves minus beta x the equal-weight
    market move (beta per coin from its own past only), residual RSI / stochastic, market features, and the close label
    made residual too.
TAGS: market residual, beta, market neutral, idiosyncratic return, equal weight market, derived artifact, rolling beta
PITFALLS: beta at bar k uses bars <= k only (rolling ``BETA_BARS``, at least ``BETA_MIN`` bars; else 1, counted);
    the market is the equal-weight mean of the artifact's own coins' close log moves per bar (each clipped at
    +-``MOVE_CLIP``, at least ``MARKET_MIN`` coins), so it depends on the universe; the close label becomes
    clip(own - beta_t x market move over the label span) and last_candles future_close is rebuilt so the codec
    decodes it; high / low labels unchanged. Owner task 2026-11-19 (method 3).

Channels (``CHANNELS``): ``res_high`` / ``res_low`` / ``res_close`` = the artifact's log(x / previous close) minus
beta_k x market move of bar k; ``res_rsi_14``, ``res_stoch_k_14_3``, ``res_stoch_d_3`` of the residual price path
(exp of the cumulative residual close move; stochastic on its closes); ``volume_rel_30``; ``mkt_close`` (the market's
log move) and ``mkt_rsi_14``; ``beta`` (the coin's beta at bar k).

    python -m nig_timenet_torch.data_io.market_residual SRC DST (--cache DIR | --raw VISION_RAW_DIR)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX, TARGET_COLUMNS, TS_COL
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.data_io.artifact import read_manifest
from nig_timenet_torch.data_io.btc_relative import rsi, stoch_close
from nig_timenet_torch.data_io.indicator_channels import VOL_CLIP, VOL_LEN, WARMUP_BARS, _coin_candles

__all__ = ["CHANNELS", "market_panel", "rolling_beta", "add_market_residual", "DERIVATION_JSON", "main"]

DERIVATION_JSON = "derivation_market_residual.json"
CHANNELS = ("res_high", "res_low", "res_close", "res_rsi_14", "res_stoch_k_14_3", "res_stoch_d_3", "volume_rel_30",
            "mkt_close", "mkt_rsi_14", "beta")
BETA_BARS, BETA_MIN = 180, 60          # 30 days of 4h bars; fewer than BETA_MIN bars -> beta 1
BETA_CLIP = (0.0, 3.0)
MOVE_CLIP = 0.5                         # a coin's bar move enters the market mean clipped at +-0.5 (data glitches)
MARKET_MIN = 5


def market_panel(closes: dict[str, pd.Series], step: int) -> tuple[int, np.ndarray, dict[str, np.ndarray]]:
    """(first open ns, market log move per grid bar (NaN with fewer than MARKET_MIN coins), {coin: its close log move
    per grid bar}) on one regular grid of ``step`` ns over all coins."""
    first = min(int(pd.DatetimeIndex(s.index).as_unit("ns").asi8.min()) for s in closes.values())
    last = max(int(pd.DatetimeIndex(s.index).as_unit("ns").asi8.max()) for s in closes.values())
    size = (last - first) // step + 1
    moves, total, count = {}, np.zeros(size), np.zeros(size)
    for name, s in closes.items():
        ns = pd.DatetimeIndex(s.index).as_unit("ns").asi8
        on = (ns - first) % step == 0
        c = np.full(size, np.nan)
        c[(ns[on] - first) // step] = s.to_numpy(np.float64)[on]
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.log(c / np.concatenate([[np.nan], c[:-1]]))
        moves[name] = m
        ok = np.isfinite(m)
        total[ok] += np.clip(m[ok], -MOVE_CLIP, MOVE_CLIP)
        count[ok] += 1
    with np.errstate(invalid="ignore"):
        mkt = np.where(count >= MARKET_MIN, total / np.maximum(count, 1), np.nan)
    return first, mkt, moves


def rolling_beta(r: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Beta of ``r`` on ``m`` over the last BETA_BARS bars ending at each bar (bars <= k only), NaN with fewer than
    BETA_MIN paired bars; clipped to BETA_CLIP."""
    df = pd.DataFrame({"r": r, "m": m})
    df.loc[~(np.isfinite(df["r"]) & np.isfinite(df["m"])), :] = np.nan
    roll = df.rolling(BETA_BARS, min_periods=BETA_MIN)
    beta = roll.cov()["r"].xs("m", level=1) / roll.var()["m"]
    return np.clip(beta.to_numpy(), *BETA_CLIP)


def _coin_series(g_r: np.ndarray, mkt: np.ndarray, vol: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(per-bar channels 4..9 (size, 6), beta used per bar (1 where unknown), bar valid) of one coin on the grid."""
    b = rolling_beta(g_r, mkt)
    known = np.isfinite(b)
    beta = np.where(known, b, 1.0)
    res = g_r - beta * mkt
    path = pd.Series(np.exp(np.nancumsum(np.where(np.isfinite(res), res, 0.0))))
    k, d = stoch_close(path)
    v = pd.Series(np.nan_to_num(vol, nan=0.0)).clip(lower=0.0)
    vr = (np.log1p(v) - np.log1p(v.rolling(VOL_LEN).mean().shift(1))).clip(-VOL_CLIP, VOL_CLIP).to_numpy()
    mkt_path = pd.Series(np.exp(np.nancumsum(np.nan_to_num(mkt))))
    ch = np.column_stack([rsi(path), k, d, vr, mkt, rsi(mkt_path), beta])
    valid = np.isfinite(g_r) & np.isfinite(mkt) & np.isfinite(ch).all(axis=1)
    valid &= np.arange(len(g_r)) >= max(VOL_LEN + 1, 14)
    return ch, beta, valid


def _fill_coin(b, f, coin_moves, mkt, cum_mkt, first, step, ts, x, h, window, y, y_new, new, scale) -> tuple:
    """Write one coin's channels into ``new`` and its residual close label into ``y_new`` (in place); returns its
    record and the count of its label rows whose beta was unknown (1 used)."""
    size = len(mkt)
    s, e = int(b["start"]), int(b["end"])
    ns = pd.DatetimeIndex(f.index).as_unit("ns").asi8
    on = (ns - first) % step == 0
    vol = np.full(size, np.nan)
    vcol = "quote_volume" if "quote_volume" in f.columns else "volume"
    vol[(ns[on] - first) // step] = f[vcol].to_numpy(np.float64)[on]
    ch, beta, ok_bar = _coin_series(coin_moves, mkt, vol)
    pos = (ts[s:e] - first) // step
    good = ((ts[s:e] - first) % step == 0) & (pos - window + 1 >= 0) & (pos + h < size)
    rows = np.flatnonzero(good)
    take = pos[rows][:, None] + np.arange(-window + 1, 1)[None, :]
    okw = ok_bar[take].all(axis=1)
    r = rows[okw]
    bt = beta[take[okw]]                                                             # (m, window)
    res3 = x[s + r].astype(np.float64) - bt[..., None] * mkt[take[okw]][..., None]
    new[s + r] = np.concatenate([res3, ch[take[okw]]], axis=-1).astype(np.float32)
    p = pos[rows]
    unknown = int((~np.isfinite(rolling_beta(coin_moves, mkt))[p]).sum())
    m_span = cum_mkt[p + h + 1] - cum_mkt[p + 1]                                      # market move over the label span
    y_new[s + rows] = np.clip(y[s + rows] - beta[p] * m_span, -scale, scale)
    return {"rows": e - s, "written": int(len(r)), "label_rows": int(len(rows))}, unknown


def add_market_residual(src, dst, *, candles=None, cache=None, fetch=None, overwrite: bool = False) -> dict:
    """Write ``dst``: ``src`` with its base-frame channels replaced by ``CHANNELS`` and the close label residual.
    ``candles(name, start, end)`` returns a coin's base-frame candles (high / low / close / volume or quote_volume,
    indexed by bar open time); default: the archive through ``_coin_candles`` (``cache`` / ``fetch``)."""
    from nig_timenet_torch.data.artifact_writer import write_artifact
    from nig_timenet_torch.data.fetch import vision_klines
    from nig_timenet_torch.model.heads import head_spec

    src, dst = Path(src), Path(dst)
    man = read_manifest(src)
    ds = npy_dir.load(src, mmap=False)
    if len(ds["timeframes"]) != 1 or list(ds["feature_order"])[:3] != list(TARGET_COLUMNS):
        raise ValueError(f"{src}: needs one frame whose first channels are {TARGET_COLUMNS}")
    base_tf, h = ds["base_tf"], int(ds["forecast_horizon"])
    step, bar = pd.Timedelta(base_tf).value, pd.Timedelta(base_tf)
    if candles is None:
        fetch = fetch or vision_klines
        cache = None if cache is None else Path(cache)

        def candles(name, start, end):
            return _coin_candles(name, base_tf, start, end, cache, fetch)

    x = np.asarray(ds[f"X_{base_tf}"])[..., :3]
    window = x.shape[1]
    lc = np.array(ds["last_candles"], dtype=np.float64)
    ts = lc[:, TS_COL].astype(np.int64)
    t0 = time.monotonic()
    frames = {}
    for b in ds["asset_bounds"]:
        t = ts[int(b["start"]):int(b["end"])]
        frames[b["name"]] = candles(b["name"], pd.Timestamp(int(t.min()), tz="UTC") - bar * (window + WARMUP_BARS),
                                    pd.Timestamp(int(t.max()), tz="UTC") + bar * (h + 1))
    first, mkt, moves = market_panel({k: f["close"] for k, f in frames.items()}, step)
    cum_mkt = np.concatenate([[0.0], np.cumsum(np.nan_to_num(mkt))])       # cum_mkt[j] = sum of mkt[:j]
    new = np.zeros((len(ts), window, len(CHANNELS)), dtype=np.float32)
    reg_key = head_spec("nig_regression", "close").label_key
    cls_key = head_spec("binary_classification", "close").label_key
    y = np.asarray(ds[reg_key], dtype=np.float64)
    y_new = y.copy()
    scale = float(ds.get("reg_target_scale", 1.0))
    per_coin, n_unknown_beta = {}, 0
    for b in ds["asset_bounds"]:
        rec, unknown = _fill_coin(b, frames[b["name"]], moves[b["name"]], mkt, cum_mkt, first, step, ts, x, h,
                                  window, y, y_new, new, scale)
        per_coin[b["name"]], n_unknown_beta = rec, n_unknown_beta + unknown
    lc[:, LAST_COLUMN_INDEX["future_close"]] = lc[:, LAST_COLUMN_INDEX["last_close"]] * np.exp(y_new)
    ds = {**ds, f"X_{base_tf}": new, "feature_order": list(CHANNELS), "last_candles": lc,
          reg_key: y_new.astype(np.asarray(ds[reg_key]).dtype),
          cls_key: (y_new > 0).astype(np.asarray(ds[cls_key]).dtype)}
    manifest = write_artifact(ds, dst, source_commit=man.source_commit, overwrite=overwrite,
                              producer=f"nig_timenet_torch.data_io.market_residual (on {man.data_fingerprint})")
    written = sum(v["written"] for v in per_coin.values())
    lab = sum(v["label_rows"] for v in per_coin.values())
    record = {"source_artifact": str(src), "source_fingerprint": man.data_fingerprint,
              "fingerprint": manifest["export"]["data_sha256"], "channels": list(CHANNELS),
              "beta_bars": BETA_BARS, "beta_min": BETA_MIN, "market": "equal-weight close log move of the artifact's "
              "coins", "rows": int(len(ts)), "rows_with_channels": int(written),
              "rows_with_residual_label": int(lab), "label_rows_with_beta_1": n_unknown_beta,
              "label_var_own": float(np.var(y)), "label_var_residual": float(np.var(y_new)),
              "seconds": time.monotonic() - t0, "per_coin": per_coin}
    (dst / DERIVATION_JSON).write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def _vision_raw_candles(raw: Path, base_tf: str):
    from nig_timenet_torch.data_io.vision_slice import load_frames

    cache: dict = {}

    def get(name, start, end):
        if name not in cache:
            f = load_frames(raw, [name], base_tf)[name]
            cache[name] = f.set_index(pd.to_datetime(f["open_time"], utc=True))[[*TARGET_COLUMNS, "volume"]]
        return cache[name]

    return get


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nig_timenet_torch.data_io.market_residual", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--cache", default=None, help="candle cache folder (archive fetch)")
    ap.add_argument("--raw", default=None, help="a vision_slice raw folder instead of the archive")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    candles = None
    if a.raw:
        candles = _vision_raw_candles(Path(a.raw), npy_dir.load(a.src, mmap=True)["base_tf"])
    r = add_market_residual(a.src, a.dst, candles=candles, cache=a.cache, overwrite=a.overwrite)
    print(json.dumps({k: v for k, v in r.items() if k != "per_coin"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
