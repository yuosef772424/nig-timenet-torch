"""PURPOSE: A REAL-data stand-in for the owner's Drive artifact, buildable anywhere: public Binance USDT-perp 4h klines
    -> a contract artifact with the real artifact's definition (log-move inputs, log-return labels).
TAGS: real data, binance vision, klines, logret artifact, contract artifact, doctor, e3 source, cpu readiness run
PITFALLS: sealed coins (data.lock) are refused and bars at or after the time seal are dropped, so the holdout stays
    clean; this is NOT the owner's artifact (other coins and history, no TF pipeline): numbers measured on it describe
    the method, not the recorded baseline. A month the archive does not have is skipped (the coin was not listed);
    a hole inside a coin's history splits it into segments, and no window or label crosses a hole.

Why it exists: the sandbox has no Drive and no GPU, so until now every check ran on synthetic E1 or on a 6-coin
fixture. With this module the whole diagnostic loop (docs/audit/*/cards) runs on real candles on a CPU.

``python -m nig_timenet_torch.data_io.vision_slice OUT [--raw DIR] [--coins A,B] [--start 2021-01] [--end 2026-09]
[--interval 4h] [--stride 1]`` downloads (resuming: a month already on disk is not fetched again) and writes the
artifact; ``build_logret_dataset`` is the pure part (frames -> dataset dict), tested without the network.
"""

import argparse
import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import LAST_COLUMNS, OHLCV_COLUMNS, TARGET_COLUMNS, TS_COL
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS, anchor_and_future, own_anchor_and_future
from nig_timenet_torch.model.heads import head_spec

__all__ = ["DEFAULT_COINS", "VISION", "fetch_month", "fetch_coin", "load_frames", "build_logret_dataset", "main"]

VISION = "https://data.binance.vision/data/futures/um/monthly/klines"
#: liquid USDT perpetuals that are NOT on the sealed holdout list (checked again at build time)
DEFAULT_COINS = tuple(f"{c}USDT" for c in (
    "BTC ETH BNB SOL XRP ADA DOGE AVAX LINK DOT LTC BCH TRX ATOM NEAR UNI ETC FIL XLM AAVE ALGO SAND MANA AXS EOS "
    "XTZ THETA VET ICP GRT APT ARB OP SUI INJ RUNE DYDX CRV SNX COMP ZEC DASH ENJ CHZ KAVA GALA IMX LDO FET EGLD "
    "HBAR QNT NEO IOTA ZIL BAT ONE CELO ANKR SUSHI").split())
_COLS = ("open_time", *OHLCV_COLUMNS)
_LABEL_CLIP = 1.0   # the TF export clips |log(future / last_close)| at 1.0 (B-054)


def fetch_month(coin: str, interval: str, month: pd.Period, raw: Path, get=None) -> Path | None:
    """One monthly kline zip -> ``raw/<coin>/<coin>-<interval>-YYYY-MM.csv`` (kept if already there). None when the
    archive has no such month (HTTP 404). ``get(url) -> bytes | None`` is injectable for tests."""
    tag = f"{coin}-{interval}-{month.year:04d}-{month.month:02d}"
    path = raw / coin / f"{tag}.csv"
    if path.exists():
        return path
    if (raw / coin / f"{tag}.missing").exists():
        return None
    if get is None:
        import requests

        def get(url):
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
    blob = get(f"{VISION}/{coin}/{interval}/{tag}.zip")
    path.parent.mkdir(parents=True, exist_ok=True)
    if blob is None:
        (raw / coin / f"{tag}.missing").write_text("")
        return None
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        path.write_bytes(z.read(z.namelist()[0]))
    return path


def fetch_coin(coin: str, interval: str, start: str, end: str, raw: Path, get=None) -> int:
    """Every month of [start, end] for one coin; returns how many monthly files exist afterwards."""
    months = pd.period_range(pd.Period(start, "M"), pd.Period(end, "M"), freq="M")
    return sum(fetch_month(coin, interval, m, raw, get) is not None for m in months)


def load_frames(raw: Path, coins, interval: str) -> dict[str, pd.DataFrame]:
    """{coin: frame(open_time ns UTC, open, high, low, close, volume)} from the monthly files, sorted, de-duplicated,
    rows with a non-positive price or high < low dropped. Coins without a file are left out."""
    out = {}
    for coin in coins:
        files = sorted((raw / coin).glob(f"{coin}-{interval}-*.csv")) if (raw / coin).is_dir() else []
        parts = []
        for f in files:
            df = pd.read_csv(f, header=None, usecols=range(6), names=_COLS, dtype=str)
            df = df[df["open_time"].str.isdigit()]                      # newer archives carry a header row
            parts.append(df.astype({c: np.float64 for c in _COLS[1:]}).astype({"open_time": np.int64}))
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
        t = df["open_time"].to_numpy()
        df["open_time"] = np.where(t > 10 ** 14, t // 1000, t) * 1_000_000   # ms (or us in 2025+ files) -> ns
        ok = (df[list(OHLCV_COLUMNS[:4])] > 0).all(axis=1) & (df["high"] >= df["low"])
        out[coin] = df[ok].reset_index(drop=True)
    return out


def _segments(t: np.ndarray, bar_ns: int) -> list[tuple[int, int]]:
    """[start, end) index ranges of gap-free runs of bar open times."""
    cuts = np.flatnonzero(np.diff(t) != bar_ns) + 1
    edges = [0, *cuts.tolist(), len(t)]
    return list(zip(edges[:-1], edges[1:], strict=True))


def build_logret_dataset(frames: dict[str, pd.DataFrame], *, interval: str = "4h", window: int = 32, horizon: int = 2,
                         stride: int = 1, time_seal=None) -> dict:
    """The dataset dict of the real artifact's definition from raw candles (one frame per coin, see ``load_frames``):

    * ``X_<interval>`` (N, window, 3) float32 = per bar [log(high / prev close), log(low / prev close),
      log(close / prev close)] (RAW: ``price_norm_state`` "unfitted", as the TF export);
    * ``y_<t>_reg`` = clip(log(future price / last close), +-1), future = max high / min low over the next ``horizon``
      bars, close after ``horizon`` bars; ``y_<t>_class`` = own future price above own last price;
    * ``last_candles`` (LAST_COLUMNS, timestamp = OPEN time of the window's last bar, ns), ``base_params`` = (median,
      IQR) of the window's closes, ``asset_bounds`` coin-major.

    A window and its label never cross a hole in the coin's bars; bars opening at or after ``time_seal`` are dropped."""
    bar_ns = int(pd.Timedelta(interval).value)
    xs, lasts, bases, bounds, n = [], [], [], [], 0
    for coin in sorted(frames):
        df = frames[coin]
        if time_seal is not None:
            df = df[df["open_time"] + (horizon + 1) * bar_ns <= pd.Timestamp(time_seal).value]
        t = df["open_time"].to_numpy()
        h, lo, c = (np.log(df[k].to_numpy()) for k in TARGET_COLUMNS)
        cx, cl, cb = [], [], []
        for a, b in _segments(t, bar_ns):
            # the window's last bar e needs `window` bars + one previous close before it and `horizon` bars after it
            ends = np.arange(a + window, b - horizon, stride)
            if not len(ends):
                continue
            prev = c[a:b - 1]
            feats = np.stack([h[a + 1:b] - prev, lo[a + 1:b] - prev, c[a + 1:b] - prev], axis=1)   # bars a+1 .. b-1
            win = np.lib.stride_tricks.sliding_window_view(feats, window, axis=0)                  # (.., 3, window)
            cx.append(win[ends - a - window].transpose(0, 2, 1))
            fut = ends[:, None] + np.arange(1, horizon + 1)
            last = np.zeros((len(ends), len(LAST_COLUMNS)))
            for name, v in (("last_high", h[ends]), ("last_low", lo[ends]), ("last_close", c[ends]),
                            ("future_close", c[ends + horizon]), ("future_low_min", lo[fut].min(axis=1)),
                            ("future_high_max", h[fut].max(axis=1))):
                last[:, LAST_COLUMNS.index(name)] = np.exp(v)
            last[:, TS_COL] = t[ends].astype(np.float64)
            closes = np.exp(np.lib.stride_tricks.sliding_window_view(c[a + 1:b], window)[ends - a - window])
            q1, med, q3 = np.quantile(closes, [0.25, 0.5, 0.75], axis=1)
            cl.append(last)
            cb.append(np.stack([med, q3 - q1], axis=1))
        if not cx:
            continue
        rows = sum(len(a) for a in cx)
        xs += cx
        lasts += cl
        bases += cb
        bounds.append({"name": coin, "start": n, "end": n + rows})
        n += rows
    if not bounds:
        raise ValueError("no coin has enough gap-free bars for one window")
    lc = np.concatenate(lasts)
    targets = list(TARGET_COLUMNS)
    ds = {"base_params": np.concatenate(bases).astype(np.float32), "last_candles": lc, "mode": "direction_robust",
          "scaler_type": "robust", "feature_order": list(TARGET_COLUMNS), "timeframes": [interval],
          "targets": [head_spec(kind, t).name for t in targets for kind in ("binary_classification",
                                                                               "nig_regression")],
          "price_targets": targets, "asset_bounds": bounds, "base_tf": interval, "window_sizes": {interval: window},
          "stride": stride, "forecast_horizon": horizon, "reg_target_mode": "return", "reg_target_scale": 1.0,
          "reg_target_clip": _LABEL_CLIP, "split_dates": None, "holdout_start": None, "skipped_assets": {},
          "price_norm_mode": "logret_std", "price_norm_state": "unfitted",
          f"X_{interval}": np.concatenate(xs).astype(np.float32)}
    codec = TARGET_CODECS["logret_close"]()
    state = codec.fit(ds)
    for t in targets:
        own = own_anchor_and_future(lc, t)
        ds[head_spec("binary_classification", t).label_key] = (own[:, 1] > own[:, 0]).astype(np.float32)
        ds[head_spec("nig_regression", t).label_key] = codec.encode(anchor_and_future(lc, t, state), state).astype(
            np.float32)
    ds["real_slice"] = {"source": VISION, "interval": interval, "window": window, "horizon": horizon, "stride": stride,
                        "coins": [b["name"] for b in bounds]}
    return ds


def main(argv: list[str] | None = None) -> int:
    from nig_timenet_torch.data.artifact_writer import write_artifact
    from nig_timenet_torch.data.lock import is_sealed, load_lock

    ap = argparse.ArgumentParser(prog="python -m nig_timenet_torch.data_io.vision_slice",
                                 description="Download public klines and write a real-data contract artifact.")
    ap.add_argument("out")
    ap.add_argument("--raw", default="", help="folder of the monthly csv files (default: OUT + '.raw')")
    ap.add_argument("--coins", default="", help="comma-separated symbols (default: DEFAULT_COINS)")
    ap.add_argument("--start", default="2021-01")
    ap.add_argument("--end", default="2026-09")
    ap.add_argument("--interval", default="4h")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-fetch", action="store_true", help="build from the files already in --raw")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    lock = load_lock()
    coins = [c.strip().upper() for c in a.coins.split(",") if c.strip()] or list(DEFAULT_COINS)
    sealed = [c for c in coins if is_sealed(c, lock)]
    if sealed:
        raise SystemExit(f"sealed holdout coins are never built: {sealed}")
    raw = Path(a.raw or a.out.rstrip("/") + ".raw")
    if not a.no_fetch:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            got = list(pool.map(lambda c: fetch_coin(c, a.interval, a.start, a.end, raw), coins))
        print(f"fetched: {sum(got)} monthly files for {sum(g > 0 for g in got)} of {len(coins)} coins", flush=True)
    ds = build_logret_dataset(load_frames(raw, coins, a.interval), interval=a.interval, window=a.window,
                              horizon=a.horizon, stride=a.stride, time_seal=lock.time_seal)
    man = write_artifact(ds, a.out, source_commit="", overwrite=a.overwrite,
                         producer="nig_timenet_torch.data_io.vision_slice")
    print(f"wrote {a.out}: {man['export']['rows']} rows, {len(man['export']['coins'])} coins, "
          f"max_bar_time {man['export']['max_bar_time']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
