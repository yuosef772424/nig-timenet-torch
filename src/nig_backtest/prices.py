"""PURPOSE: Where the backtest gets RAW candles: an in-memory dict, a folder of per-coin files (the TF raw layout or
    parquet) or the public Binance archive with a disk cache; every source loads one coin on demand and caches it.
TAGS: price source, raw candles, ohlcv, lazy load, cache, binance archive, csv layout, atr, true range
PITFALLS: bars are indexed by their OPEN time (naive UTC) and must be on one interval per coin; the archive source
    needs the network on its first read of a coin and writes ``<cache_dir>/<interval>/<coin>.parquet`` (a later read
    of a wider range fetches again). ``atr`` at bar i uses bars <= i only; the engine reads it at the bar BEFORE the
    entry bar, so no exit level knows the entry bar's own range.
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from nig_backtest._base import OHLCV as OHLCV_COLUMNS
from nig_backtest.contract import to_utc_naive

__all__ = ["PriceSource", "FramePrices", "FilePrices", "ArchivePrices", "atr", "bar_interval", "vision_klines",
           "VISION"]

VISION = "https://data.binance.vision"
_KLINE_COLUMNS = ("open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades")

_OHLC = OHLCV_COLUMNS[:4]


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in _OHLC if c not in frame.columns]
    if missing:
        raise ValueError(f"price frame without {missing}")
    out = frame.copy()
    out.index = pd.DatetimeIndex(to_utc_naive(out.index).to_numpy())
    out = out[~out.index.duplicated(keep="last")].sort_index()
    cols = [c for c in OHLCV_COLUMNS if c in out.columns]
    return out[cols].astype("float64")


class PriceSource:
    """``bars(coin, start=None, end=None)``: the coin's candles with bar open time in [start, end) (None = all).
    Subclasses implement ``_load(coin) -> frame``; this class caches the last ``cache_coins`` coins."""

    def __init__(self, cache_coins: int = 64):
        self.cache_coins = int(cache_coins)
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()

    def _load(self, coin: str) -> pd.DataFrame:
        raise NotImplementedError

    def bars(self, coin: str, start=None, end=None) -> pd.DataFrame:
        if coin not in self._cache:
            self._cache[coin] = _clean(self._load(coin))
            while len(self._cache) > self.cache_coins:
                self._cache.popitem(last=False)
        self._cache.move_to_end(coin)
        f = self._cache[coin]
        if start is not None:
            f = f[f.index >= pd.Timestamp(start)]
        if end is not None:
            f = f[f.index < pd.Timestamp(end)]
        return f


class FramePrices(PriceSource):
    """Candles already in memory: {coin: frame indexed by bar open time}."""

    def __init__(self, frames: dict, **kw):
        super().__init__(**kw)
        self.frames = frames

    def _load(self, coin):
        if coin not in self.frames:
            raise KeyError(f"no candles for {coin!r}")
        return self.frames[coin]


class FilePrices(PriceSource):
    """One file per coin under ``root``: ``pattern`` with ``{coin}`` (default the TF raw layout ``{coin}.csv.gz``;
    a ``.parquet`` pattern reads parquet). The time comes from the index (parquet) or the ``timestamp`` (ms) /
    ``datetime_utc`` column (csv)."""

    def __init__(self, root, pattern: str = "{coin}.csv.gz", **kw):
        super().__init__(**kw)
        self.root, self.pattern = Path(root), pattern

    def _load(self, coin):
        path = self.root / self.pattern.format(coin=coin)
        if not path.exists():
            raise KeyError(f"no candle file for {coin!r}: {path}")
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        f = pd.read_csv(path)
        t = pd.to_datetime(f["timestamp"], unit="ms") if "timestamp" in f else pd.to_datetime(f["datetime_utc"])
        return f.set_index(t)


class _Http:
    def get_bytes(self, url: str) -> bytes | None:
        import urllib.error
        import urllib.request
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise


def _zip_frame(blob: bytes | None) -> pd.DataFrame:
    if blob is None:
        return pd.DataFrame(columns=list(_KLINE_COLUMNS))
    import io
    import zipfile
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        text = z.read(z.namelist()[0]).decode("utf-8")
    rows = [r.split(",")[:len(_KLINE_COLUMNS)] for r in text.splitlines() if r[:1].isdigit()]
    return pd.DataFrame(rows, columns=list(_KLINE_COLUMNS))


def vision_klines(symbol: str, interval: str, start, end, http=None) -> pd.DataFrame:
    """Candles of a USDT-perpetual from the public Binance archive (monthly zips, daily zips for a month without
    one) with bar open time in [start, end), indexed by that time (naive UTC). ``http`` = an object with
    ``get_bytes(url) -> bytes | None`` (tests pass a fake)."""
    http = http or _Http()
    lo, hi = (pd.Timestamp(t).tz_localize(None) if pd.Timestamp(t).tzinfo else pd.Timestamp(t) for t in (start, end))
    parts = []
    for month in pd.period_range(lo, hi - pd.Timedelta(microseconds=1), freq="M"):
        tag = f"{symbol}-{interval}-{month.year:04d}-{month.month:02d}"
        blob = http.get_bytes(f"{VISION}/data/futures/um/monthly/klines/{symbol}/{interval}/{tag}.zip")
        if blob is not None:
            parts.append(_zip_frame(blob))
            continue
        for day in pd.date_range(month.start_time, month.end_time.normalize(), freq="D"):
            url = f"{VISION}/data/futures/um/daily/klines/{symbol}/{interval}/{tag}-{day.day:02d}.zip"
            parts.append(_zip_frame(http.get_bytes(url)))
    f = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(_KLINE_COLUMNS))
    t = pd.to_datetime(f["open_time"].astype("int64"), unit="ms")
    f = f.assign(t=t).drop_duplicates("t").sort_values("t")
    f = f[(f["t"] >= lo) & (f["t"] < hi)]
    out = f[["open", "high", "low", "close", "volume", "quote_volume", "trades"]].astype(float)
    out.index = pd.DatetimeIndex(f["t"].to_numpy())
    return out


class ArchivePrices(PriceSource):
    """The public Binance futures archive (``vision_klines``) between ``start`` and ``end``, cached as
    parquet under ``cache_dir`` (None = memory only)."""

    def __init__(self, interval: str = "4h", start="2020-01-01", end=None, cache_dir=None, http=None, **kw):
        super().__init__(**kw)
        self.interval, self.start, self.http = interval, start, http
        self.end = end or pd.Timestamp.now(tz="UTC").ceil("D").tz_localize(None)
        self.cache_dir = None if cache_dir is None else Path(cache_dir) / interval

    def _load(self, coin):
        path = None if self.cache_dir is None else self.cache_dir / f"{coin}.parquet"
        if path is not None and path.exists():
            return pd.read_parquet(path)
        frame = vision_klines(coin, self.interval, self.start, self.end, self.http)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path)
        return frame


def bar_interval(frame: pd.DataFrame) -> pd.Timedelta:
    """The coin's bar length (the most common gap between bar open times)."""
    if len(frame) < 2:
        raise ValueError("a price frame needs >= 2 bars to know its interval")
    gaps = np.diff(frame.index.to_numpy()).astype("timedelta64[ns]")
    vals, counts = np.unique(gaps, return_counts=True)
    return pd.Timedelta(vals[np.argmax(counts)])


def atr(frame: pd.DataFrame, n: int = 14) -> np.ndarray:
    """Wilder ATR per bar (true range smoothed with alpha 1 / n); NaN until ``n`` bars exist."""
    h, lo, c = (frame[k].to_numpy() for k in _OHLC[1:])
    prev = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.stack([h - lo, np.abs(h - prev), np.abs(lo - prev)]), axis=0)
    out = pd.Series(tr).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().to_numpy()
    return out
