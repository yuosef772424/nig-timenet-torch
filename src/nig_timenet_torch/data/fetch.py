"""Binance candle download

PURPOSE: Download raw candles (Binance USDT-perpetual API and the public data.binance.vision archive) into the
    layout the old project reads.
TAGS: fetch, binance, klines, csv layout, asset registry, resume
PITFALLS: network code, not exercised by CI (tests use a fake Http); files resume from the last row, so a corrupt
    last row must be removed by hand.

Download raw candles into the layout crypto-signal-prediction reads, with the same file and column names:

    <root>/<raw_dir>/<SYMBOL>.csv[.gz]       timestamp (ms), datetime_utc, open, high, low, close, volume,
                                             quote_volume, trades, taker_buy_volume, taker_buy_quote_volume
    <root>/crypto_data/asset_registry.csv    one row per coin (name, status, first/last candle, gaps, ...)

Two sources, the same two the reference tools use: the live USDT-perpetual futures API (fapi.binance.com
``/fapi/v1/klines``) and the public archive data.binance.vision (monthly zips, daily zips for the open month).
Both resume from the last row already in a file. All network access goes through ``Http`` so tests inject a fake.
"""

import csv
import gzip
import io
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from nig_timenet_torch.core.schema import OHLCV_COLUMNS
from nig_timenet_torch.data.settings import SourceSettings
from nig_timenet_torch.data.timeframe import tf_timedelta

__all__ = ["CSV_HEADER", "Http", "kline_row", "fmt_dt", "raw_path", "fetch_symbol", "fetch_all", "update_registry",
           "file_stats", "read_registry"]

CSV_HEADER = ["timestamp", "datetime_utc", *OHLCV_COLUMNS, "quote_volume", "trades",
              "taker_buy_volume", "taker_buy_quote_volume"]
FAPI = "https://fapi.binance.com"
VISION = "https://data.binance.vision"
PAGE = 1500
REGISTRY_COLUMNS = ["name", "status", "onboard_date_utc", "first_candle_utc", "last_candle_utc", "interval",
                    "n_candles", "n_gaps", "missing_candles", "max_gap_hours", "zero_volume_candles",
                    "bad_ohlc_rows", "extended_columns", "file"]


class Http:
    """The two requests the downloaders make. `get_bytes` returns None for a missing (404) object."""

    def get_json(self, url: str, params: dict | None = None):
        import requests
        r = requests.get(url, params=params, timeout=25)
        r.raise_for_status()
        return r.json()

    def get_bytes(self, url: str) -> bytes | None:
        import requests
        r = requests.get(url, timeout=60)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.content


def fmt_dt(ms: int) -> str:
    return pd.Timestamp(int(ms), unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M:%S")


def kline_row(k: list) -> list:
    """Binance kline [openTime, o, h, l, c, v, closeTime, quoteVol, trades, takerBase, takerQuote, _] -> CSV row."""
    return [int(k[0]), fmt_dt(int(k[0])), k[1], k[2], k[3], k[4], k[5], k[7], k[8], k[9], k[10]]


def raw_path(s: SourceSettings, name: str) -> Path:
    return Path(s.root) / s.raw_dir / s.pattern.format(name=name, interval=s.interval)


def _open_text(path: Path, mode: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", newline="", encoding="utf-8")
    return open(path, mode, newline="", encoding="utf-8")


def _last_ts(path: Path) -> int | None:
    if not path.exists():
        return None
    last = None
    with _open_text(path, "r") as f:
        for row in csv.reader(f):
            if row and row[0].isdigit():
                last = int(row[0])
    return last


def _append(path: Path, rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with _open_text(path, "a") as f:
        w = csv.writer(f)
        if new:
            w.writerow(CSV_HEADER)
        w.writerows(rows)


def _ms(text: str) -> int:
    return int(pd.Timestamp(text, tz="UTC").timestamp() * 1000) if text else 0


def _futures_rows(http: Http, symbol: str, interval: str, lo: int, hi: int) -> list[list]:
    rows, cur = [], lo
    while cur < hi:
        params = {"symbol": symbol, "interval": interval, "startTime": cur, "limit": PAGE}
        if hi < 2 ** 62:
            params["endTime"] = hi - 1
        batch = http.get_json(f"{FAPI}/fapi/v1/klines", params)
        if not batch:
            break
        rows.extend(kline_row(k) for k in batch)
        cur = int(batch[-1][0]) + 1
        if len(batch) < PAGE:
            break
    return rows


def _zip_rows(blob: bytes | None) -> list[list]:
    if blob is None:
        return []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        text = z.read(z.namelist()[0]).decode("utf-8")
    return [kline_row(r) for r in csv.reader(io.StringIO(text)) if r and r[0].isdigit()]


def _vision_rows(http: Http, symbol: str, interval: str, lo: int, hi: int) -> list[list]:
    rows = []
    for month in pd.period_range(pd.Timestamp(lo, unit="ms"), pd.Timestamp(hi - 1, unit="ms"), freq="M"):
        tag = f"{symbol}-{interval}-{month.year:04d}-{month.month:02d}"
        blob = http.get_bytes(f"{VISION}/data/futures/um/monthly/klines/{symbol}/{interval}/{tag}.zip")
        if blob is not None:
            rows.extend(_zip_rows(blob))
            continue
        for day in pd.date_range(month.start_time, month.end_time.normalize(), freq="D"):
            blob = http.get_bytes(f"{VISION}/data/futures/um/daily/klines/{symbol}/{interval}/{tag}-{day.day:02d}.zip")
            rows.extend(_zip_rows(blob))
    return rows


def fetch_symbol(s: SourceSettings, name: str, http: Http | None = None) -> int:
    """Download `name` (resuming after the last stored row); returns how many rows were added."""
    http = http or Http()
    path = raw_path(s, name)
    last = _last_ts(path)
    lo = max(_ms(s.start), (last + 1) if last is not None else 0)
    hi = _ms(s.end) or 2 ** 62
    fn = _futures_rows if s.fetch == "futures" else _vision_rows
    if fn is _vision_rows:
        hi = min(hi, int(pd.Timestamp.now(tz="UTC").timestamp() * 1000))
    rows = [r for r in fn(http, name, s.interval, lo, hi) if lo <= r[0] < hi]
    rows.sort(key=lambda r: r[0])
    if rows:
        _append(path, rows)
    return len(rows)


def file_stats(path: Path, interval_ms: int) -> dict:
    """Quality counters of one candle file, read from the file itself."""
    n = gaps = missing = zero_vol = bad = 0
    first = last = prev = None
    max_gap = 0
    with _open_text(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None) or []
        idx = {c: i for i, c in enumerate(header)}
        for row in reader:
            try:
                ts = int(row[0])
                o, h, lo, c, v = (float(row[idx[k]]) for k in OHLCV_COLUMNS)
            except (ValueError, IndexError, KeyError):
                bad += 1
                continue
            n += 1
            first = ts if first is None else first
            last = ts
            if prev is not None and ts - prev > interval_ms:
                gaps += 1
                missing += (ts - prev) // interval_ms - 1
                max_gap = max(max_gap, ts - prev)
            prev = ts
            zero_vol += v == 0
            bad += not (lo <= min(o, c) and max(o, c) <= h and lo > 0)
        extended = header == CSV_HEADER
    return {"n_candles": n, "first_candle_utc": fmt_dt(first) if first else "",
            "last_candle_utc": fmt_dt(last) if last else "", "n_gaps": gaps, "missing_candles": missing,
            "max_gap_hours": round(max_gap / 3_600_000, 2), "zero_volume_candles": zero_vol, "bad_ohlc_rows": bad,
            "extended_columns": extended}


def read_registry(s: SourceSettings) -> list[dict]:
    path = Path(s.root) / s.registry_path
    if not path.exists():
        raise FileNotFoundError(f"asset registry not found: {path}")
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if r.get("name")]


def update_registry(s: SourceSettings, names: list[str]) -> int:
    """Rebuild the registry rows of `names` from their files; other rows and extra columns are kept."""
    path = Path(s.root) / s.registry_path
    rows = {r["name"]: r for r in read_registry(s)} if path.exists() else {}
    extra = []
    if path.exists():
        with open(path, newline="", encoding="utf-8-sig") as f:
            extra = [c for c in (csv.DictReader(f).fieldnames or []) if c and c not in REGISTRY_COLUMNS]
    ms = int(tf_timedelta(s.interval).total_seconds() * 1000)
    for name in names:
        p = raw_path(s, name)
        if p.exists():
            rows[name] = {**rows.get(name, {"name": name}), "interval": s.interval, "file": p.name,
                          **file_stats(p, ms)}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS + extra, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows[k] for k in sorted(rows))
    os.replace(tmp, path)
    return len(rows)


def fetch_all(s: SourceSettings, names: list[str], http: Http | None = None) -> dict[str, int]:
    """Download every coin of `names` (`s.fetch` picks the source) and refresh the registry."""
    if s.fetch not in ("futures", "vision"):
        raise ValueError(f"SourceSettings.fetch must be 'futures' or 'vision', got {s.fetch!r}")
    http = http or Http()
    with ThreadPoolExecutor(max_workers=max(1, min(s.fetch_workers, len(names)))) as pool:
        added = dict(zip(names, pool.map(lambda n: fetch_symbol(s, n, http), names), strict=True))
    update_registry(s, names)
    return added
