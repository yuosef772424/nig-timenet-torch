"""Raw candle sources. A source is a callable ``load(name) -> DataFrame`` indexed by UTC time with
open/high/low/close/volume. The ``files`` source reads the layout crypto-signal-prediction reads
(``<root>/<raw_dir>/<SYMBOL>.csv`` or ``.csv.gz``, columns ``datetime_utc`` or ``timestamp`` + OHLCV; a pattern ending
in ``.parquet`` reads the public Speirsy11 dataset files ``ohlc_<SYMBOL>_<interval>.parquet``); data.fetch downloads
that layout from the same two Binance sources. ``synthetic`` is a seeded market for tests and demos.
Adding a source is one function and one ``SOURCES`` registration."""

import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from nig_timenet_torch.core.registry import Contract, Registry
from nig_timenet_torch.data.coins import coins_of
from nig_timenet_torch.data.fetch import raw_path, read_registry
from nig_timenet_torch.data.settings import SourceSettings

__all__ = ["SOURCES", "make_loader", "resolve_coins", "read_candles", "synthetic_market",
           "normalize_tf", "resample_ohlcv"]

SOURCES = Registry("source", Contract("source", required_meta=("needs_path",)))
_COLUMNS = ["open", "high", "low", "close", "volume"]
_TIME_COLUMNS = ("datetime_utc", "timestamp", "open_time", "time", "date")
#: how base bars combine into a higher-timeframe bar (the old project's OHLCV_AGG, verbatim)
OHLCV_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
_TF_UNIT_FIX = {"d": "D", "w": "W"}


def normalize_tf(tf: str) -> str:
    """The ONE timeframe-string normaliser (the old project's core/timeframe.py): ``'1d'`` -> ``'1D'``,
    ``'2w'`` -> ``'2W'`` (pandas 3 deprecates the lowercase day/week spellings); anything else -
    ``'4h'``, ``'1D'``, ``'15min'`` - is returned unchanged. A non-string raises TypeError."""
    if not isinstance(tf, str):
        raise TypeError(f"timeframe must be a string, got {type(tf).__name__}: {tf!r}")
    m = re.match(r"^(\d+)([A-Za-z]+)$", tf)
    if m is None:
        return tf
    return m.group(1) + _TF_UNIT_FIX.get(m.group(2), m.group(2))


def resample_ohlcv(df: pd.DataFrame, tf: str, base: str = "1h") -> pd.DataFrame:
    """Re-aggregate `base`-timeframe OHLCV bars up to `tf` - what the old project's pipeline did when
    the owner picked 4h/1d: open first, high max, low min, close last, volume sum, and only bars built
    from the FULL count of base bars survive (the old project's 'closed' mode), so a partial trailing
    bar or a gap-filled bar never enters a window. A no-op when `tf` is the base itself; a target
    finer than the base raises - bars can be combined, never split."""
    step, base_step = pd.Timedelta(normalize_tf(tf)), pd.Timedelta(normalize_tf(base))
    if step == base_step:
        return df
    if step < base_step:
        raise ValueError(f"target timeframe {tf!r} is finer than the base {base!r}: bars cannot be split")
    want = int(step / base_step)
    grouped = df.resample(normalize_tf(tf))
    out = grouped.agg(OHLCV_AGG)
    full = grouped.size() >= want
    return out[full].dropna(how="all")


def resolve_coins(s: SourceSettings) -> list[str]:
    """The coin names to build: explicit `coins`, else the categories (or the whole registry for "all_global"),
    minus `exclude`; a coin listed twice is an error."""
    if s.coins:
        names = list(s.coins)
    elif tuple(s.categories) == ("all_global",):
        names = [r["name"] for r in read_registry(s)]
    else:
        names = coins_of(s.categories)
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"SourceSettings: coins listed twice: {dup}")
    drop = set(s.exclude)
    return [n for n in names if n not in drop]


def _clean(df: pd.DataFrame, name: str, s: SourceSettings) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        ts_col = next((c for c in _TIME_COLUMNS if c in df.columns), None)
        if ts_col is None:
            raise ValueError(f"[{name}] no time column ({'/'.join(_TIME_COLUMNS)}); columns: {list(df.columns)}")
        numeric = pd.api.types.is_numeric_dtype(df[ts_col])
        unit = "ms" if numeric and df[ts_col].abs().max() > 1e11 else None
        df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce", unit=unit)
        df = df.dropna(subset=["timestamp"]).set_index("timestamp")
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index.name = "timestamp"
    missing = [c for c in _COLUMNS[:4] if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] missing price columns {missing}; columns: {list(df.columns)}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if s.start:
        df = df[df.index >= pd.Timestamp(s.start, tz="UTC")]
    if s.end:
        df = df[df.index < pd.Timestamp(s.end, tz="UTC")]
    return df[_COLUMNS].astype("float64")


def read_candles(path: Path) -> pd.DataFrame:
    """One candle file by extension (.csv, .csv.gz, .parquet)."""
    if ".parquet" in path.suffixes:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _files_loader(s: SourceSettings) -> Callable[[str], pd.DataFrame]:
    root = Path(s.root) / s.raw_dir
    if not root.is_dir():
        raise FileNotFoundError(f"candle folder not found: {root} (SourceSettings.root / raw_dir)")

    def load(name: str) -> pd.DataFrame:
        path = raw_path(s, name)
        if not path.exists():
            alt = path.with_name(path.name[:-3] if path.name.endswith(".gz") else path.name + ".gz")
            if not alt.exists():
                raise FileNotFoundError(f"[{name}] candle file not found: {path} (also tried {alt.name})")
            path = alt
        return _clean(read_candles(path), name, s)
    return load


_STAGGER_HOURS = (0, 3, 5, 11, 24, 0)


def synthetic_market(names: list[str], days: int, seed: int, freq: str = "1h") -> dict[str, pd.DataFrame]:
    """Seeded random-walk candles, coin i with seed `seed + i` and a start staggered by a few hours (so the coins'
    bars do not all begin together). The same generator the reference project's tests use."""
    step = pd.Timedelta(freq)
    n = int(days * pd.Timedelta("1D") / step)
    out = {}
    for i, name in enumerate(names):
        rng = np.random.default_rng(seed + i)
        start = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=_STAGGER_HOURS[i % len(_STAGGER_HOURS)])
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
        op = np.r_[close[0], close[:-1]]
        out[name] = pd.DataFrame(
            {"open": op, "high": np.maximum(op, close) * (1 + rng.random(n) * 0.003),
             "low": np.minimum(op, close) * (1 - rng.random(n) * 0.003), "close": close,
             "volume": rng.random(n) * 1000 + 50}, index=pd.date_range(start, periods=n, freq=step))
    return out


def _synthetic_loader(s: SourceSettings) -> Callable[[str], pd.DataFrame]:
    names = resolve_coins(s)
    market = synthetic_market(names, s.synthetic_days, s.synthetic_seed, s.interval)
    return lambda name: _clean(market[name], name, s)


SOURCES.register("files", _files_loader, version=1, needs_path=True)
SOURCES.register("synthetic", _synthetic_loader, version=1, needs_path=False)


def make_loader(s: SourceSettings) -> Callable[[str], pd.DataFrame]:
    """The `load(name)` function of the configured source."""
    return SOURCES[s.kind](s)
