"""PURPOSE: One compressed file of raw candles for many coins: built once from any price source (archive, files, the
    artifact through a bridge), then loaded in ONE read, keeping only the chosen coins; the prices stay raw.
TAGS: price store, compressed, parquet zstd, one file, all coins, load once, select coins, raw candles, no decoding
PITFALLS: the store holds candles exactly as the source gave them (no codec, no scaling); one interval per store
    (written in its metadata and checked on load); building a big store needs the source's network / disk once,
    loading needs neither. A coin the source cannot give is listed in the metadata, not written.
"""

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from nig_backtest.prices import PriceSource, bar_interval

__all__ = ["build_store", "StorePrices", "store_info"]

_META = b"nig_backtest_store"


def build_store(source: PriceSource, coins, path, *, start=None, end=None, compression: str = "zstd") -> dict:
    """Write the candles of ``coins`` (bar open time in [start, end)) from ``source`` into ONE parquet file; returns
    the metadata (coins written, coins missing, interval, rows)."""
    parts, missing, interval = [], [], None
    for coin in coins:
        try:
            f = source.bars(coin, start, end)
        except KeyError:
            missing.append(coin)
            continue
        if len(f) < 2:
            missing.append(coin)
            continue
        step = str(bar_interval(f))
        if interval not in (None, step):
            raise ValueError(f"{coin}: interval {step} differs from the store's {interval}")
        interval = step
        parts.append(f.reset_index(names="time").assign(coin=coin))
    if not parts:
        raise ValueError("build_store: no coin had candles")
    frame = pd.concat(parts, ignore_index=True)
    meta = {"interval": interval, "coins": sorted(frame["coin"].unique()), "missing": missing, "rows": len(frame)}
    table = pa.Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), _META: json.dumps(meta).encode()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression=compression)
    return meta


def store_info(path) -> dict:
    """The metadata of a store (interval, coins, missing coins, rows) without reading the candles."""
    return json.loads(pq.read_schema(path).metadata[_META])


class StorePrices(PriceSource):
    """All candles of a store read at once (only ``coins`` when given) and served from memory."""

    def __init__(self, path, coins=None, **kw):
        super().__init__(cache_coins=10 ** 9, **kw)
        self.info = store_info(path)
        filters = None if coins is None else [("coin", "in", [str(c) for c in coins])]
        frame = pd.read_parquet(path, filters=filters)
        self.frames = {c: g.drop(columns="coin").set_index("time") for c, g in frame.groupby("coin")}

    def _load(self, coin):
        if coin not in self.frames:
            raise KeyError(f"no candles for {coin!r} in the store")
        return self.frames[coin]
