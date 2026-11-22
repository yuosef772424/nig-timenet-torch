"""Derived-artifact pipeline

PURPOSE: What the five derived-artifact tools share: load and check the source, the per-coin candle cache and thread
    pool, writing the new artifact with its derivation json, the common command-line flags.
TAGS: derived artifact, derive pipeline, candle cache, per-coin pool, derivation json, write artifact, argparse
PITFALLS: the candle cache is one parquet per coin and frame (``<symbol>_<tf>.parquet``) that remembers the range it
    was fetched for; a file that is narrower than the columns asked or does not cover the asked range is refetched
    and replaced, never used (a stale cache would silently give zero rows); the derivation record starts with the
    same three keys in every tool (source_artifact, source_fingerprint, fingerprint) and is written AFTER the
    artifact, so it is not in the manifest's file hashes. The tools keep their old ``data_io.<tool>`` producer
    strings so a rewritten manifest is byte-identical (HB5).
"""

import argparse
import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from nig_timenet_torch.core.schema import QUOTE_VOLUME_COL, TARGET_COLUMNS
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.data_io.artifact import ArtifactManifest, read_manifest
from nig_timenet_torch.data_io.artifact_writer import write_artifact

__all__ = ["CANDLE_COLS", "load_source", "require_one_frame", "cache_dir", "coin_candles", "per_coin_map",
           "write_derived", "make_parser", "print_record"]

#: high, low, close (core.schema) + the quote volume: what the indicator tools read from a coin's candles
CANDLE_COLS = (*TARGET_COLUMNS, QUOTE_VOLUME_COL)
_RANGE_KEYS = ("cache_start", "cache_end")


def load_source(src) -> tuple[ArtifactManifest, dict]:
    """(manifest, dataset) of the source artifact ``src`` (arrays read into memory)."""
    return read_manifest(src), npy_dir.load(src, mmap=False)


def require_one_frame(ds: dict, src, hint: str) -> None:
    """Raise unless ``ds`` holds exactly one timeframe; ``hint`` says what the tool needs instead."""
    if len(ds["timeframes"]) != 1:
        raise ValueError(f"{src} holds {list(ds['timeframes'])}: {hint}")


def cache_dir(cache) -> Path | None:
    """``cache`` as a Path with the folder created (None stays None)."""
    if cache is None:
        return None
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _cache_covers(frame: pd.DataFrame, cols: Sequence[str], start, end) -> bool:
    """True when a cached frame has every asked column and was fetched for a range that holds [start, end]."""
    if not set(cols) <= set(frame.columns) or not all(k in frame.attrs for k in _RANGE_KEYS):
        return False
    return _utc(frame.attrs["cache_start"]) <= _utc(start) and _utc(frame.attrs["cache_end"]) >= _utc(end)


def coin_candles(symbol: str, tf: str, start, end, cache: Path | None, fetch,
                 cols: Sequence[str] = CANDLE_COLS) -> pd.DataFrame:
    """The ``cols`` of ``symbol``'s ``tf`` candles from ``start`` to ``end`` (indexed by bar open time): from the
    cache file ``cache/<symbol>_<tf>.parquet`` when it has every column and covers the range, else
    ``fetch(symbol, tf, start, end)`` (and the cache file is replaced, written tmp + rename so a cut run resumes)."""
    cols = list(cols)
    path = None if cache is None else cache / f"{symbol}_{tf}.parquet"
    if path is not None and path.is_file():
        cached = pd.read_parquet(path)
        if _cache_covers(cached, cols, start, end):
            return cached[cols]
    frame = fetch(symbol, tf, start, end)[cols]
    if path is not None:
        out = frame.copy()
        out.attrs = {"cache_start": _utc(start).isoformat(), "cache_end": _utc(end).isoformat()}
        tmp = path.with_suffix(".tmp")
        out.to_parquet(tmp)
        tmp.rename(path)
    return frame


def per_coin_map(ds: dict, fn: Callable[[dict], tuple], *, workers: int,
                 progress_every: int = 25) -> tuple[list[tuple], float]:
    """``fn(asset_bounds entry)`` for every coin of ``ds`` in a thread pool (results in coin order) with a progress
    line every ``progress_every`` coins. Returns (results, start time of ``time.monotonic``)."""
    t0 = time.monotonic()
    bounds = ds["asset_bounds"]
    results = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, res in enumerate(pool.map(fn, bounds)):
            results.append(res)
            if (i + 1) % progress_every == 0:
                print(f"{i + 1}/{len(bounds)} coins, {time.monotonic() - t0:.0f} s", flush=True)
    return results, t0


def write_derived(src, dst, man: ArtifactManifest, ds: dict, *, producer: str, derivation_json: str, record: dict,
                  overwrite: bool = False) -> dict:
    """Write ``ds`` as the contract artifact ``dst`` (the source's commit), then ``dst/<derivation_json>``: the
    source artifact path, source and new fingerprint, then ``record``. Returns the full derivation record."""
    manifest = write_artifact(ds, dst, source_commit=man.source_commit, overwrite=overwrite, producer=producer)
    full = {"source_artifact": str(src), "source_fingerprint": man.data_fingerprint,
            "fingerprint": manifest["export"]["data_sha256"], **record}
    (Path(dst) / derivation_json).write_text(json.dumps(full, indent=1), encoding="utf-8")
    return full


def make_parser(prog: str, doc: str | None, *, cache: bool = True, workers: bool = True) -> argparse.ArgumentParser:
    """The flags every derived-artifact tool has: SRC DST [--cache DIR] [--workers N] [--overwrite]."""
    ap = argparse.ArgumentParser(prog=prog, description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    if cache:
        ap.add_argument("--cache", default=None)
    if workers:
        ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    return ap


def print_record(record: dict) -> None:
    """Print the derivation record without its per-coin table."""
    print(json.dumps({k: v for k, v in record.items() if k != "per_coin"}, indent=1))
