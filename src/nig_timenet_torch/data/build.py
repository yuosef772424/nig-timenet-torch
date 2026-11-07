"""Dataset build

PURPOSE: Turn the raw candles of many coins into the dataset dict every other module reads, with two passes so
    cross-asset columns see the whole universe.
TAGS: dataset build, two pass, cross asset, windows, skipped assets, feature order
PITFALLS: sealed coins are never loaded; a too-short coin is recorded in skipped_assets with its reason, not
    silently dropped.

Raw candles of many coins -> the dataset dict every other module reads (the same keys as the npy_dir format).

Two passes, like the reference pipeline: a light first pass resamples high/low/close of every coin so the
cross-asset columns (breadth, momentum rank, orthogonal momentum) can see the whole universe; the second pass builds
each coin's indicators, attaches the market and cross-asset columns and cuts windows (data.windows). Sealed coins
(data.lock) are never loaded. A coin that is too short is recorded in ``skipped_assets`` with the reason.
"""

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.prediction_schema import OUTPUT_PREFIX
from nig_timenet_torch.core.schema import TARGET_COLUMNS
from nig_timenet_torch.data.align import HLC_AGG, is_closed_mode, keep_full_bars, resample_frames
from nig_timenet_torch.data.context import (
    add_breadth,
    add_market_context,
    add_orth,
    add_rank,
    build_breadth,
    build_orth,
    build_rank,
    cross_columns,
    market_return_frame,
)
from nig_timenet_torch.data.fetch import fetch_all
from nig_timenet_torch.data.indicators import add_indicators, base_feature_columns
from nig_timenet_torch.data.lock import HoldoutLock, is_sealed, load_lock
from nig_timenet_torch.data.settings import DataSettings
from nig_timenet_torch.data.sources import make_loader, resolve_coins
from nig_timenet_torch.data.windows import AssetWindows, head_names, prepare_asset

__all__ = ["BuildContext", "feature_order", "build_dataset", "DATASET_MODE"]

#: The reference pipeline stamps every dataset with this string (its feature-direction mode); kept so a dataset
#: built here has the same keys as one built there.
DATASET_MODE = "direction_robust"


def feature_order(settings: DataSettings) -> list[str]:
    """Columns of X in order: per-coin indicators/custom features first, then market, rank, breadth, orth."""
    f, w = settings.features, settings.windows
    cols = base_feature_columns(f, w.tf_order) + cross_columns(f)
    drop = set(f.exclude_from_features)
    out = [c for c in cols if c not in drop]
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate feature columns: {sorted({c for c in out if out.count(c) > 1})}")
    return out


@dataclass
class BuildContext:
    """Everything the per-coin stage needs from the whole universe."""
    market: dict | None = None
    breadth: dict | None = None
    rank: dict | None = None
    orth: dict | None = None
    reference: str = ""


@dataclass
class _Outcome:
    name: str
    result: AssetWindows | None = None
    reason: str = ""
    extra: dict = field(default_factory=dict)


def _norm(name: str) -> str:
    return str(name).strip().upper()


def _frames(raw: pd.DataFrame, settings: DataSettings) -> dict[str, pd.DataFrame]:
    """Raw candles -> {tf: frame with indicators}, ready for the cross-asset columns."""
    f, w = settings.features, settings.windows
    out = resample_frames(raw, w)
    return {tf: add_indicators(df, f, w.tf_order) for tf, df in out.items()}


def _tail(raw: pd.DataFrame, settings: DataSettings) -> pd.DataFrame:
    t = settings.source.tail
    return raw.iloc[-t:] if t else raw


def _first_pass(names: list[str], loader, settings: DataSettings) -> dict:
    """{coin: {tf: high/low/close frame}} for the cross-asset builders (coins shorter than 2 windows are left out)."""
    w = settings.windows
    min_len = w.window_sizes[w.tf_order[0]]
    closed = is_closed_mode(w)
    out = {}
    for name in names:
        try:
            raw = _tail(loader(name), settings)
        except FileNotFoundError:
            continue
        if len(raw) < min_len * 2:
            continue
        frames = resample_frames(raw[list(TARGET_COLUMNS)], w, HLC_AGG)
        out[name] = {tf: df.dropna(how="all") for tf, df in frames.items()}
        if closed:
            base = out[name][w.tf_order[0]]
            out[name] = {tf: (df if tf == w.tf_order[0] else keep_full_bars(df, base.index, tf, w.tf_order[0]))
                         for tf, df in out[name].items()}
    return out


def _context(names: list[str], loader, settings: DataSettings, lock: HoldoutLock) -> BuildContext:
    f, w = settings.features, settings.windows
    ctx = BuildContext(reference=f.market.reference_symbol)
    need_cross = f.breadth.enabled or f.momentum_rank.enabled or f.orth_natr.enabled
    if need_cross:
        data = _first_pass(names, loader, settings)
        ctx.breadth = build_breadth(data, f, w.tf_order)
        ctx.rank = build_rank(data, f, w.tf_order)
        ctx.orth = build_orth(data, f, w, settings.scaling)
    if f.market.enabled:
        ref = f.market.reference_symbol
        if is_sealed(ref, lock):
            raise ValueError(f"market.reference_symbol {ref!r} is a sealed coin and cannot be used")
        try:
            raw = _tail(loader(ref), settings)
        except (FileNotFoundError, KeyError) as e:
            raise ValueError(f"market context needs the reference coin {ref!r} in the source (or set "
                             "features.market.enabled=False)") from e
        frames = _frames(raw, settings)
        ctx.market = {tf: market_return_frame(df, f) for tf, df in frames.items()}
    return ctx


def _one_coin(name: str, loader, ctx: BuildContext, features: list[str], settings: DataSettings) -> _Outcome:
    w, f = settings.windows, settings.features
    min_len = w.window_sizes[w.tf_order[0]]
    try:
        raw = _tail(loader(name), settings)
    except FileNotFoundError as e:
        return _Outcome(name, reason=f"no candle file: {e}")
    if len(raw) < min_len * 2:
        return _Outcome(name, reason=f"too short: {len(raw)} raw rows < {min_len * 2}")
    dfs = _frames(raw, settings)
    dfs = add_market_context(dfs, ctx.market, _norm(name) == _norm(ctx.reference), f)
    dfs = add_rank(dfs, (ctx.rank or {}).get(name), f)
    dfs = add_breadth(dfs, ctx.breadth, f)
    dfs = add_orth(dfs, (ctx.orth or {}).get(name), f)
    n_base = len(dfs[w.tf_order[0]])
    if n_base < 1.5 * min_len:
        return _Outcome(name, reason=f"too short after indicators: {n_base} rows < {1.5 * min_len:g}")
    res = prepare_asset(dfs, features, settings)
    if res.n == 0:
        return _Outcome(name, reason="no valid window")
    return _Outcome(name, result=res)


def _workers(settings: DataSettings, n: int) -> int:
    k = settings.runtime.workers or min(4, os.cpu_count() or 1)
    return max(1, min(k, n))


def _run_coins(names, loader, ctx, features, settings) -> list[_Outcome]:
    def job(name):
        return _one_coin(name, loader, ctx, features, settings)
    k = _workers(settings, len(names))
    if k == 1:
        return [job(n) for n in names]
    with ThreadPoolExecutor(max_workers=k) as pool:
        return list(pool.map(job, names))


def _spill(outcomes: list[_Outcome], spill_dir: str) -> None:
    """Move every coin's arrays to disk so only one copy of the data is alive when they are merged."""
    root = Path(spill_dir)
    root.mkdir(parents=True, exist_ok=True)
    for o in outcomes:
        if o.result is None:
            continue
        r = o.result
        np.savez(root / f"{o.name}.npz", bases=r.bases, last=r.last,
                 **{f"X_{tf}": a for tf, a in r.X.items()}, **{f"{OUTPUT_PREFIX}{h}": a for h, a in r.y.items()})
        o.extra["spilled"] = root / f"{o.name}.npz"
        o.result = AssetWindows({}, {}, np.empty((r.n, 2), "float32"), np.empty((r.n, 0)), [])


def _merge(outcomes: list[_Outcome], settings: DataSettings, n_feat: int) -> tuple[dict, list[dict]]:
    """Concatenate the coins' arrays (filling preallocated arrays coin by coin) and number their rows."""
    w, rt = settings.windows, settings.runtime
    kept = [o for o in outcomes if o.result is not None]
    total = sum(o.result.n for o in kept)
    x_dtype = np.dtype(rt.x_storage_dtype or "float32")
    heads = head_names(settings.targets.targets)
    out = {"base_params": np.empty((total, 2), "float32"), "last_candles": np.empty((total, 7), "float64")}
    for tf in w.tf_order:
        out[f"X_{tf}"] = np.empty((total, w.window_sizes[tf], n_feat), x_dtype)
    for h in heads:
        out[f"{OUTPUT_PREFIX}{h}"] = np.empty(total, "float32")
    bounds, pos = [], 0
    for o in kept:
        n = o.result.n
        sl = slice(pos, pos + n)
        if "spilled" in o.extra:
            with np.load(o.extra["spilled"]) as z:
                out["base_params"][sl], out["last_candles"][sl] = z["bases"], z["last"]
                for tf in w.tf_order:
                    out[f"X_{tf}"][sl] = z[f"X_{tf}"]
                for h in heads:
                    out[f"{OUTPUT_PREFIX}{h}"][sl] = z[f"{OUTPUT_PREFIX}{h}"]
        else:
            r = o.result
            out["base_params"][sl], out["last_candles"][sl] = r.bases, r.last
            for tf in w.tf_order:
                out[f"X_{tf}"][sl] = r.X[tf]
            for h in heads:
                out[f"{OUTPUT_PREFIX}{h}"][sl] = r.y[h]
        bounds.append({"name": o.name, "start": pos, "end": pos + n})
        pos += n
    return out, bounds


def build_dataset(settings: DataSettings, loader=None, lock: HoldoutLock | None = None) -> dict:
    """Run the whole pipeline and return the dataset dict (nothing is written; see artifact_writer)."""
    lock = lock or load_lock()
    loader = loader or make_loader(settings.source)
    requested = resolve_coins(settings.source)
    names = [n for n in requested if not is_sealed(n, lock)]
    if not names:
        raise ValueError("no coin is left to build: the list is empty or every coin is sealed")
    features = feature_order(settings)
    if settings.source.fetch:
        ref = settings.features.market.reference_symbol
        wanted = names + ([ref] if settings.features.market.enabled and ref not in names else [])
        fetch_all(settings.source, [n for n in wanted if not is_sealed(n, lock)])
    ctx = _context(names, loader, settings, lock)
    outcomes = _run_coins(names, loader, ctx, features, settings)
    if settings.runtime.spill_dir:
        _spill(outcomes, settings.runtime.spill_dir)
    arrays, bounds = _merge(outcomes, settings, len(features))
    if settings.runtime.spill_dir:
        shutil.rmtree(settings.runtime.spill_dir, ignore_errors=True)
    if not bounds:
        raise ValueError("no coin produced a window; reasons: " + "; ".join(f"{o.name}: {o.reason}" for o in outcomes))
    skipped = {o.name: o.reason for o in outcomes if o.result is None}
    return _assemble(settings, arrays, bounds, features, skipped, sorted(set(requested) - set(names)))


def _assemble(settings, arrays, bounds, features, skipped, sealed_dropped) -> dict:
    w, t, o = settings.windows, settings.targets, settings.output
    x_keys = [f"X_{tf}" for tf in w.tf_order]
    y_keys = [f"{OUTPUT_PREFIX}{h}" for h in head_names(t.targets)]
    ds = {"base_params": arrays["base_params"], "last_candles": arrays["last_candles"], "mode": DATASET_MODE,
          "scaler_type": settings.scaling.scaler_type, "feature_order": features,
          "timeframes": list(w.tf_order),
          "targets": head_names(t.targets), "price_targets": list(t.targets), "asset_bounds": bounds,
          "base_tf": w.tf_order[0], "window_sizes": {tf: w.window_sizes[tf] for tf in w.tf_order},
          "stride": w.stride, "forecast_horizon": t.horizon, "reg_target_mode": t.reg_target_mode,
          "reg_target_scale": t.reg_target_scale, "return_target_scale": t.reg_target_scale}
    if t.reg_target_clip is not None:
        ds["reg_target_clip"] = t.reg_target_clip
    if o.split_dates:
        ds["split_dates"] = dict(o.split_dates)
    if o.holdout_start:
        ds["holdout_start"] = o.holdout_start
    ds["skipped_assets"] = skipped
    ds["sealed_dropped"] = sealed_dropped
    ds["data_settings"] = settings.to_dict()
    for k in x_keys + y_keys:
        ds[k] = arrays[k]
    return ds
