"""Raw candles -> per-candle features -> gap-free (context + horizon) windows for the autoregressive
next-candle model (``model/ar_candle_transformer.py``).

The horizon is a parameter: any ``(context_len, horizon_len)`` pair works (default 32 + 16 = a 48-candle
window; 32 + 4, 32 + 8, ... equally valid). Windows are cut from RAW bars only: a window is kept when its
context and horizon bars are consecutive on the exchange grid (no gap, no synthetic filler), the same
purity rule the main pipeline's ``windows.py`` applies — the raw data is never re-aggregated or filled,
because every window is emitted as an independent sample.

Features are per-candle and scale-free (a return, two wick ratios, the body position, a volume ratio), so
a window normalises itself and the model can consume its own predictions as inputs unchanged.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["FEATURE_NAMES", "RET_COL", "candle_features", "fit_volume_median", "contiguous_runs",
           "build_windows", "split_windows", "raw_candles", "ar_window_arrays"]

#: the five per-candle features, in column order; RET_COL is the log return of the close
FEATURE_NAMES = ("ret", "upper", "lower", "body", "vol")
RET_COL = 0


def fit_volume_median(df: pd.DataFrame, train_share: float = 0.7) -> float:
    """Median volume over the FIRST ``train_share`` of the bars (the train part in time order), so the
    volume-ratio feature never sees future data (rule D2)."""
    if "volume" not in df.columns:
        return 0.0
    n = max(1, int(len(df) * float(train_share)))
    return float(np.median(df["volume"].to_numpy(dtype="float64")[:n]))


def candle_features(df: pd.DataFrame, volume_median: float) -> np.ndarray:
    """(T, 5) float32 per-candle features, each computed from bar t and earlier only:
    ``ret`` log(close_t/close_{t-1}) (0 for the first bar), ``upper``/``lower`` log wick ratios,
    ``body`` (close-open)/(high-low) in [-1, 1], ``vol`` log1p(volume) - log1p(median)."""
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"column {col!r} is missing from the candle frame; features need it")
    o = df["open"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    c = df["close"].to_numpy(dtype="float64")
    ret = np.zeros_like(c)
    ret[1:] = np.log(c[1:] / np.maximum(c[:-1], 1e-12))
    body_hi = np.maximum(o, c)
    body_lo = np.minimum(o, c)
    upper = np.log(np.maximum(h, 1e-12) / np.maximum(body_hi, 1e-12))
    lower = np.log(np.maximum(body_lo, 1e-12) / np.maximum(low, 1e-12))
    rng = h - low
    body = np.divide(c - o, rng, out=np.zeros_like(c), where=rng > 1e-12)
    vol = np.log1p(df["volume"].to_numpy(dtype="float64")) if "volume" in df.columns else np.zeros_like(c)
    vol = vol - np.log1p(max(volume_median, 0.0))
    return np.stack([ret, upper, lower, np.clip(body, -1.0, 1.0), vol], axis=1).astype("float32")


def contiguous_runs(index: pd.DatetimeIndex, freq: str) -> list[slice]:
    """Slices of ``index`` where every pair of neighbours is exactly ``freq`` apart (gap-free stretches)."""
    step = pd.Timedelta(freq)
    ns = index.as_unit("ns").asi8
    out, start = [], 0
    for i in range(1, len(ns)):
        if ns[i] - ns[i - 1] != int(step // pd.Timedelta(1, "ns")):
            out.append(slice(start, i))
            start = i
    out.append(slice(start, len(ns)))
    return out


@dataclass
class CoinWindows:
    """One coin's windows: ``X`` (n, context, F) is the model's input, ``Y`` (n, horizon, F) the ground
    truth the model must generate candle by candle, ``dirs`` (n, horizon) the sign of each future return,
    ``trend_dir`` (n,) the sign of the whole horizon move, ``last_ret`` (n,) the last context return and
    ``end_times`` (n,) the timestamp of each window's last candle (None when built without an index)."""
    X: np.ndarray
    Y: np.ndarray
    dirs: np.ndarray
    trend_dir: np.ndarray
    last_ret: np.ndarray
    end_times: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.X.shape[0])


def build_windows(features: np.ndarray, runs: list[slice], context_len: int, horizon_len: int,
                  stride: int = 1, index: pd.DatetimeIndex | None = None) -> CoinWindows:
    """Cut every gap-free run into (context + horizon) windows; ``stride`` spaces the window ends. When
    ``index`` (the frame's DatetimeIndex the runs refer to) is given, each window's end timestamp is
    recorded - the panel builder aligns coins on it."""
    if context_len < 1 or horizon_len < 1:
        raise ValueError(f"context_len and horizon_len must be >= 1, got {context_len}/{horizon_len}")
    xs, ys, ends = [], [], []
    for run in runs:
        need = context_len + horizon_len
        for end in range(run.start + need - 1, run.stop, stride):
            xs.append(features[end - need + 1:end - horizon_len + 1])
            ys.append(features[end - horizon_len + 1:end + 1])
            ends.append(index[end] if index is not None else None)
    empty = CoinWindows(np.zeros((0, context_len, features.shape[1]), "float32"),
                        np.zeros((0, horizon_len, features.shape[1]), "float32"),
                        np.zeros((0, horizon_len), "float32"), np.zeros(0, "float32"), np.zeros(0, "float32"))
    if not xs:
        return empty
    X, Y = np.stack(xs), np.stack(ys)
    dirs = np.sign(Y[:, :, RET_COL]).astype("float32")
    trend = np.sign(Y[:, :, RET_COL].sum(axis=1)).astype("float32")
    last = X[:, -1, RET_COL].astype("float32")
    times = pd.DatetimeIndex(ends).as_unit("ns").asi8 if index is not None else None
    return CoinWindows(X.astype("float32"), Y.astype("float32"), dirs, trend, last, times)


def _coin_splits(w: CoinWindows, val_share: float, test_share: float) -> dict:
    """One coin's windows cut in time order into the three splits (train head, val middle, test tail);
    ``end_times`` rides along when the coin was built with an index (else None: the caller drops it)."""
    n = w.n
    cuts = (int(n * (1.0 - val_share - test_share)), int(n * (1.0 - test_share)))
    spans = {"train": slice(0, cuts[0]), "val": slice(cuts[0], cuts[1]), "test": slice(cuts[1], n)}
    out = {}
    for name, sl in spans.items():
        out[name] = {k: arr[sl] for k, arr in (("X", w.X), ("Y", w.Y), ("dirs", w.dirs),
                                               ("trend_dir", w.trend_dir), ("last_ret", w.last_ret))}
        if w.end_times is not None:
            out[name]["end_times"] = w.end_times[sl]
    return out


def split_windows(coins: dict[str, CoinWindows], val_share: float = 0.15, test_share: float = 0.15) -> dict:
    """Time-ordered train/val/test split of every coin's windows (per coin: the first rows train, the
    middle val, the tail test), concatenated over coins; returns ``{split}_{X,Y,dirs,trend_dir,last_ret}``
    plus ``{split}_coins`` (the coin name of every row) and ``{split}_end_times`` (the window end
    timestamps, ONLY when every coin was built with an index - else the keys are absent)."""
    if val_share < 0 or test_share < 0 or val_share + test_share >= 1.0:
        raise ValueError(f"val_share + test_share must be in [0, 1), got {val_share}/{test_share}")
    merged = {name: {k: [] for k in ("X", "Y", "dirs", "trend_dir", "last_ret", "coins", "end_times")}
              for name in ("train", "val", "test")}
    for coin, w in coins.items():
        for name, part in _coin_splits(w, val_share, test_share).items():
            for k, arr in part.items():
                if k == "end_times" and arr is None:
                    continue
                if arr.shape[0]:
                    merged[name][k].append(arr)
            if part["X"].shape[0]:
                merged[name]["coins"].append(np.full(part["X"].shape[0], coin))
    out = {}
    for name, part in merged.items():
        if not part["X"]:
            raise ValueError(f"split {name!r} is empty: more data or smaller val/test shares needed")
        for k, arrs in part.items():
            if not arrs:
                continue
            out[f"{name}_{k}"] = (np.concatenate(arrs, axis=0).astype("float32")
                                  if k in ("X", "Y", "dirs", "trend_dir", "last_ret")
                                  else np.concatenate(arrs, axis=0))
    return out


def ar_window_arrays(w: CoinWindows) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The (X, Y, dirs) triple the trainer consumes, as float32 arrays."""
    return w.X.astype("float32"), w.Y.astype("float32"), w.dirs.astype("float32")


def panel_windows(coins: dict[str, CoinWindows]) -> dict:
    """Align the coins' windows on their END TIMESTAMPS - a window position shared by every coin - into
    the panel grid the cross-coin model eats: ``X`` (B, N, C, F), ``Y`` (B, N, H, F), ``dirs`` (B, N, H),
    ``last_ret`` (B, N), ``coin_names`` (N,) and ``end_times`` (B,). Coins are the order of ``coins``' keys."""
    names = list(coins)
    if not names:
        raise ValueError("panel_windows needs at least one coin")
    rows = {}
    for name, w in coins.items():
        if w.end_times is None:
            raise ValueError(f"coin {name!r} has no end times: build_windows needs the frame index")
        rows[name] = {int(t): i for i, t in enumerate(w.end_times)}
    common = sorted(set(rows[names[0]]).intersection(*(set(rows[n]) for n in names[1:])))
    if not common:
        raise ValueError("no timestamp is covered by a window of every coin: align the data first")
    take = {name: np.array([rows[name][t] for t in common], dtype=np.int64) for name in names}
    stack = lambda field: np.stack([getattr(coins[n], field)[take[n]] for n in names], axis=1)  # noqa: E731
    return {"X": stack("X").astype("float32"), "Y": stack("Y").astype("float32"),
            "dirs": stack("dirs").astype("float32"), "last_ret": stack("last_ret").astype("float32"),
            "coin_names": np.array(names), "end_times": np.array(common, dtype=np.int64),
            "n": len(common)}


def split_panel(panel: dict, val_share: float = 0.15, test_share: float = 0.15) -> dict:
    """Time-ordered train/val/test split of a panel grid along the window axis (train head, val middle,
    test tail); returns the same fields per split plus the shared ``coin_names``."""
    if val_share < 0 or test_share < 0 or val_share + test_share >= 1.0:
        raise ValueError(f"val_share + test_share must be in [0, 1), got {val_share}/{test_share}")
    n = int(panel["n"])
    cuts = (int(n * (1.0 - val_share - test_share)), int(n * (1.0 - test_share)))
    spans = {"train": slice(0, cuts[0]), "val": slice(cuts[0], cuts[1]), "test": slice(cuts[1], n)}
    out = {"coin_names": panel["coin_names"]}
    for name, sl in spans.items():
        rows = sl.start
        stop = sl.stop
        if stop - rows == 0:
            raise ValueError(f"split {name!r} is empty: more data or smaller val/test shares needed")
        for k in ("X", "Y", "dirs", "last_ret", "end_times"):
            out[f"{name}_{k}"] = panel[k][rows:stop]
    return out


def resolve_coin_list(coins, root: str = ".") -> list[str]:
    """The coins to prepare data with: ``"all"`` (the registry's default category), ``"all_global"``
    (every coin in the registry) or an explicit list/tuple of names — minus duplicates; the same
    resolution the main pipeline's ``SourceSettings`` applies. ``root`` is where the registry
    (``<root>/crypto_data/asset_registry.csv``) is read from - the Drive root for the ``drive`` source."""
    from nig_timenet_torch.data.settings import SourceSettings
    from nig_timenet_torch.data.sources import resolve_coins
    if isinstance(coins, str):
        if coins not in ("all", "all_global"):
            raise ValueError(f"unknown coin selection {coins!r}; use 'all', 'all_global' or a list of names")
        return resolve_coins(SourceSettings(categories=(coins,), root=root))
    names = [str(c) for c in coins]
    if not names:
        raise ValueError("the coin list is empty; pass 'all', 'all_global' or at least one name")
    return resolve_coins(SourceSettings(coins=tuple(names)))


def raw_candles(source: str, coins, days: int, interval: str = "1h",
                root: str = ".", seed: int = 0) -> dict[str, pd.DataFrame]:
    """Raw OHLCV frames for the notebook: ``synthetic`` (seeded, offline), ``binance`` (download the
    newest candles via ``data.fetch`` then read them back; synthetic fallback when the network refuses)
    or ``drive`` (read the candle files the Drive root already holds - the crypto-signal-prediction
    layout - with NO fallback: a missing file is a clear error), so the notebook runs from the first
    attempt either way. ``coins`` is free: 'all', 'all_global' or an explicit list — see
    ``resolve_coin_list``; the newest ``days`` days are kept (``days`` <= 0 keeps everything)."""
    names = resolve_coin_list(coins, root=root)
    try:
        make = _RAW_SOURCES[source]
    except KeyError:
        raise ValueError(f"unknown source {source!r}; known: {sorted(_RAW_SOURCES)}") from None
    return make(names, int(days), int(seed), interval, root=root)


def _synthetic_frames(names, days, seed, interval, root="."):
    from nig_timenet_torch.data.sources import synthetic_market
    return synthetic_market(names, days, seed, interval)


def _binance_frames(names, days, seed, interval, root="."):
    """Download the newest futures candles via ``data.fetch``, read them back, and fall back to the
    seeded synthetic market when the network refuses — the notebook must run from the first attempt."""
    from nig_timenet_torch.data.fetch import fetch_all
    from nig_timenet_torch.data.settings import SourceSettings
    from nig_timenet_torch.data.sources import make_loader
    s = SourceSettings(kind="files", root=root, coins=tuple(names), interval=interval, fetch="futures")
    try:
        fetch_all(s, names)
        loader = make_loader(s)
        frames = {name: loader(name) for name in names}
        if all(len(f) for f in frames.values()):
            return frames
        raise RuntimeError("a coin came back empty")
    except Exception as exc:  # noqa: BLE001 - the fallback is the point: the run must not die here
        print(f"fetch failed ({type(exc).__name__}: {exc}); falling back to the synthetic market")
        from nig_timenet_torch.data.sources import synthetic_market
        return synthetic_market(names, days, seed, interval)


def _drive_frames(names, days, seed, interval, root="."):
    """Read the candle files the owner's Drive already holds - the SAME layout crypto-signal-prediction
    and the data-pipeline notebook use (``<root>/history_1h/<SYMBOL>.csv`` or ``.csv.gz``; a parquet
    pattern reads the public Speirsy11 files) via the SAME files loader the data pipeline uses. No
    download and NO synthetic fallback: a missing file is a clear error naming the searched path,
    because the owner asked for THAT data. The newest ``days`` days of every coin are kept (``days``
    <= 0 keeps everything)."""
    from nig_timenet_torch.data.settings import SourceSettings
    from nig_timenet_torch.data.sources import make_loader, resample_ohlcv
    s = SourceSettings(kind="files", root=root, coins=tuple(names), interval=interval)
    loader = make_loader(s)
    frames = {}
    for name in names:
        frame = loader(name)
        if not len(frame):
            raise RuntimeError(f"[{name}] the candle file on Drive is empty: {root}")
        frame = resample_ohlcv(frame, interval)  # the Drive files are 1h bars: combine them up to INTERVAL
        if not len(frame):
            raise RuntimeError(f"[{name}] no full {interval} bars after re-aggregating the 1h file: {root}")
        if days > 0:
            frame = frame[frame.index >= frame.index.max() - pd.Timedelta(days=days)]
        frames[name] = frame
    return frames


_RAW_SOURCES = {"synthetic": _synthetic_frames, "binance": _binance_frames, "drive": _drive_frames}
