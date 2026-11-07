"""Windows of one coin

PURPOSE: One coin's frames to windows, targets and the last-candle record, the unit the pipeline builds datasets
    from.
TAGS: windows, last candle, targets, horizon, stride, sample filter
PITFALLS: a window ends at base bar t and its future bars are strictly after t; the time seal is applied later by
    the artifact writer.

One coin's frames -> windows, targets and the last-candle record (the unit the pipeline builds datasets from).

For a window that ends at base bar ``t`` and a horizon of ``h`` base bars:
* ``{p}_class`` is 1 when the future price of ``p`` (high: the highest high of the next h bars, low: the lowest
  low, close: the close h bars ahead) is above the last price of ``p`` at ``t``, else 0;
* ``{p}_reg`` is ``clip(r, +-clip) * scale`` with ``r`` given by the target mode (data.targets; "return":
  ``(future - last) / last``);
* ``last`` holds ``LAST_COLUMNS`` (last prices, timestamp, future prices) so any other target definition can be
  rebuilt later (data.retarget), and ``bases`` the (centre, scale) of the window's closes.
A window is dropped when its future is not fully inside the data, when bars are missing inside the window or the
horizon, or when a sample filter rejects its last bar.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import LAST_COLUMNS, LAST_DTYPE, TARGET_COLUMNS
from nig_timenet_torch.data.align import align_windows, is_closed_mode
from nig_timenet_torch.data.scaling import process_windows, window_scale_params
from nig_timenet_torch.data.settings import DataSettings
from nig_timenet_torch.data.targets import target_arrays
from nig_timenet_torch.data.timeframe import normalize_tf

__all__ = ["AssetWindows", "prepare_asset", "head_names", "sample_filter_mask", "FILTER_OPS"]

FILTER_OPS = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal,
              "==": np.equal, "!=": np.not_equal}


@dataclass
class AssetWindows:
    """What `prepare_asset` returns for one coin. `X` has one array per time frame."""
    X: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    bases: np.ndarray
    last: np.ndarray
    end_times: list

    @property
    def n(self) -> int:
        return int(self.bases.shape[0])


def head_names(targets) -> list[str]:
    """Target heads in dataset order: high -> high_class, high_reg, low_class, ..."""
    return [f"{t}_{kind}" for t in targets for kind in ("class", "reg")]


def _empty(settings: DataSettings, n_feat: int) -> AssetWindows:
    w = settings.windows
    return AssetWindows({tf: np.zeros((0, w.window_sizes[tf], n_feat), dtype="float32") for tf in w.tf_order},
                        {h: np.zeros(0, dtype="float32") for h in head_names(settings.targets.targets)},
                        np.empty((0, 2), dtype="float32"), np.empty((0, len(LAST_COLUMNS)), dtype=LAST_DTYPE), [])


def sample_filter_mask(df: pd.DataFrame, filters) -> np.ndarray:
    """True where every filter accepts the bar. A filter is {"feature", "op", "value"} or {"expr": "<pandas eval>"}."""
    mask = np.ones(len(df), dtype=bool)
    for flt in filters:
        if "expr" in flt:
            out = np.asarray(df.eval(flt["expr"]))
        elif {"feature", "op", "value"} <= set(flt):
            if flt["feature"] not in df.columns or flt["op"] not in FILTER_OPS:
                raise ValueError(f"sample filter {flt!r}: unknown column or operator; columns {sorted(df.columns)}, "
                                 f"operators {sorted(FILTER_OPS)}")
            with np.errstate(invalid="ignore"):
                out = FILTER_OPS[flt["op"]](df[flt["feature"]].to_numpy(dtype="float64"), flt["value"])
        else:
            raise ValueError(f"sample filter {flt!r} needs {{'feature','op','value'}} or {{'expr'}}")
        if out.shape != (len(df),):
            raise ValueError(f"sample filter {flt!r} returned shape {out.shape}, expected {(len(df),)}")
        mask &= pd.Series(out).fillna(False).to_numpy().astype(bool)
    return mask


def _future(arr: np.ndarray, end_idx: np.ndarray, h: int, reducer) -> np.ndarray:
    """reducer over arr[end+1 : end+1+h] for every end (NaN where the future leaves the data)."""
    out = np.full(len(end_idx), np.nan)
    ok = end_idx + 1 + h <= len(arr)
    if ok.any() and len(arr) >= h:
        view = np.lib.stride_tricks.sliding_window_view(arr, h)
        out[ok] = reducer(view[end_idx[ok] + 1], axis=1)
    return out


def _valid_windows(price_df: pd.DataFrame, tf: str, end_idx: np.ndarray, h: int, win: int,
                   entry_ok: np.ndarray) -> np.ndarray:
    """Windows whose bars and horizon are gap-free and whose future is inside the data."""
    t = len(price_df)
    idx_ns = price_df.index.as_unit("ns").asi8
    bar_ns = int(pd.Timedelta(normalize_tf(tf)) // pd.Timedelta(1, "ns"))
    fe = end_idx + 1 + h
    ok = fe <= t
    fe_c = np.clip(fe - 1, 0, t - 1)
    start_c = np.clip(end_idx - win + 1, 0, t - 1)
    ok &= idx_ns[fe_c] - idx_ns[end_idx] == h * bar_ns
    ok &= (end_idx >= win - 1) & (idx_ns[end_idx] - idx_ns[start_c] == (win - 1) * bar_ns)
    return ok & entry_ok[end_idx]


def prepare_asset(dfs: dict[str, pd.DataFrame], features: list[str], settings: DataSettings) -> AssetWindows:
    """Frames of one coin ({tf: processed frame with all feature columns}) -> normalised windows and targets."""
    w, tg, sc = settings.windows, settings.targets, settings.scaling
    base_tf = w.tf_order[0]
    price_df = dfs[base_tf].sort_index()
    for col in TARGET_COLUMNS:
        if col not in price_df.columns:
            raise ValueError(f"column {col!r} is missing from the base frame {base_tf!r}; targets need it even "
                             "when it is excluded from the features")
    feat_dfs = {tf: dfs[tf][features] for tf in w.tf_order}
    windows, end_times = align_windows(feat_dfs, w)
    if windows[base_tf].shape[0] == 0:
        return _empty(settings, len(features))
    end_idx = price_df.index.get_indexer(pd.DatetimeIndex(end_times))
    entry_ok = sample_filter_mask(price_df, w.sample_filters)
    ok = _valid_windows(price_df, base_tf, end_idx, tg.horizon, w.window_sizes[base_tf], entry_ok) & (end_idx >= 0)
    end_idx = end_idx[ok]
    windows = {tf: arr[ok] for tf, arr in windows.items()}
    end_times = [t for t, k in zip(end_times, ok, strict=True) if k]
    if len(end_idx) == 0:
        return _empty(settings, len(features))
    return _assemble(settings, price_df, features, windows, end_idx, end_times, sc)


def _assemble(settings, price_df, features, windows, end_idx, end_times, sc) -> AssetWindows:
    w, tg = settings.windows, settings.targets
    base_tf = w.tf_order[0]
    high, low, close = (price_df[c].to_numpy(dtype="float64") for c in TARGET_COLUMNS)
    last = np.zeros((len(end_idx), len(LAST_COLUMNS)), dtype=LAST_DTYPE)
    last[:, 0], last[:, 1], last[:, 2] = high[end_idx], low[end_idx], close[end_idx]
    last[:, 3] = price_df.index.as_unit("ns").asi8[end_idx]
    last[:, 4] = close[end_idx + tg.horizon]
    last[:, 5] = _future(low, end_idx, tg.horizon, np.min)
    last[:, 6] = _future(high, end_idx, tg.horizon, np.max)
    if "close" in features:
        close_win = windows[base_tf][:, :, features.index("close")]
    else:
        win = w.window_sizes[base_tf]
        close_win = np.stack([close[e - win + 1:e + 1] for e in end_idx])
    med, iqr = window_scale_params(close_win, sc.scaler_type)
    bases = np.stack([med, iqr], axis=1).astype("float32")
    y, _ = target_arrays(last, bases, tg.reg_target_mode, tg.targets, clip=tg.reg_target_clip,
                         scale=tg.reg_target_scale, close_reg=tg.entry_close_reg)
    X = {}
    for tf in w.tf_order:
        if tf != base_tf and is_closed_mode(w) and "close" in features:
            centers, scales = window_scale_params(windows[tf][:, :, features.index("close")], sc.scaler_type)
        else:
            centers, scales = bases[:, 0], bases[:, 1]
        X[tf] = process_windows(windows[tf], features, centers, scales, sc)
    return AssetWindows(X, y, bases, last, end_times)
