"""Resampling and aligned windows

PURPOSE: Resample candles to each time frame and cut windows that end on the base frame, without look-ahead across
    frames.
TAGS: resample, time frame, aligned windows, closed bars, legacy mode, no look-ahead
PITFALLS: closed mode keeps complete higher bars only (an unfinished bar would leak the future); legacy mode is the
    TF fixed-count behaviour.

Resampling to the requested time frames and cutting aligned windows out of the per-frame feature matrices.

Windows end on the base frame; with several frames every higher-frame window is chosen so that it only contains
information that was already known at the base bar the window ends on ("closed": complete higher bars only;
"legacy": a fixed number of higher bars back).
"""

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import TARGET_COLUMNS
from nig_timenet_torch.data.settings import WindowSettings
from nig_timenet_torch.data.timeframe import normalize_tf

__all__ = ["OHLCV_AGG", "HLC_AGG", "is_closed_mode", "keep_full_bars", "resample_frames", "window_end_indices",
           "align_windows"]

OHLCV_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
HLC_AGG = {k: OHLCV_AGG[k] for k in TARGET_COLUMNS}


def is_closed_mode(w: WindowSettings) -> bool:
    """True when higher frames are used as complete bars only (needs more than one frame)."""
    return w.higher_tf_mode == "closed" and len(w.tf_order) > 1


def keep_full_bars(frame: pd.DataFrame, base_index: pd.DatetimeIndex, tf: str, base_tf: str) -> pd.DataFrame:
    """Drop higher-frame bars that were built from fewer base bars than a full bar holds (the unfinished last one,
    and bars with gaps), so a 'closed' bar really is complete."""
    try:
        n_full = int(pd.Timedelta(normalize_tf(tf)) // pd.Timedelta(normalize_tf(base_tf)))
    except (ValueError, TypeError):
        return frame
    if n_full <= 1:
        return frame
    counts = pd.Series(1, index=base_index).resample(normalize_tf(tf)).sum().reindex(frame.index)
    return frame[(counts >= n_full).to_numpy()]


def resample_frames(df: pd.DataFrame, w: WindowSettings, agg: dict | None = None) -> dict[str, pd.DataFrame]:
    """Raw candles -> {tf: frame}; the base frame is the first of `tf_order`. Empty bins are dropped."""
    agg = OHLCV_AGG if agg is None else agg
    df = df.sort_index()
    base_tf = w.tf_order[0]
    base = df.resample(normalize_tf(base_tf)).agg(agg).dropna(how="all")
    closed = is_closed_mode(w)
    out = {}
    for tf in w.tf_order:
        if tf == base_tf:
            out[tf] = base
            continue
        frame = base.resample(normalize_tf(tf)).agg(agg).dropna(how="all")
        out[tf] = keep_full_bars(frame, base.index[base["close"].notna()], tf, base_tf) if closed else frame
    return out


def window_end_indices(index: pd.DatetimeIndex, win: int, stride: int, tf: str, w: WindowSettings) -> np.ndarray:
    """Positions of the bars a window may end on. With grid alignment a window ends on a bar whose number
    (counted from `grid_anchor`) is a multiple of `stride`, so different coins cut windows at the same timestamps."""
    n, stride = len(index), max(int(stride), 1)
    if n < win:
        return np.empty(0, dtype=np.int64)
    if stride == 1 or not w.align_to_grid:
        return np.arange(win - 1, n, stride, dtype=np.int64)
    ts_ns = index.as_unit("ns").asi8
    try:
        bar_ns = int(pd.Timedelta(normalize_tf(tf)) // pd.Timedelta(1, "ns"))
    except (ValueError, TypeError):
        bar_ns = int(np.median(np.diff(ts_ns))) if n > 1 else 1
    anchor_ns = pd.Timestamp(w.grid_anchor).as_unit("ns").value
    bar_no = (ts_ns - anchor_ns) // max(bar_ns, 1)
    cand = np.flatnonzero(bar_no % stride == 0)
    cand = cand[cand >= win - 1]
    keep: list[int] = []
    for c in cand.tolist():
        if not keep or c - keep[-1] >= stride:
            keep.append(c)
    return np.asarray(keep, dtype=np.int64)


def _windows_at(values: np.ndarray, starts: np.ndarray, win: int) -> np.ndarray:
    view = np.moveaxis(np.lib.stride_tricks.sliding_window_view(values, win, axis=0), -1, 1)
    return np.ascontiguousarray(view[starts])


def _empty(win: int, n_feat: int) -> np.ndarray:
    return np.zeros((0, win, n_feat), dtype="float32")


def _check_index(dfs: dict) -> None:
    for tf, df in dfs.items():
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"the frame of time frame {tf!r} must have a DatetimeIndex")


def align_windows(dfs: dict[str, pd.DataFrame], w: WindowSettings) -> tuple[dict[str, np.ndarray], list]:
    """{tf: feature frame} -> ({tf: float32 windows (N, win, F)}, end timestamps of the N base windows)."""
    _check_index(dfs)
    base_tf = w.tf_order[0]
    base = dfs[base_tf].sort_index()
    win = w.window_sizes[base_tf]
    ends = window_end_indices(base.index, win, w.stride, base_tf, w)
    others = {tf: dfs[tf].sort_index() for tf in w.tf_order[1:]}
    last_pos: dict[str, np.ndarray] = {}
    keep = np.ones(len(ends), dtype=bool)
    end_ns = base.index.as_unit("ns").asi8[ends]
    for tf, df_tf in others.items():
        pos, ok = (_closed_positions if is_closed_mode(w) else _legacy_positions)(df_tf, tf, end_ns, w)
        keep &= ok
        last_pos[tf] = pos
    ends = ends[keep]
    out = {base_tf: _windows_at(base.to_numpy(dtype="float32", copy=False), ends - (win - 1), win)
           if len(ends) else _empty(win, base.shape[1])}
    for tf, df_tf in others.items():
        n_win = w.window_sizes[tf]
        out[tf] = (_windows_at(df_tf.to_numpy(dtype="float32", copy=False), last_pos[tf][keep] - (n_win - 1), n_win)
                   if len(ends) else _empty(n_win, df_tf.shape[1]))
    return out, list(base.index[ends])


def _closed_positions(df_tf: pd.DataFrame, tf: str, end_ns: np.ndarray, w: WindowSettings):
    """Last complete higher bar at each base end time, and whether its whole window is gap-free and recent enough."""
    n_win = w.window_sizes[tf]
    try:
        dur = int(pd.Timedelta(normalize_tf(tf)) // pd.Timedelta(1, "ns"))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"higher_tf_mode='closed' needs a fixed-length time frame, not {tf!r}") from exc
    idx_ns = df_tf.index.as_unit("ns").asi8
    pos = np.searchsorted(idx_ns, end_ns - dur, side="right") - 1
    start = pos - (n_win - 1)
    ok = start >= 0
    p, s0 = np.clip(pos, 0, None), np.clip(start, 0, None)
    ok &= idx_ns[p] - idx_ns[s0] == (n_win - 1) * dur
    ok &= end_ns - (idx_ns[p] + dur) < dur
    return pos, ok


def _legacy_positions(df_tf: pd.DataFrame, tf: str, end_ns: np.ndarray, w: WindowSettings):
    """The higher bar `higher_tf_offset` bars before the one containing the end time."""
    n_win = w.window_sizes[tf]
    idx_ns = df_tf.index.as_unit("ns").asi8
    pos = np.searchsorted(idx_ns, end_ns, side="right") - w.higher_tf_offset
    return pos, (pos >= 0) & (pos - n_win + 1 >= 0)
