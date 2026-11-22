"""Channel maths of the derived artifacts

PURPOSE: The indicator maths and the grid helpers the five derived-artifact tools share: Wilder RSI, the slow
    stochastic, the relative quote volume, "index -> UTC ns -> regular grid" and the window gather.
TAGS: rsi, stochastic, relative volume, regular grid, utc ns, window take, derived artifact, channel math, no lookahead
PITFALLS: every value at bar i uses bars <= i only (rule D2); the constants are the ones the written artifacts were
    built with (RSI 14, stochastic 14-3-3, volume mean of 30 previous bars, clip +-5): changing one changes every
    derived artifact's bytes. ``to_grid`` puts missing bars as NaN rows; a bar off the step is dropped.
"""

import numpy as np
import pandas as pd

__all__ = ["RSI_LEN", "STOCH_LEN", "STOCH_SMOOTH", "VOL_LEN", "VOL_CLIP", "WARMUP_BARS", "rsi_scaled", "stoch_slow",
           "volume_rel", "to_utc_index", "to_ns_utc", "to_grid", "window_take"]

RSI_LEN, STOCH_LEN, STOCH_SMOOTH, VOL_LEN = 14, 14, 3, 30
VOL_CLIP = 5.0
WARMUP_BARS = 100            # history fetched before a coin's first window (RSI converges in ~ 5 x 14 bars)


def rsi_scaled(close: pd.Series) -> np.ndarray:
    """Wilder RSI(14) of ``close`` scaled to (x - 50) / 50 (bar i uses bars <= i); 50 where there is no move."""
    d = close.diff()
    a = 1.0 / RSI_LEN
    au = d.clip(lower=0.0).ewm(alpha=a, adjust=False).mean()
    ad = (-d).clip(lower=0.0).ewm(alpha=a, adjust=False).mean()
    tot = au + ad
    r = np.where(tot > 0, 100.0 * au / tot.where(tot > 0, 1.0), 50.0)
    return (np.asarray(r, np.float64) - 50.0) / 50.0


def stoch_slow(close: pd.Series, high: pd.Series | None = None,
               low: pd.Series | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Slow stochastic (%K, %D) scaled to (x - 50) / 50: raw %K over 14 bars of ``high`` / ``low`` (default: the
    closes only), %K = its 3-bar mean, %D = the 3-bar mean of %K; a bar whose 14-bar range is 0 counts as 50."""
    hh = (close if high is None else high).rolling(STOCH_LEN).max()
    ll = (close if low is None else low).rolling(STOCH_LEN).min()
    rng = hh - ll
    raw = (100.0 * (close - ll) / rng.where(rng > 0, 1.0)).where(rng > 0, 50.0).where(rng.notna())
    k = raw.rolling(STOCH_SMOOTH).mean()
    return (k.to_numpy() - 50.0) / 50.0, (k.rolling(STOCH_SMOOTH).mean().to_numpy() - 50.0) / 50.0


def volume_rel(volume: pd.Series) -> pd.Series:
    """log(1 + volume) - log(1 + mean volume of the 30 PREVIOUS bars), clipped at +-5 (NaN volume counts as 0)."""
    v = volume.fillna(0.0).clip(lower=0.0)
    return (np.log1p(v) - np.log1p(v.rolling(VOL_LEN).mean().shift(1))).clip(-VOL_CLIP, VOL_CLIP)


def to_utc_index(index) -> pd.DatetimeIndex:
    """``index`` as a UTC DatetimeIndex (a naive index is taken as UTC)."""
    idx = pd.DatetimeIndex(index)
    return idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")


def to_ns_utc(index) -> np.ndarray:
    """A datetime index (tz-aware or naive = UTC) as int64 ns since the epoch."""
    return to_utc_index(index).as_unit("ns").asi8


def to_grid(candles: pd.DataFrame, step: int, columns=None, *, first: int | None = None,
            size: int | None = None) -> tuple[int, pd.DataFrame]:
    """(first open ns, the ``columns`` of ``candles`` on a regular grid of ``step`` ns: missing bars are NaN rows).
    ``first`` / ``size`` default to the candles' own first bar and span; bars off the step or outside the grid are
    dropped."""
    cols = list(candles.columns if columns is None else columns)
    ns = to_ns_utc(candles.index)
    if first is None:
        first = int(ns.min())
    pos = (ns - first) // step
    ok = ((ns - first) % step == 0) & (pos >= 0)
    if size is None:
        size = int(pos.max()) + 1
    ok &= pos < size
    out = np.full((size, len(cols)), np.nan)
    out[pos[ok]] = candles[cols].to_numpy(np.float64)[ok]
    return first, pd.DataFrame(out, columns=cols)


def window_take(end_bar: np.ndarray, window: int) -> np.ndarray:
    """(m, window) grid indices of the ``window`` bars ending at each bar of ``end_bar``."""
    return np.asarray(end_bar)[:, None] + np.arange(-window + 1, 1)[None, :]
