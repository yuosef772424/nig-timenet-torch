"""Custom feature families built from the bars alone: returns, relative range, candle shape, position in range,
volatility ratios, volume z-score, calendar, supertrend stretch and the short/long volatility term structure.
All are causal: they use the current and earlier bars only (positive shifts, rolling windows without `center`)."""

import numpy as np
import pandas as pd

from nig_timenet_torch.data.custom_registry import CUSTOM_FEATURES, EPS, Bars, hour_features_meaningful
from nig_timenet_torch.data.settings import CustomFeatureSettings

REG = CUSTOM_FEATURES.decorator


@REG("returns", version=1, order=10, prefixes=("RET_",))
def returns(b: Bars, s: CustomFeatureSettings) -> None:
    """RET_h: log return over h bars, in percent."""
    for h in s.returns:
        b.df[f"RET_{h}"] = (b.log_close - b.log_close.shift(h)) * 100.0


@REG("range_rel", version=1, order=20, prefixes=("RANGE_", "BODY_", "WICK_"))
def range_rel(b: Bars, s: CustomFeatureSettings) -> None:
    """RANGE_rel: (high-low)/close in percent (always on); with candle_structure also body and wick shares."""
    hl_range = b.high - b.low
    b.df["RANGE_rel"] = hl_range / b.close.clip(lower=EPS) * 100.0
    if s.candle_structure:
        denom = hl_range.replace(0, np.nan)
        b.df["BODY_ratio"] = ((b.close - b.open_) / denom).fillna(0.0)
        b.df["WICK_upper"] = ((b.high - np.maximum(b.close, b.open_)) / denom).fillna(0.0)
        b.df["WICK_lower"] = ((np.minimum(b.close, b.open_) - b.low) / denom).fillna(0.0)


@REG("position_in_range", version=1, order=30, prefixes=("POS_",))
def position_in_range(b: Bars, s: CustomFeatureSettings) -> None:
    """POS_w: where the close sits inside the w-bar high/low range (0 = at the low, 1 = at the high)."""
    for w in s.range_windows:
        lo = b.low.rolling(w, min_periods=w).min()
        hi = b.high.rolling(w, min_periods=w).max()
        b.df[f"POS_{w}"] = ((b.close - lo) / (hi - lo).replace(0, np.nan)).fillna(0.5)


@REG("vol_ratio", version=1, order=40, prefixes=("VOLR_",))
def vol_ratio(b: Bars, s: CustomFeatureSettings) -> None:
    """VOLR_s_l: log of the short-window over the long-window standard deviation of 1-bar returns."""
    for short, long in s.vol_windows:
        v_s = b.ret_1.rolling(short, min_periods=short).std()
        v_l = b.ret_1.rolling(long, min_periods=long).std()
        b.df[f"VOLR_{short}_{long}"] = np.log((v_s / v_l.replace(0, np.nan)).clip(lower=EPS)).fillna(0.0)


@REG("volume_z", version=1, order=50, prefixes=("VOLZ_",))
def volume_z(b: Bars, s: CustomFeatureSettings) -> None:
    """VOLZ_w: z-score of log1p(volume) over the last w bars."""
    if "volume" not in b.df:
        return
    w = int(s.volume_window)
    lv = np.log1p(b.df["volume"].clip(lower=0).astype("float64"))
    mu = lv.rolling(w, min_periods=w).mean()
    sd = lv.rolling(w, min_periods=w).std()
    b.df[f"VOLZ_{w}"] = ((lv - mu) / sd.replace(0, np.nan)).fillna(0.0)


@REG("calendar", version=1, order=60, prefixes=("TIME_",))
def calendar(b: Bars, s: CustomFeatureSettings) -> None:
    """TIME_hour / TIME_dow sine-cosine pairs (the hour pair only when a frame shorter than a day is used)."""
    if not (s.time_features and isinstance(b.df.index, pd.DatetimeIndex)):
        return
    hour = b.df.index.hour.values + b.df.index.minute.values / 60.0
    dow = b.df.index.dayofweek.values
    if hour_features_meaningful(b.tf_order):
        b.df["TIME_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        b.df["TIME_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    b.df["TIME_dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    b.df["TIME_dow_cos"] = np.cos(2 * np.pi * dow / 7.0)


@REG("supertrend", version=1, order=130, prefixes=("SUPERT_",))
def supertrend(b: Bars, s: CustomFeatureSettings) -> None:
    """SUPERT_DIR_w (+1/-1) and SUPERT_STRETCH_w: distance of the close from the supertrend line, percent of close."""
    for w in s.supertrend_windows:
        st = b.ta().supertrend(length=w, multiplier=3.0)
        line, direction = st[f"SUPERT_{w}_3.0"], st[f"SUPERTd_{w}_3.0"]
        b.df[f"SUPERT_DIR_{w}"] = direction.fillna(0.0)
        b.df[f"SUPERT_STRETCH_{w}"] = ((b.close - line) / b.close.clip(lower=EPS) * 100.0).fillna(0.0)


@REG("vol_term_structure", version=1, order=220, prefixes=("VOL_TERM_",))
def vol_term_structure(b: Bars, s: CustomFeatureSettings) -> None:
    """VOL_TERM_s_l: short-window NATR over long-window NATR (above 1 = volatility is rising)."""
    if not s.vol_term_structure_pairs:
        return
    prev_close = b.close.shift(1)
    tr = pd.concat([b.high - b.low, (b.high - prev_close).abs(), (b.low - prev_close).abs()], axis=1).max(axis=1)
    for short_w, long_w in s.vol_term_structure_pairs:
        natr_short = 100.0 * tr.rolling(short_w, min_periods=short_w).mean() / b.close
        natr_long = 100.0 * tr.rolling(long_w, min_periods=long_w).mean() / b.close
        b.df[f"VOL_TERM_{short_w}_{long_w}"] = (natr_short / natr_long.replace(0, np.nan)).fillna(1.0)
