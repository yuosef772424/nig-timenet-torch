"""Advanced custom features

PURPOSE: Custom feature families that are off by default: fractals, RVI, Fisher, range volatility, mean reversion,
    jumps, SAR, regime and risk-adjusted momentum.
TAGS: custom features, advanced, volatility estimators, mean reversion, hurst, jumps
PITFALLS: all causal (current and earlier bars only); a new family needs a causality test, never center=True rolling
    windows.

Custom feature families that are off by default: fractals, RVI / Fisher, range-based volatility estimators and
illiquidity, mean reversion (variance ratio, Hurst), jumps, RSI divergence, trend age, direction consistency,
volatility asymmetry, parabolic SAR, crash/rebound regime, risk-adjusted momentum and distance from the high.
All are causal (see custom_basic).
"""

from functools import partial

import numpy as np
import pandas as pd

from nig_timenet_torch.data.custom_registry import CUSTOM_FEATURES, EPS, Bars
from nig_timenet_torch.data.settings import CustomFeatureSettings

__all__ = ["fractal_columns", "bars_since_last_true", "variance_ratio_stat", "hurst_variance_scaling",
           "bipower_variation", "fractal", "rvi", "fisher", "vol_estimators", "mean_reversion", "jump",
           "rsi_divergence", "trend_age", "direction_consistency", "vol_asymmetry", "psar", "crash_rebound",
           "risk_adj_momentum", "ath_distance"]

REG = CUSTOM_FEATURES.decorator


def fractal_columns(high: pd.Series, low: pd.Series, window: int):
    """Williams fractals (local extremes of a centred window), shifted by half a window so they are causal."""
    half = window // 2
    raw_high = (high == high.rolling(window, center=True).max()).astype("float64")
    raw_low = (low == low.rolling(window, center=True).min()).astype("float64")
    return raw_high.shift(half).fillna(0.0), raw_low.shift(half).fillna(0.0)


def bars_since_last_true(mask: pd.Series, cap: float, default: float) -> pd.Series:
    idx = np.arange(len(mask))
    last = np.maximum.accumulate(np.where(mask.to_numpy(), idx, -1))
    since = np.where(last < 0, default, idx - last).astype("float64")
    return pd.Series(np.clip(since, 0, cap), index=mask.index)


def variance_ratio_stat(x, q: int) -> float:
    x = np.asarray(x, dtype="float64")
    if len(x) <= q:
        return np.nan
    var1 = np.var(x, ddof=1)
    if var1 <= EPS:
        return np.nan
    qsums = pd.Series(x).rolling(q).sum().dropna().to_numpy()
    if len(qsums) < 2:
        return np.nan
    return np.var(qsums, ddof=1) / (q * var1)


def hurst_variance_scaling(x, taus) -> float:
    x = np.asarray(x, dtype="float64")
    log_taus, log_vars = [], []
    for tau in taus:
        if tau >= len(x):
            continue
        sums = pd.Series(x).rolling(tau).sum().dropna().to_numpy()
        if len(sums) < 2:
            continue
        var = np.var(sums, ddof=1)
        if var <= EPS:
            continue
        log_taus.append(np.log(tau))
        log_vars.append(np.log(var))
    if len(log_taus) < 2:
        return np.nan
    return np.polyfit(log_taus, log_vars, 1)[0] / 2.0


def bipower_variation(x) -> float:
    x = np.asarray(x, dtype="float64")
    n = len(x)
    if n < 2:
        return np.nan
    a = np.abs(x)
    return np.pi / 2.0 * (n / (n - 1)) * np.sum(a[1:] * a[:-1])


@REG("fractal", version=1, order=70, prefixes=("FRACTAL_",))
def fractal(b: Bars, s: CustomFeatureSettings) -> None:
    for w in s.fractal_windows:
        b.df[f"FRACTAL_high_{w}"], b.df[f"FRACTAL_low_{w}"] = fractal_columns(b.high, b.low, w)


@REG("rvi", version=1, order=80, prefixes=("RVI_",))
def rvi(b: Bars, s: CustomFeatureSettings) -> None:
    for w in s.rvi_windows:
        b.ta().rvi(length=w, append=True)


@REG("fisher", version=1, order=90, prefixes=("FISHERT_",))
def fisher(b: Bars, s: CustomFeatureSettings) -> None:
    for w in s.fisher_windows:
        b.ta().fisher(length=w, append=True)


def _log_ratio(a: pd.Series, c: pd.Series) -> pd.Series:
    return np.log(a.clip(lower=EPS) / c.clip(lower=EPS))


def _roll(w: int, x: pd.Series):
    return x.rolling(w, min_periods=w)


@REG("vol_estimators", version=1, order=100, prefixes=("PARKINSON_", "GK_", "RS_", "YZ_", "AMIHUD_", "ROLL_SPREAD_"))
def vol_estimators(b: Bars, s: CustomFeatureSettings) -> None:
    """Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang volatility, Amihud illiquidity, Roll spread."""
    for w in s.vol_estimator_windows:
        u, dn, cc = _log_ratio(b.high, b.open_), _log_ratio(b.low, b.open_), _log_ratio(b.close, b.open_)
        overnight = _log_ratio(b.open_, b.close.shift(1))
        hl_log = _log_ratio(b.high, b.low)
        rs_bar = u * (u - cc) + dn * (dn - cc)
        b.df[f"PARKINSON_{w}"] = np.sqrt(_roll(w, 1.0 / (4.0 * np.log(2)) * hl_log ** 2).mean()).fillna(0.0)
        gk_bar = 0.5 * hl_log ** 2 - (2 * np.log(2) - 1) * cc ** 2
        b.df[f"GK_{w}"] = np.sqrt(_roll(w, gk_bar.clip(lower=0)).mean()).fillna(0.0)
        b.df[f"RS_{w}"] = np.sqrt(_roll(w, rs_bar.clip(lower=0)).mean()).fillna(0.0)
        k = 0.34 / (1.34 + (w + 1) / (w - 1))
        yz = (_roll(w, overnight).var() + k * _roll(w, cc).var() + (1 - k) * _roll(w, rs_bar).mean()).clip(lower=0)
        b.df[f"YZ_{w}"] = np.sqrt(yz).fillna(0.0)
        if "volume" in b.df:
            dollar = (b.df["volume"].clip(lower=0).astype("float64") * b.close).replace(0, np.nan)
            b.df[f"AMIHUD_{w}"] = _roll(w, b.ret_1.abs() / dollar * 1e6).mean().fillna(0.0)
        cov = b.close.diff().rolling(w, min_periods=w).apply(
            lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 2 else np.nan, raw=True)
        b.df[f"ROLL_SPREAD_{w}"] = np.sqrt(np.clip(-cov, 0, None)).fillna(0.0)


@REG("mean_reversion", version=1, order=110, prefixes=("VR_", "HURST_"))
def mean_reversion(b: Bars, s: CustomFeatureSettings) -> None:
    for w in s.mean_reversion_windows:
        q = max(2, w // 4)
        taus = tuple(t for t in (1, 2, 4, 8) if t < w) or (1,)
        r = b.ret_1.rolling(w, min_periods=w)
        b.df[f"VR_{w}"] = r.apply(partial(variance_ratio_stat, q=q), raw=True).fillna(1.0)
        b.df[f"HURST_{w}"] = r.apply(partial(hurst_variance_scaling, taus=taus), raw=True).fillna(0.5)


@REG("jump", version=1, order=120, prefixes=("JUMP_RATIO_",))
def jump(b: Bars, s: CustomFeatureSettings) -> None:
    for w in s.jump_windows:
        rv = (b.ret_1 ** 2).rolling(w, min_periods=w).sum()
        bv = b.ret_1.rolling(w, min_periods=w).apply(bipower_variation, raw=True)
        b.df[f"JUMP_RATIO_{w}"] = ((rv - bv).clip(lower=0) / rv.replace(0, np.nan)).clip(0.0, 1.0).fillna(0.0)


def _rolling_z(x: pd.Series, n: int) -> pd.Series:
    return (x - x.rolling(n, min_periods=n).mean()) / x.rolling(n, min_periods=n).std().replace(0, np.nan)


@REG("rsi_divergence", version=1, order=140, prefixes=("RSI_DIVERGENCE_",))
def rsi_divergence(b: Bars, s: CustomFeatureSettings) -> None:
    if not s.rsi_divergence_windows:
        return
    rsi_14 = b.df["RSI_14"] if "RSI_14" in b.df.columns else b.ta().rsi(length=14)
    for length in s.rsi_divergence_windows:
        norm_w = max(4 * length, 20)
        z_price = _rolling_z(b.log_close - b.log_close.shift(length), norm_w)
        z_rsi = _rolling_z(rsi_14 - rsi_14.shift(length), norm_w)
        b.df[f"RSI_DIVERGENCE_{length}"] = (z_price - z_rsi).fillna(0.0)


@REG("trend_age", version=1, order=150, prefixes=("BARS_SINCE_",))
def trend_age(b: Bars, s: CustomFeatureSettings) -> None:
    for w in s.trend_age_windows:
        fh, fl = fractal_columns(b.high, b.low, w)
        b.df[f"BARS_SINCE_HIGH_{w}"] = bars_since_last_true(fh == 1.0, cap=10 * w, default=w)
        b.df[f"BARS_SINCE_LOW_{w}"] = bars_since_last_true(fl == 1.0, cap=10 * w, default=w)


@REG("direction_consistency", version=1, order=160, prefixes=("DIR_CONSISTENCY_",))
def direction_consistency(b: Bars, s: CustomFeatureSettings) -> None:
    same_sign = (np.sign(b.ret_1) == np.sign(b.ret_1.shift(1))).astype("float64")
    for w in s.direction_consistency_windows:
        b.df[f"DIR_CONSISTENCY_{w}"] = same_sign.rolling(w, min_periods=w).mean().fillna(0.5)


@REG("vol_asymmetry", version=1, order=170, prefixes=("VOL_ASYMMETRY_",))
def vol_asymmetry(b: Bars, s: CustomFeatureSettings) -> None:
    ret_sq = b.ret_1 ** 2
    down_sq, up_sq = ret_sq.where(b.ret_1 < 0, 0.0), ret_sq.where(b.ret_1 > 0, 0.0)
    for w in s.vol_asymmetry_windows:
        down = down_sq.rolling(w, min_periods=w).mean()
        up = up_sq.rolling(w, min_periods=w).mean()
        b.df[f"VOL_ASYMMETRY_{w}"] = ((down - up) / (down + up).replace(0, np.nan)).fillna(0.0)


@REG("psar", version=1, order=180, prefixes=("PSAR_",))
def psar(b: Bars, s: CustomFeatureSettings) -> None:
    if not s.psar_enabled:
        return
    p = b.ta().psar(af0=0.02, af=0.02, max_af=0.2)
    long_col = next(c for c in p.columns if c.startswith("PSARl_"))
    short_col = next(c for c in p.columns if c.startswith("PSARs_"))
    b.df["PSAR_DIR"] = np.where(p[long_col].notna(), 1.0, np.where(p[short_col].notna(), -1.0, 0.0))


@REG("crash_rebound", version=1, order=190, prefixes=("CRASH_REBOUND_",))
def crash_rebound(b: Bars, s: CustomFeatureSettings) -> None:
    crr = s.crash_rebound_regime
    if crr is None:
        return
    high_lb, recent_w = int(crr.get("high_lookback", 60)), int(crr.get("recent_window", 30))
    threshold = float(crr.get("drawdown_threshold", -0.3))
    rolling_high = b.close.rolling(high_lb, min_periods=max(2, high_lb // 2)).max()
    drawdown = b.close / rolling_high - 1.0
    b.df["CRASH_REBOUND_REGIME"] = (drawdown.rolling(recent_w, min_periods=1).min() <= threshold).astype("float64")


@REG("risk_adj_momentum", version=1, order=200, prefixes=("RISK_ADJ_MOM_",))
def risk_adj_momentum(b: Bars, s: CustomFeatureSettings) -> None:
    for w in s.risk_adj_momentum_windows:
        mu = b.ret_1.rolling(w, min_periods=w).mean()
        sigma = b.ret_1.rolling(w, min_periods=w).std()
        b.df[f"RISK_ADJ_MOM_{w}"] = (mu / sigma.replace(0, np.nan)).fillna(0.0)


@REG("ath_distance", version=1, order=210, prefixes=("PCT_FROM_ATH", "DAYS_SINCE_ATH"))
def ath_distance(b: Bars, s: CustomFeatureSettings) -> None:
    if not s.ath_distance_enabled:
        return
    ath = b.close.expanding(min_periods=1).max()
    b.df["PCT_FROM_ATH"] = (b.close / ath - 1.0).fillna(0.0)
    b.df["DAYS_SINCE_ATH"] = bars_since_last_true(b.close >= ath, cap=np.inf, default=0.0)
