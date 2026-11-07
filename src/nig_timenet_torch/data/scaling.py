"""Per-window input scaling

PURPOSE: Per-window normalisation of the input features (window_scale and pct_change) with a kind chosen by the
    column-name prefix.
TAGS: scaling, normalisation, window scale, pct change, feature kinds, prefix
PITFALLS: a column takes the kind of its longest matching prefix and unmatched columns are standardised per window;
    the prefix table is generated from the TF pipeline.

Per-window normalisation of the inputs (price_norm_mode "window_scale" and "pct_change").

Every feature column has a kind chosen by its name prefix (RSI is an oscillator in [0, 100], RET_ a heavy-tailed
return, volume is log-scaled, ...); the kind decides how a window of that column is mapped to roughly [-1, 1].
Prices are centred and scaled with the window's own robust statistics so a window never sees another window's level.
"""

import numpy as np

from nig_timenet_torch.core.schema import OHLCV_COLUMNS
from nig_timenet_torch.data.settings import ScalingSettings

__all__ = ["classify_feature", "calc_scale_params", "window_scale_params", "process_windows", "KIND_PREFIXES",
           "SCALE_ABS_FLOOR", "SCALE_REL_FLOOR", "DEFAULT_KIND", "NEVER_ZEROED_KINDS"]

SCALE_ABS_FLOOR = 1e-8
SCALE_REL_FLOOR = 1e-3
DEFAULT_KIND = "cumulative"
NEVER_ZEROED_KINDS = frozenset({"binary_flag", "funding_rate"})
FLAG_SUFFIX = "_available"

# generated from the reference pipeline's FEATURE_KINDS (176 prefixes); a column takes the kind of its longest
# matching prefix, anything unmatched is "cumulative" (standardised per window).
KIND_PREFIXES: dict[str, tuple[str, ...]] = {
    "price_level": (
        *OHLCV_COLUMNS[:4], "hl2", "HLC3", "OHLC4", "WCP", "SMA", "EMA", "WMA", "DEMA", "TEMA",
        "TRIMA", "HMA", "RMA", "VWMA", "VWAP", "ALMA", "T3_", "ZL_EMA", "ZLMA", "KAMA", "FWMA", "SINWMA",
        "SSF", "SWMA", "PWMA", "LINREG", "MIDPOINT", "MIDPRICE", "BBL", "BBM", "BBU", "DCL", "DCM", "DCU",
        "KCL", "KCB", "KCU", "SUPERT", "PSAR", "ICHIMOKU", "ISA_", "ISB_", "ITS_", "IKS_",
    ),
    "price_scale": (
        "ATR", "ATRr", "TRUERANGE", "TR_", "STDEV", "VAR_", "MAD_", "MACD", "MOM", "SLOPE", "MIDP", "DPO",
        "ABER",
    ),
    "percent": (
        "ROC", "PPO", "NATR", "BBB", "CHOP", "MASSI", "PVR", "RANGE_", "SUPERT_STRETCH_", "ITD_RVOL",
    ),
    "osc_0_100": (
        "RSI", "STOCHk", "STOCHd", "STOCHh", "STOCHRSIk", "STOCHRSId", "ADX", "ADXR", "DMP", "DMN", "AROOND",
        "AROONU", "MFI", "RVI", "UO_", "CRSI", "PSL", "RVGI", "KST",
    ),
    "osc_sym_100": (
        "CMO", "AROONOSC", "CFO", "CTI",
    ),
    "osc_willr": (
        "WILLR",
    ),
    "unit_sym": (
        "BOP", "CMF", "QS_", "INC_", "DEC_", "BODY_", "TIME_", "SUPERT_DIR_", "PSAR_DIR", "VOL_ASYMMETRY_",
        "MKT_CORR_", "PCT_FROM_ATH", "MOM_ORTH_NATR", "LSR_", "TAKER_LSR_",
    ),
    "unit_0_1": (
        "ER_", "BBP", "SQZ_ON", "SQZ_OFF", "SQZ_NO", "UI_", "WICK_", "POS_", "MOM_RANK_", "MKT_BREADTH_",
        "DIR_CONSISTENCY_", "EFF_RATIO_", "VOL_CONC_", "CRASH_REBOUND_REGIME", "MKT_CRASH_REBOUND_REGIME",
        "ITD_TAKER_BUY_RATIO", "ITD_VOL_TOPK",
    ),
    "zscore": (
        "ZS_", "SKEW", "KURT", "VOLZ_", "VOL_TERM_", "RISK_ADJ_MOM_", "MKT_BETA_", "FUND_rate_z",
        "ITD_TRADES_Z",
    ),
    "sign_robust": (
        "CCI", "TSI", "TRIX", "FISHERT", "LOGRET", "PCTRET", "PVO", "PVI", "NVI", "BIAS", "COPC", "EOM", "SMI",
        "STC", "QQE", "TD_SEQ", "RET_", "VOLR_", "MKT_ret_", "VWAP_DEVIATION", "MKT_LAG_RET_", "OI_chg_",
        "LSR_GLOBAL_chg_",
    ),
    "cumulative": (
        "OBV", "AD", "ADOSC", "KVO", "EFI", "PVT", "CMF_SUM", "VP_", "ENTP",
    ),
    "log_volume": (
        "volume", "VOL_",
    ),
    "binary_flag": (
        "FUND_available", "OI_available", "ITD_available", "MET_available",
    ),
    "funding_rate": (
        "FUND_rate", "FUND_sum_",
    ),
    "log_centered": (
        "ITD_TRADES_LOG",
    ),
}

_SORTED_PREFIXES = sorted(((p, k) for k, ps in KIND_PREFIXES.items() for p in ps),
                          key=lambda pk: len(pk[0]), reverse=True)


def classify_feature(name: str) -> str:
    """The kind of a feature column: the kind of its longest matching prefix, or DEFAULT_KIND."""
    if name.endswith(FLAG_SUFFIX):
        return "binary_flag"
    for prefix, kind in _SORTED_PREFIXES:
        if name.startswith(prefix):
            return kind
    return DEFAULT_KIND


def calc_scale_params(data: np.ndarray, method: str = "robust") -> tuple[float, float]:
    """(centre, scale) of one window of closes: median / IQR ("robust") or min / range ("minmax"), floored."""
    if method == "robust":
        q75, q50, q25 = np.percentile(data, [75, 50, 25])
        return q50, max(q75 - q25, abs(q50) * SCALE_REL_FLOOR, SCALE_ABS_FLOOR)
    lo, hi = data.min(), data.max()
    return lo, max(hi - lo, abs(lo) * SCALE_REL_FLOOR, SCALE_ABS_FLOOR)


def window_scale_params(close_windows: np.ndarray, method: str = "robust") -> tuple[np.ndarray, np.ndarray]:
    """`calc_scale_params` for many windows at once: (centres, scales), both shape (N,)."""
    x = np.asarray(close_windows)
    if method == "robust":
        q75, q50, q25 = np.percentile(x, [75, 50, 25], axis=1)
        return q50, np.maximum(np.maximum(q75 - q25, np.abs(q50) * SCALE_REL_FLOOR), SCALE_ABS_FLOOR)
    lo, hi = x.min(axis=1), x.max(axis=1)
    return lo, np.maximum(np.maximum(hi - lo, np.abs(lo) * SCALE_REL_FLOOR), SCALE_ABS_FLOOR)


def _standardize(sub: np.ndarray, method: str, eps: float) -> np.ndarray:
    if method == "minmax":
        lo, hi = sub.min(axis=1, keepdims=True), sub.max(axis=1, keepdims=True)
        return (sub - lo) / (hi - lo + eps)
    q25, q50, q75 = np.percentile(sub, [25, 50, 75], axis=1, keepdims=True)
    return (sub - q50) / np.maximum(q75 - q25, 1e-8)


def _sign_robust(sub: np.ndarray, eps: float) -> np.ndarray:
    mag = 1.4826 * np.median(np.abs(sub), axis=1, keepdims=True)
    fallback = np.std(sub, axis=1, keepdims=True)
    return sub / np.where(mag < eps, np.maximum(fallback, eps), mag)


def pct_change_encode(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Bar-to-bar percent change along `axis` (the first bar is 0; non-positive or non-finite previous prices -> 0)."""
    x = np.moveaxis(np.asarray(x, dtype="float64"), axis, -1)
    out = np.zeros_like(x)
    prev, cur = x[..., :-1], x[..., 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100.0 * (cur / prev - 1.0)
    out[..., 1:] = np.where(np.isfinite(r) & (prev > 0), r, 0.0)
    return np.moveaxis(out, -1, axis)


def _pct_change_columns(raw: np.ndarray, axis: int) -> np.ndarray:
    x = np.asarray(raw, dtype="float64")
    finite = np.isfinite(x)
    if not finite.all():
        med = np.nanmedian(np.where(finite, x, np.nan), axis=axis, keepdims=True)
        x = np.where(finite, x, np.nan_to_num(med, nan=0.0))
    return pct_change_encode(x, axis=axis)


def _kind_transform(kind: str, sub: np.ndarray, c: np.ndarray, s: np.ndarray, method: str, eps: float) -> np.ndarray:
    """One kind's columns, shape (N, T, k), -> normalised."""
    simple = {
        "price_level": lambda: (sub - c) / s,
        "price_scale": lambda: sub / s,
        "percent": lambda: sub / 10.0,
        "osc_0_100": lambda: (sub - 50.0) / 50.0,
        "osc_sym_100": lambda: sub / 100.0,
        "osc_willr": lambda: (sub + 50.0) / 50.0,
        "unit_sym": lambda: sub,
        "unit_0_1": lambda: (sub - 0.5) * 2.0,
        "zscore": lambda: sub / 3.0,
        "sign_robust": lambda: _sign_robust(sub, eps),
        "binary_flag": lambda: sub * 2.0 - 1.0,
        "funding_rate": lambda: sub * 1000.0,
        "log_volume": lambda: _standardize(np.log1p(np.clip(sub, 0, None)), method, eps),
        "log_centered": lambda: sub - np.median(sub, axis=1, keepdims=True),
    }
    return simple.get(kind, lambda: _standardize(sub, method, eps))()


def process_windows(windows: np.ndarray, columns: list[str], centers: np.ndarray, scales: np.ndarray,
                    s: ScalingSettings) -> np.ndarray:
    """Raw windows (N, T, F) -> normalised float32 inputs; constant columns (except flags) become 0, values are
    clipped to +-clip_abs (percent-change prices to +-price_pct_clip)."""
    raw = windows
    x = np.asarray(windows, dtype="float32")
    n, _, n_cols = x.shape
    c = np.asarray(centers, dtype="float32").reshape(n, 1, 1)
    sc = np.asarray(scales, dtype="float32").reshape(n, 1, 1)
    finite = np.isfinite(x)
    if not finite.all():
        med = np.nanmedian(np.where(finite, x, np.nan), axis=1, keepdims=True)
        x = np.where(finite, x, np.nan_to_num(med, nan=0.0)).astype("float32", copy=False)
    lo, hi = x.min(axis=1), x.max(axis=1)
    keep_const = np.array([classify_feature(nm) in NEVER_ZEROED_KINDS for nm in columns[:n_cols]])
    degenerate = ((hi == lo) & ~keep_const[None, :]) | ~np.isfinite(lo) | ~np.isfinite(hi)
    by_kind: dict[str, list[int]] = {}
    for j, name in enumerate(columns[:n_cols]):
        by_kind.setdefault(classify_feature(name), []).append(j)
    pct_mode = s.price_norm_mode == "pct_change"
    for kind, idx in by_kind.items():
        if kind == "price_level" and pct_mode:
            x[:, :, idx] = _pct_change_columns(np.asarray(raw)[:, :, idx], axis=1)
        else:
            x[:, :, idx] = _kind_transform(kind, x[:, :, idx], c, sc, s.scaler_type, s.eps)
    x[np.broadcast_to(degenerate[:, None, :], x.shape)] = 0.0
    x = np.nan_to_num(x, nan=0.0, posinf=s.clip_abs, neginf=-s.clip_abs, copy=False)
    pct_idx = by_kind.get("price_level", []) if pct_mode else []
    pct_vals = x[:, :, pct_idx].copy() if pct_idx else None
    if s.clip_abs:
        np.clip(x, -s.clip_abs, s.clip_abs, out=x)
    if pct_idx:
        x[:, :, pct_idx] = np.clip(pct_vals, -s.price_pct_clip, s.price_pct_clip)
    return x
