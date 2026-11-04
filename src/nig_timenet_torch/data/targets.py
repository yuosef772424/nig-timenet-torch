"""Regression/class targets from the last-candle record, and switching them after the data is built.

A target mode is a function ``raw(P, price, close_reg) -> r`` registered in ``TARGET_RAW`` (G8: one function, one
registration line). ``target_arrays`` turns ``r`` into ``{price}_class`` (1.0 when the head's direction is up) and
``{price}_reg = clip(r, +-clip) * scale``. ``retarget`` rebuilds them on a whole dataset, so the same windows can
be trained against another normalisation (the reference project's ``retarget_splits``):

    return        (future - last) / last of the head's own price          window_scale   (future - last) / iqr
    return_close  (future - last_close) / last_close                       scaled         same, unit already 1
    volnorm       return / (window iqr / last_close)                       magnitude      |r|, class by train median
    entry_range   non-negative distances from the entry (last close)       "<mode>+relative"  minus the per-time median

Every mode but ``window_scale`` and ``scaled`` is in return units and is multiplied by ``reg_target_scale``.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nig_timenet_torch.core.prediction_schema import OUTPUT_PREFIX
from nig_timenet_torch.core.registry import Contract, Registry
from nig_timenet_torch.core.schema import LAST_COLUMNS, LAST_DTYPE

__all__ = ["TARGET_RAW", "ENTRY_CLOSE_REGS", "TargetInfo", "parse_mode", "target_arrays", "retarget", "decode_reg",
           "mode_names", "price_columns"]

EPS = 1e-12
ENTRY_CLOSE_REGS = ("abs_return", "range_pos")
_FUTURE = {"high": "future_high_max", "low": "future_low_min", "close": "future_close"}
_DEFAULT_CLIP = 1.0
_OWN_RULES = ("own",)
_FIT_RULES = ("threshold",)   # class rules that need a statistic fitted on the training rows

#: ``unit_scaled``: multiplied by reg_target_scale (bool, or ``f(price, close_reg) -> bool``); ``clip``: default
#: clip; ``class_rule``: how the class label is
#: derived ("sign" of r, "threshold" = train median of r, "own" = the head's own direction); ``needs_bases``: uses the
#: window (centre, scale); ``per_row``: computable one coin at a time (no train statistic, no cross-section);
#: ``decode``: ``f(P, price, u) -> future price`` for a prediction in raw units (None = not invertible);
#: ``relative_ok``: may take the '+relative' suffix.
TARGET_RAW = Registry("target_raw", Contract(
    "target_raw", required_meta=("unit_scaled", "clip", "class_rule", "needs_bases", "per_row", "decode",
                                 "relative_ok")))


def price_columns(last: np.ndarray, bases: np.ndarray | None) -> dict[str, np.ndarray]:
    """{name: column} of the last-candle record (+ bp_center / bp_scale when the window bases are given)."""
    lc = np.asarray(last, dtype=LAST_DTYPE)
    P = {name: lc[:, i] for i, name in enumerate(LAST_COLUMNS)}
    if bases is not None:
        b = np.asarray(bases, dtype="float64")
        P["bp_center"], P["bp_scale"] = b[:, 0], b[:, 1]
    return P


def _own_last(P, price):
    return P[f"last_{price}"]


def _need_bases(P, mode):
    if "bp_scale" not in P:
        raise ValueError(f"target mode {mode!r} needs the window bases (base_params) of every row")


def _safe_scale(P):
    return np.where(np.abs(P["bp_scale"]) > EPS, P["bp_scale"], 1e-8)


def _rel_vol(P):
    return np.abs(P["bp_scale"]) / np.maximum(np.abs(P["last_close"]), EPS)


def _is_unit(meta, price, close_reg) -> bool:
    u = meta["unit_scaled"]
    return bool(u(price, close_reg) if callable(u) else u)


@TARGET_RAW.decorator("return", version=1, unit_scaled=True, clip=1.0, class_rule="sign", needs_bases=True,
                      per_row=True, decode=lambda P, t, u: P[f"last_{t}"] * (1.0 + u), relative_ok=True)
def _return(P, price, close_reg):
    """Own-price return; when the last price is ~0 the window centre (then 1e-12) is the denominator."""
    last = _own_last(P, price)
    med = P.get("bp_center", np.zeros_like(last))
    denom = np.where(np.abs(last) > EPS, last, np.where(np.abs(med) > EPS, med, EPS))
    return (P[_FUTURE[price]] - last) / denom


@TARGET_RAW.decorator("return_close", version=1, unit_scaled=True, clip=1.0, class_rule="sign", needs_bases=False,
                      per_row=True, decode=lambda P, t, u: P["last_close"] * (1.0 + u), relative_ok=True)
def _return_close(P, price, close_reg):
    return P[_FUTURE[price]] / P["last_close"] - 1.0


@TARGET_RAW.decorator("window_scale", version=1, unit_scaled=True, clip=10.0, class_rule="sign", needs_bases=True,
                      per_row=True, decode=lambda P, t, u: P[f"last_{t}"] + u * P["bp_scale"], relative_ok=True)
def _window_scale(P, price, close_reg):
    _need_bases(P, "window_scale")
    return (P[_FUTURE[price]] - _own_last(P, price)) / P["bp_scale"]


@TARGET_RAW.decorator("scaled", version=1, unit_scaled=False, clip=10.0, class_rule="sign", needs_bases=True,
                      per_row=True, decode=lambda P, t, u: P[f"last_{t}"] + u * _safe_scale(P), relative_ok=True)
def _scaled(P, price, close_reg):
    _need_bases(P, "scaled")
    return (P[_FUTURE[price]] - _own_last(P, price)) / _safe_scale(P)


@TARGET_RAW.decorator("volnorm", version=1, unit_scaled=False, clip=10.0, class_rule="sign", needs_bases=True,
                      per_row=False, decode=lambda P, t, u: P[f"last_{t}"] * (1.0 + u * _rel_vol(P)), relative_ok=True)
def _volnorm(P, price, close_reg):
    """Return divided by the window's relative volatility (iqr / last close); both are known at entry. The fallback
    for a near-zero volatility is the median over the rows given, so the mode needs a whole split (not one coin)."""
    _need_bases(P, "volnorm")
    vol = _rel_vol(P)
    good = vol > 1e-6
    vol = np.where(good, vol, np.nanmedian(vol[good]) if good.any() else 1.0)
    return (P[_FUTURE[price]] / _own_last(P, price) - 1.0) / vol


@TARGET_RAW.decorator("magnitude", version=1, unit_scaled=True, clip=1.0, class_rule="threshold", needs_bases=False,
                      per_row=False, decode=None, relative_ok=True)
def _magnitude(P, price, close_reg):
    r = P[_FUTURE[price]] / P["last_close"] - 1.0
    return {"close": np.abs(r), "high": np.maximum(r, 0.0), "low": np.maximum(-r, 0.0)}[price]


def _range_close(P, close_reg):
    hi, lo, fut = P["future_high_max"], P["future_low_min"], P["future_close"]
    rng = hi - lo
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = np.where(rng > 0, (fut - lo) / rng, 0.5)
    return np.abs(fut / P["last_close"] - 1.0) if close_reg == ENTRY_CLOSE_REGS[0] else pos


@TARGET_RAW.decorator("entry_range", version=1, unit_scaled=lambda t, close_reg: not (
    t == "close" and close_reg == ENTRY_CLOSE_REGS[1]), clip=1.0, class_rule="own", needs_bases=False, per_row=True,
                      decode=None, relative_ok=False)
def _entry_range(P, price, close_reg):
    """Non-negative distances from the entry (last close); direction lives only in the class heads. Gaps (the
    horizon high below the entry, or its low above it) are zero, not negative."""
    entry = P["last_close"]
    if price in ("high", "low"):
        sign = 1.0 if price == "high" else -1.0
        ratio = P[_FUTURE[price]] / entry - 1.0
        return np.maximum(sign * ratio, 0.0)
    return _range_close(P, close_reg)


def mode_names() -> tuple[str, ...]:
    return TARGET_RAW.names()


def parse_mode(mode: str) -> tuple[str, bool]:
    """'scaled+relative' -> ('scaled', True); 'relative' -> ('return', True)."""
    base, _, suffix = mode.partition("+")
    if base == "relative" and not suffix:
        return "return", True
    if base not in TARGET_RAW.names() or suffix not in ("", "relative"):
        raise ValueError(f"unknown target mode {mode!r}; known: {list(TARGET_RAW.names())}, any may take '+relative'")
    if suffix and not TARGET_RAW.entry(base).meta["relative_ok"]:
        raise ValueError("'entry_range+relative' is not supported: the cross-sectional median does not apply to "
                         "non-negative distances; use 'entry_range'")
    return base, suffix == "relative"


@dataclass
class TargetInfo:
    """What a call produced besides the arrays: the scale of every head, thresholds and group statistics."""
    mode: str
    scales: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    group_size: np.ndarray | None = None
    clip: float = _DEFAULT_CLIP
    close_reg: str = ENTRY_CLOSE_REGS[0]


def _group_keys(ts: np.ndarray, group_freq: str) -> np.ndarray:
    ts = ts.astype("int64")
    return ts if not group_freq else pd.to_datetime(ts).floor(group_freq).asi8


def target_arrays(last: np.ndarray, bases: np.ndarray | None, mode: str, targets, *, clip: float | None = None,
                  scale: float = 1.0, close_reg: str = ENTRY_CLOSE_REGS[0], center: str = "median",
                  group_freq: str = "", thresholds: dict | None = None) -> tuple[dict[str, np.ndarray], TargetInfo]:
    """``({price}_class, {price}_reg) -> float32 arrays`` for every price in `targets`, from `last` (N, 7)."""
    base, relative = parse_mode(mode)
    meta = TARGET_RAW.entry(base).meta
    if close_reg not in ENTRY_CLOSE_REGS:
        raise ValueError(f"close_reg must be one of {ENTRY_CLOSE_REGS}, got {close_reg!r}")
    P = price_columns(last, bases)
    clip = float(meta["clip"] if clip is None else clip)
    info = TargetInfo(mode=mode, clip=clip, close_reg=close_reg)
    keys = _group_keys(P["timestamp"], group_freq) if relative else None
    if keys is not None:
        info.group_size = pd.Series(keys).groupby(keys).transform("size").to_numpy()
    out = {}
    for t in targets:
        r = TARGET_RAW[base](P, t, close_reg)
        if relative:
            c = pd.Series(r).groupby(keys).transform(center).to_numpy()
            reg, up = r - c, r > c
        elif meta["class_rule"] in _FIT_RULES:
            thr = (thresholds or {}).get(t)
            if thr is None:
                thr = info.thresholds[t] = float(np.median(r))
            reg, up = r, r > thr
        elif meta["class_rule"] in _OWN_RULES:
            reg, up = r, P[_FUTURE[t]] > _own_last(P, t)
        else:
            reg, up = r, r > 0
        info.scales[t] = float(scale) if _is_unit(meta, t, close_reg) else 1.0
        out[f"{t}_reg"] = (np.clip(reg, -clip, clip) * info.scales[t]).astype("float32")
        out[f"{t}_class"] = up.astype("float32")
    return out, info


def decode_reg(pred: np.ndarray, last: np.ndarray, bases: np.ndarray | None, mode: str, price: str,
               scale: float = 1.0) -> np.ndarray:
    """A predicted `{price}_reg` -> a price, where that is defined (not magnitude, entry_range or '+relative')."""
    base, relative = parse_mode(mode)
    meta = TARGET_RAW.entry(base).meta
    if relative or meta["decode"] is None:
        raise ValueError(f"target mode {mode!r} cannot be turned back into a price (it drops the sign, the anchor or "
                         "the market level)")
    u = np.asarray(pred, dtype="float64") / (scale if _is_unit(meta, price, ENTRY_CLOSE_REGS[0]) else 1.0)
    return meta["decode"](price_columns(last, bases), price, u)


def retarget(dataset: dict, mode: str, *, train_idx: np.ndarray | None = None, clip: float | None = None,
             targets=None, close_reg: str = ENTRY_CLOSE_REGS[0], center: str = "median", group_freq: str = "",
             min_group: int = 5, reg_target_scale: float | None = None) -> tuple[dict, dict]:
    """A copy of `dataset` (X shared, not copied) whose y arrays follow `mode`; returns it with a report dict.

    ``train_idx`` fits the magnitude threshold (default: every row); ``min_group`` only feeds the report (share of
    rows whose time group has fewer coins than that, '+relative' only). The dataset is stamped with
    target_mode / reg_target_scale (the one the heads use) / reg_target_scales (per head) / return_target_scale
    (the scale of the return-unit modes, kept so a later retarget can go back) / reg_target_clip / entry_close_reg."""
    targets = list(targets or dataset["price_targets"])
    given = dataset.get("return_target_scale", dataset.get("reg_target_scale", 1.0))
    scale = float(given if reg_target_scale is None else reg_target_scale)
    last, bases = dataset["last_candles"], dataset.get("base_params")
    thresholds = None
    base, relative = parse_mode(mode)
    if TARGET_RAW.entry(base).meta["class_rule"] in _FIT_RULES and not relative:
        idx = np.arange(len(last)) if train_idx is None else np.asarray(train_idx)
        _, fit = target_arrays(last[idx], None if bases is None else bases[idx], mode, targets, clip=clip,
                               scale=scale, close_reg=close_reg)
        thresholds = fit.thresholds
    y, info = target_arrays(last, bases, mode, targets, clip=clip, scale=scale, close_reg=close_reg, center=center,
                            group_freq=group_freq, thresholds=thresholds)
    out = dict(dataset)
    for k, v in y.items():
        out[f"{OUTPUT_PREFIX}{k}"] = v
    out.update(target_mode=mode, reg_target_scales=dict(info.scales), reg_target_clip=info.clip)
    out["return_target_scale"] = scale
    out["reg_target_scale"] = scale if _is_unit(TARGET_RAW.entry(base).meta, "high", close_reg) else 1.0
    out["reg_target_mode"] = mode
    if TARGET_RAW.entry(base).meta["class_rule"] in _OWN_RULES:
        out["entry_close_reg"] = close_reg
    else:
        out.pop("entry_close_reg", None)
    report = {"mode": mode, "clip": info.clip, "scales": info.scales, "thresholds": thresholds or {}}
    if info.group_size is not None:
        report["share_small_groups"] = float((info.group_size < min_group).mean())
    return out, report

