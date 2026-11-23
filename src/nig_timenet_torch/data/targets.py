"""Targets from the last candle

PURPOSE: Regression and class targets from the last-candle record, retarget / retarget_splits to switch the target
    mode after the data is built (train-only fit, any order), and decode_reg / entry_range_to_prices back to prices.
TAGS: targets, target modes, retarget, retarget_splits, decode, entry range, relative, last candle, clip
PITFALLS: a target mode is one function in TARGET_RAW (unknown names raise); scale and clip are part of the target,
    so decode with decode_reg; magnitude / entry_range need the sign of the move for the close head, '+relative'
    needs the per-time centre (target_center_<t>): decode raises without them; mode None restores the pipeline y.

Regression/class targets from the last-candle record, and switching them after the data is built.

A target mode is a function ``raw(P, price, close_reg) -> r`` registered in ``TARGET_RAW`` (G8: one function, one
registration line). ``target_arrays`` turns ``r`` into ``{price}_class`` (1.0 when the head's direction is up) and
``{price}_reg = clip(r, +-clip) * scale``. ``retarget`` rebuilds them on a whole dataset, so the same windows can
be trained against another normalisation (the reference project's ``retarget_splits``):

    return        (future - last) / last of the head's own price          window_scale   (future - last) / iqr
    return_close  (future - last_close) / last_close                       scaled         same, unit already 1
    volnorm       return / (window iqr / last_close)                       magnitude      |r|, class by train median
    entry_range   non-negative distances from the entry (last close)       "<mode>+relative"  minus the per-time median

Every mode but ``window_scale`` and ``scaled`` is in return units and is multiplied by ``reg_target_scale``.

``retarget_splits(dataset, (train, val, test), mode)`` is the same on row sets: every statistic (the magnitude
threshold) is fitted on the TRAIN rows and reused on val / test, the cross-section of '+relative' is the rows of
each split, ``mode=None`` restores the pipeline's own y and stamps, and any order of modes gives the result of the
last one (the arrays are always rebuilt from ``last_candles``, never from the previous y).

Decoding: each mode registers ``decode(P, price, u) -> future price`` for a prediction ``u`` in raw units. Extra
columns of ``P`` that a mode may need: ``sign`` (+1 up / -1 down per row; the close head of magnitude and
entry_range), ``close_reg`` and ``range_high`` / ``range_low`` (decoded prices of the high and low heads; the
range_pos close head). Every decode is affine in ``u``, so ``decode_affine`` gives ``(anchor, slope)`` with
``price = anchor + slope * reg`` for any consumer that decodes row by row (data_io.target_codec "target_raw").
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nig_timenet_torch.core.prediction_schema import OUTPUT_PREFIX
from nig_timenet_torch.core.registry import Contract, Registry
from nig_timenet_torch.core.schema import FUTURE_OF, LAST_COLUMNS, LAST_DTYPE, LAST_OF, TARGET_COLUMNS
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS

__all__ = ["TARGET_RAW", "ENTRY_CLOSE_REGS", "TargetInfo", "parse_mode", "target_arrays", "retarget",
           "retarget_splits", "decode_reg", "decode_affine", "entry_range_to_prices", "mode_names", "price_columns",
           "center_key", "PIPELINE_TARGETS", "head_scale", "row_anchors", "TargetModeCodec"]

EPS = 1e-12
ENTRY_CLOSE_REGS = ("abs_return", "range_pos")
_FUTURE = FUTURE_OF
_DEFAULT_CLIP = 1.0
_OWN_RULES = ("own",)
_FIT_RULES = ("threshold",)   # class rules that need a statistic fitted on the training rows
CENTER_PREFIX = "target_center_"
PIPELINE_TARGETS = "pipeline_targets"   # the dataset key that keeps the pipeline's own y and stamps (mode None)
#: dataset keys a retarget stamps (restored by ``mode=None``; a key the pipeline dataset lacks is removed again)
STAMP_KEYS = ("target_mode", "reg_target_scales", "reg_target_clip", "return_target_scale", "reg_target_scale",
              "reg_target_mode", "entry_close_reg")

#: ``unit_scaled``: multiplied by reg_target_scale (bool, or ``f(price, close_reg) -> bool``); ``clip``: default
#: clip; ``class_rule``: how the class label is derived ("sign" of r, "threshold" = train median of r, "own" = the
#: head's own direction); ``needs_bases``: uses the window (centre, scale); ``per_row``: computable one coin at a
#: time (no train statistic, no cross-section); ``decode``: ``f(P, price, u) -> future price`` for a prediction ``u``
#: in raw units, affine in ``u`` (the module card lists the extra columns of ``P``); ``signed``: the decoded price is
#: a forecast on its own (False: the close head needs the sign of the move); ``relative_ok``: may take '+relative'.
TARGET_RAW = Registry("target_raw", Contract(
    "target_raw", required_meta=("unit_scaled", "clip", "class_rule", "needs_bases", "per_row", "decode", "signed",
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
    return P[LAST_OF[price]]


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


def _own_return(P, t, u):
    return P[f"last_{t}"] * (1.0 + u)


def _close_return(P, t, u):
    return P["last_close"] * (1.0 + u)


def _window_scale_price(P, t, u):
    return P[f"last_{t}"] + u * P["bp_scale"]


def _scaled_price(P, t, u):
    return P[f"last_{t}"] + u * _safe_scale(P)


def _volnorm_price(P, t, u):
    return P[f"last_{t}"] * (1.0 + u * _rel_vol(P))


def _from_entry(P, price, u):
    """high = P (1 + u), low = P (1 - u); the close head needs the direction (``sign``) or, for the range_pos
    definition, the decoded high and low prices (``range_high`` / ``range_low``)."""
    entry = P["last_close"]
    if price == "high":
        return entry * (1.0 + u)
    if price == "low":
        return entry * (1.0 - u)
    if P.get("close_reg") == ENTRY_CLOSE_REGS[1]:
        if "range_high" not in P or "range_low" not in P:
            raise ValueError("the range_pos close head decodes inside the predicted range: pass the decoded high and "
                             "low prices (range_prices=(high, low))")
        return P["range_low"] + np.clip(u, 0.0, 1.0) * (P["range_high"] - P["range_low"])
    if "sign" not in P:
        raise ValueError("the close head of a magnitude target has no sign: pass sign=+1 (up) / -1 (down) per row "
                         "(the true direction, or the one the class head predicts)")
    return entry * (1.0 + np.where(np.asarray(P["sign"]) >= 0, 1.0, -1.0) * u)


@TARGET_RAW.decorator("return", version=1, unit_scaled=True, clip=1.0, class_rule="sign", needs_bases=True,
                      per_row=True, decode=_own_return, signed=True, relative_ok=True)
def _return(P, price, close_reg):
    """Own-price return; when the last price is ~0 the window centre (then 1e-12) is the denominator."""
    last = _own_last(P, price)
    med = P.get("bp_center", np.zeros_like(last))
    denom = np.where(np.abs(last) > EPS, last, np.where(np.abs(med) > EPS, med, EPS))
    return (P[_FUTURE[price]] - last) / denom


@TARGET_RAW.decorator("return_close", version=1, unit_scaled=True, clip=1.0, class_rule="sign", needs_bases=False,
                      per_row=True, decode=_close_return, signed=True, relative_ok=True)
def _return_close(P, price, close_reg):
    return P[_FUTURE[price]] / P["last_close"] - 1.0


@TARGET_RAW.decorator("window_scale", version=1, unit_scaled=True, clip=10.0, class_rule="sign", needs_bases=True,
                      per_row=True, decode=_window_scale_price, signed=True, relative_ok=True)
def _window_scale(P, price, close_reg):
    _need_bases(P, "window_scale")
    return (P[_FUTURE[price]] - _own_last(P, price)) / P["bp_scale"]


@TARGET_RAW.decorator("scaled", version=1, unit_scaled=False, clip=10.0, class_rule="sign", needs_bases=True,
                      per_row=True, decode=_scaled_price, signed=True, relative_ok=True)
def _scaled(P, price, close_reg):
    _need_bases(P, "scaled")
    return (P[_FUTURE[price]] - _own_last(P, price)) / _safe_scale(P)


@TARGET_RAW.decorator("volnorm", version=1, unit_scaled=False, clip=10.0, class_rule="sign", needs_bases=True,
                      per_row=False, decode=_volnorm_price, signed=True, relative_ok=True)
def _volnorm(P, price, close_reg):
    """Return divided by the window's relative volatility (iqr / last close); both are known at entry. The fallback
    for a near-zero volatility is the median over the rows given, so the mode needs a whole split (not one coin)."""
    _need_bases(P, "volnorm")
    vol = _rel_vol(P)
    good = vol > 1e-6
    vol = np.where(good, vol, np.nanmedian(vol[good]) if good.any() else 1.0)
    return (P[_FUTURE[price]] / _own_last(P, price) - 1.0) / vol


@TARGET_RAW.decorator("magnitude", version=1, unit_scaled=True, clip=1.0, class_rule="threshold", needs_bases=False,
                      per_row=False, decode=_from_entry, signed=False, relative_ok=True)
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
                      decode=_from_entry, signed=False, relative_ok=False)
def _entry_range(P, price, close_reg):
    """Non-negative distances from the entry (last close); direction lives only in the class heads. Gaps (the
    horizon high below the entry, or its low above it) are zero, not negative."""
    entry = P["last_close"]
    if price in TARGET_COLUMNS[:2]:
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


def center_key(price: str) -> str:
    """The dataset key of the per-row centre a '+relative' retarget subtracted from head ``price`` (raw units)."""
    return f"{CENTER_PREFIX}{price}"


@dataclass
class TargetInfo:
    """What a call produced besides the arrays: the scale of every head, thresholds, group statistics and, for
    '+relative', the centre subtracted from every row of every head (raw units, float64)."""
    mode: str
    scales: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    group_size: np.ndarray | None = None
    clip: float = _DEFAULT_CLIP
    close_reg: str = ENTRY_CLOSE_REGS[0]
    centers: dict[str, np.ndarray] = field(default_factory=dict)


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
            info.centers[t] = c.astype("float64")
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
               scale: float = 1.0, *, sign=None, center=None, close_reg: str = ENTRY_CLOSE_REGS[0],
               range_prices=None) -> np.ndarray:
    """A predicted ``{price}_reg`` -> the future price of head `price`, for every mode.

    ``scale``: the head's scale (``reg_target_scales[price]``; ignored by the modes that are not unit scaled).
    ``sign``: +1 (up) / -1 (down) per row, needed by the close head of magnitude and entry_range (abs_return).
    ``center``: the per-row centre of a '+relative' mode (raw units, ``target_center_<price>`` of the dataset),
    needed by every '+relative' mode. ``range_prices``: ``(high_price, low_price)`` decoded from the high and low
    heads, needed by the close head of entry_range with close_reg 'range_pos'. A missing input raises."""
    base, relative = parse_mode(mode)
    meta = TARGET_RAW.entry(base).meta
    if close_reg not in ENTRY_CLOSE_REGS:
        raise ValueError(f"close_reg must be one of {ENTRY_CLOSE_REGS}, got {close_reg!r}")
    u = np.asarray(pred, dtype="float64") / (float(scale) if _is_unit(meta, price, close_reg) else 1.0)
    if relative:
        if center is None:
            raise ValueError(f"target mode {mode!r} subtracted the median of the same time: pass the per-row "
                             f"centre (dataset key {center_key(price)!r}) to decode a price")
        u = u + np.asarray(center, dtype="float64")
    P = price_columns(last, bases)
    if sign is not None:
        P["sign"] = np.asarray(sign, dtype="float64")
    P["close_reg"] = close_reg
    if range_prices is not None:
        P["range_high"], P["range_low"] = (np.asarray(v, dtype="float64") for v in range_prices)
    return meta["decode"](P, price, u)


def decode_affine(last: np.ndarray, bases: np.ndarray | None, mode: str, price: str, scale: float = 1.0, *,
                  sign=None, center=None, close_reg: str = ENTRY_CLOSE_REGS[0],
                  range_prices=None) -> tuple[np.ndarray, np.ndarray]:
    """``(anchor, slope)`` per row with ``decode_reg(reg) == anchor + slope * reg`` (every decode is affine in the
    prediction; the head's scale is inside ``slope``). Same arguments as ``decode_reg``. A consumer that decodes row
    by row (the eval path) keeps these two arrays instead of the dataset."""
    kw = {"sign": sign, "center": center, "close_reg": close_reg, "range_prices": range_prices}
    zero = np.zeros(len(np.asarray(last)), dtype="float64")
    a = decode_reg(zero, last, bases, mode, price, scale, **kw)
    return a, decode_reg(zero + 1.0, last, bases, mode, price, scale, **kw) - a


def entry_range_to_prices(last_close, high=None, low=None, close=None, close_reg: str = ENTRY_CLOSE_REGS[0],
                          p_close_up=None) -> dict[str, np.ndarray]:
    """Entry_range predictions (in return units: the model output divided by the head's scale) -> prices, P = last
    close: high = P (1 + high), low = P (1 - low); close (abs_return) = P (1 + s close) with s = +1 when
    ``p_close_up`` >= 0.5, else -1, and both directions as ``close_up`` / ``close_down`` (no ``close`` without
    ``p_close_up``); close (range_pos) = low + clip(close, 0, 1) (high - low), which needs high and low. Returns
    only what can be computed. The formulas are ``TARGET_RAW['entry_range'].meta['decode']``'s (one copy)."""
    if close_reg not in ENTRY_CLOSE_REGS:
        raise ValueError(f"close_reg must be one of {ENTRY_CLOSE_REGS}, got {close_reg!r}")
    decode = TARGET_RAW.entry("entry_range").meta["decode"]
    P = {"last_close": np.asarray(last_close, dtype="float64"), "close_reg": close_reg}
    out = {}
    if high is not None:
        out["high"] = decode(P, "high", np.asarray(high, dtype="float64"))
    if low is not None:
        out["low"] = decode(P, "low", np.asarray(low, dtype="float64"))
    if close is None:
        return out
    c = np.asarray(close, dtype="float64")
    if close_reg == ENTRY_CLOSE_REGS[1]:
        if "high" in out and "low" in out:
            out["close"] = decode({**P, "range_high": out["high"], "range_low": out["low"]}, "close", c)
        return out
    out["close_up"] = decode({**P, "sign": np.ones_like(c)}, "close", c)
    out["close_down"] = decode({**P, "sign": -np.ones_like(c)}, "close", c)
    if p_close_up is not None:
        out["close"] = np.where(np.asarray(p_close_up, dtype="float64") >= 0.5, out["close_up"], out["close_down"])
    return out


def _pipeline_of(dataset: dict) -> dict:
    """The pipeline's own y arrays and stamps, taken once (before the first retarget) and carried by reference."""
    if PIPELINE_TARGETS in dataset:
        return dataset[PIPELINE_TARGETS]
    ys = {k: dataset[k] for k in dataset
          if k.startswith(OUTPUT_PREFIX) and (k.endswith("_reg") or k.endswith("_class"))}
    return {"y": ys, "stamps": {k: dataset[k] for k in STAMP_KEYS if k in dataset}}


def _restore(dataset: dict) -> tuple[dict, dict]:
    pipe = _pipeline_of(dataset)
    out = {k: v for k, v in dataset.items() if k not in STAMP_KEYS and not k.startswith(CENTER_PREFIX)}
    out.update(pipe["y"])
    out.update(pipe["stamps"])
    out[PIPELINE_TARGETS] = pipe
    return out, {"mode": None, "restored": sorted(pipe["y"])}


def _fit_thresholds(last, bases, mode, targets, fit_idx, clip, scale, close_reg):
    base, relative = parse_mode(mode)
    if TARGET_RAW.entry(base).meta["class_rule"] not in _FIT_RULES or relative:
        return None
    idx = np.arange(len(last)) if fit_idx is None else np.asarray(fit_idx)
    _, fit = target_arrays(last[idx], None if bases is None else bases[idx], mode, targets, clip=clip, scale=scale,
                           close_reg=close_reg)
    return fit.thresholds


def _retarget_rows(dataset: dict, mode: str | None, parts: dict, fit_idx, *, clip, targets, close_reg, center,
                   group_freq, min_group, reg_target_scale) -> tuple[dict, dict]:
    """The one implementation behind ``retarget`` (parts = {}: the whole dataset is one group) and
    ``retarget_splits`` (parts = {name: row indices}): y rebuilt from last_candles, statistics fitted on ``fit_idx``."""
    if mode is None:
        return _restore(dataset)
    if dataset.get("price_norm_meta") is not None:
        raise ValueError("retarget does not support a dataset whose targets are standardised with train statistics "
                         "tied to its inputs (price_norm_meta); use mode None")
    targets = list(targets or dataset["price_targets"])
    given = dataset.get("return_target_scale", dataset.get("reg_target_scale", 1.0))
    scale = float(given if reg_target_scale is None else reg_target_scale)
    last, bases = np.asarray(dataset["last_candles"]), dataset.get("base_params")
    bases = None if bases is None else np.asarray(bases)
    base, _ = parse_mode(mode)
    meta = TARGET_RAW.entry(base).meta
    thr = _fit_thresholds(last, bases, mode, targets, fit_idx, clip, scale, close_reg)
    kw = {"clip": clip, "scale": scale, "close_reg": close_reg, "center": center, "group_freq": group_freq,
          "thresholds": thr}
    y, info = target_arrays(last, bases, mode, targets, **kw)
    centers = dict(info.centers)
    report = {"mode": mode, "clip": info.clip, "scales": info.scales, "thresholds": thr or {}}
    small = {}
    if info.group_size is not None:
        small[""] = float((info.group_size < min_group).mean())
    for name, idx in parts.items():
        idx = np.asarray(idx)
        if not len(idx):
            continue
        yy, ii = target_arrays(last[idx], None if bases is None else bases[idx], mode, targets, **kw)
        for k, v in yy.items():
            y[k][idx] = v
        for t, c in ii.centers.items():
            centers[t][idx] = c
        if ii.group_size is not None:
            small[name] = float((ii.group_size < min_group).mean())
    out = {k: v for k, v in dataset.items() if not k.startswith(CENTER_PREFIX)}
    out[PIPELINE_TARGETS] = _pipeline_of(dataset)
    for k, v in y.items():
        out[f"{OUTPUT_PREFIX}{k}"] = v
    for t, c in centers.items():
        out[center_key(t)] = c
    out.update(target_mode=mode, reg_target_scales=dict(info.scales), reg_target_clip=info.clip,
               return_target_scale=scale, reg_target_mode=mode,
               reg_target_scale=scale if _is_unit(meta, "high", close_reg) else 1.0)
    if meta["class_rule"] in _OWN_RULES:
        out["entry_close_reg"] = close_reg
    else:
        out.pop("entry_close_reg", None)
    if small:
        report["share_small_groups"] = small[""] if not parts else {k: v for k, v in small.items() if k}
    return out, report


def retarget(dataset: dict, mode: str | None, *, train_idx: np.ndarray | None = None, clip: float | None = None,
             targets=None, close_reg: str = ENTRY_CLOSE_REGS[0], center: str = "median", group_freq: str = "",
             min_group: int = 5, reg_target_scale: float | None = None) -> tuple[dict, dict]:
    """A copy of `dataset` (X shared, not copied) whose y arrays follow `mode`; returns it with a report dict.
    ``mode=None`` restores the pipeline's own y and stamps.

    ``train_idx`` fits the magnitude threshold (default: every row); ``min_group`` only feeds the report (share of
    rows whose time group has fewer coins than that, '+relative' only). The dataset is stamped with
    target_mode / reg_target_scale (the one the heads use) / reg_target_scales (per head) / return_target_scale
    (the scale of the return-unit modes, kept so a later retarget can go back) / reg_target_clip / entry_close_reg,
    and a '+relative' mode stores the subtracted centre of every row as ``target_center_<price>``."""
    return _retarget_rows(dataset, mode, {}, train_idx, clip=clip, targets=targets, close_reg=close_reg,
                          center=center, group_freq=group_freq, min_group=min_group,
                          reg_target_scale=reg_target_scale)


def retarget_splits(dataset: dict, splits: tuple, mode: str | None = None, *, clip: float | None = None, targets=None,
                    close_reg: str = ENTRY_CLOSE_REGS[0], center: str = "median", group_freq: str = "",
                    min_group: int = 5, reg_target_scale: float | None = None) -> tuple[dict, dict]:
    """``retarget`` on the (train, val, test) row sets of one split (index arrays into the dataset, as
    ``workflow.run_common.make_plan`` returns them). Every statistic (the magnitude threshold) is fitted on the TRAIN
    rows and reused on val and test; a '+relative' centre is the median over the rows of the same split at one time;
    rows in no split get the whole-dataset values (never read by a run). ``mode=None`` returns the pipeline targets;
    modes can be switched in any order (the arrays come from ``last_candles``, not from the previous y). The report
    holds per-split ``share_small_groups`` for '+relative'."""
    if len(splits) != 3:
        raise ValueError("splits must be (train, val, test) row index arrays")
    parts = dict(zip(("train", "val", "test"), splits, strict=True))
    return _retarget_rows(dataset, mode, parts, parts["train"], clip=clip, targets=targets, close_reg=close_reg,
                          center=center, group_freq=group_freq, min_group=min_group,
                          reg_target_scale=reg_target_scale)


def head_scale(dataset: dict, target: str) -> float:
    """The scale ``target``'s regression label was multiplied by: the stamped ``reg_target_scales[target]`` of a
    retargeted dataset, else the mode's rule applied to the dataset's ``reg_target_scale``."""
    stamped = dataset.get("reg_target_scales")
    if stamped is not None and target in stamped:
        return float(stamped[target])
    meta = TARGET_RAW.entry(parse_mode(dataset["reg_target_mode"])[0]).meta
    close_reg = dataset.get("entry_close_reg", ENTRY_CLOSE_REGS[0])
    return float(dataset["reg_target_scale"]) if _is_unit(meta, target, close_reg) else 1.0


def row_anchors(dataset: dict, idx: np.ndarray, targets, *, truth: bool = False) -> dict[str, np.ndarray]:
    """``{target: (n, 2)}`` = ``[anchor, slope]`` of the rows ``idx`` with ``price = anchor + slope * reg`` (the head's
    scale is inside the slope), from the dataset's stamped mode: the one input of ``TargetModeCodec.decode``.

    ``truth=False`` (evaluation; nothing of the future enters): the close head of magnitude / entry_range is decoded
    UP (sign +1; read only ``|price - entry|`` and widths there), a '+relative' mode decodes with centre 0 (the
    market-neutral price), the range_pos close head is NaN (it needs the predicted range). ``truth=True`` (a decode
    check): the true sign, the stored ``target_center_<t>`` and the true high / low as the range."""
    idx = np.asarray(idx)
    mode = dataset["reg_target_mode"]
    close_reg = dataset.get("entry_close_reg", ENTRY_CLOSE_REGS[0])
    relative = parse_mode(mode)[1]
    last = np.asarray(dataset["last_candles"])[idx]
    bases = dataset.get("base_params")
    bases = None if bases is None else np.asarray(bases)[idx]
    P = price_columns(last, bases)
    n = len(idx)
    sign = np.where(P["future_close"] >= P["last_close"], 1.0, -1.0) if truth else np.ones(n)
    range_prices = (P["future_high_max"], P["future_low_min"]) if truth else (np.full(n, np.nan),) * 2
    out = {}
    for t in targets:
        centre = None
        if relative:
            centre = np.asarray(dataset[center_key(t)])[idx] if truth else np.zeros(n)
        a, s = decode_affine(last, bases, mode, t, head_scale(dataset, t), sign=sign, center=centre,
                             close_reg=close_reg, range_prices=range_prices)
        out[t] = np.stack([a, s], axis=1)
    return out


class TargetModeCodec:
    """The ``TARGET_CODECS`` entry of every ``TARGET_RAW`` mode but the own-anchor "return" (data_io.target_codec):
    state = the dataset's stamps, decode = ``anchor + slope * pred`` on the rows ``row_anchors`` gives. Labels are
    never encoded here (``retarget`` / ``target_arrays`` are the encoder); nothing is re-implemented."""
    name = "target_raw"
    anchor = "rows"
    row_anchors = True     # its anchors are the (n, 2) rows of ``row_anchors`` (the price codecs' are a price column)

    def fit(self, train: dict) -> dict:
        for key in ("reg_target_mode", "reg_target_scale"):
            if key not in train:
                raise ValueError(f"the dataset lacks {key!r}: the target-mode codec cannot decode without it")
        mode = train["reg_target_mode"]
        base, relative = parse_mode(mode)
        meta = TARGET_RAW.entry(base).meta
        clip = train.get("reg_target_clip")
        return {"mode": self.name, "version": 1, "target_mode": mode, "scale": float(train["reg_target_scale"]),
                "clip": float(meta["clip"] if clip is None else clip),
                "scales": {t: head_scale(train, t) for t in TARGET_COLUMNS},
                "close_reg": train.get("entry_close_reg", ENTRY_CLOSE_REGS[0]),
                "price_forecast": bool(meta["signed"] and not relative)}

    def _check(self, state: dict) -> None:
        if state.get("mode") != self.name:
            raise ValueError(f"state is for codec {state.get('mode')!r}, not {self.name!r}")

    def encode(self, raw, state: dict):
        raise NotImplementedError("labels of a target mode are produced by data.targets.retarget / target_arrays")

    def decode(self, pred: np.ndarray, state: dict, anchors: np.ndarray) -> np.ndarray:
        self._check(state)
        a = np.asarray(anchors, dtype="float64")
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError("anchors of the target-mode codec are (n, 2) = [anchor, slope] (data.targets.row_anchors)")
        return a[:, 0] + a[:, 1] * np.asarray(pred, dtype="float64")

    def forecasts_price(self, state: dict) -> bool:
        """True when ``decode(mu)`` is a price forecast on its own (a signed mode without '+relative'); False when
        the sign or the market level is missing, so a trading rule must not read it as the expected price."""
        self._check(state)
        return bool(state["price_forecast"])

    def eval_spec(self, state: dict) -> dict:
        self._check(state)
        return {"mode": self.name, "target_mode": state["target_mode"], "scale": state["scale"],
                "clip": state["clip"], "decoded": "price" if state["price_forecast"] else "magnitude only",
                "anchor": "per row [anchor, slope] from data.targets.row_anchors", "round_trip_exact": False}


TARGET_CODECS.register(TargetModeCodec.name, TargetModeCodec, version=1, supports_relative=True)
