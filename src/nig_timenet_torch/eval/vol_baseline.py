"""PURPOSE: Rule C1 for the high / low heads: volatility baselines from the model's own input window (last-bar range,
    HAR means, ATR-type true range), the model's rank IC net of them, a HAR OLS table and 80 % intervals at matched
    calibration.
TAGS: volatility baseline, rule c1, high low heads, har, atr, partial rank ic, interval sharpness, range forecast
PITFALLS: the X channels must be log(price / previous close) per bar (checked: high >= low on every bar), else the
    ranges are meaningless; OLS and interval scales are fitted on VAL and scored on TEST only; the low head is
    judged on -pred / -fut so a bigger move ranks higher for both heads (direction is never read here).

A range model that only learned "volatile coins move more" ties these baselines. ``window_ranges`` builds them from
the SAME 32-bar window the model sees, ``head_vs_baselines`` compares rank ICs (cross-sectional, by timestamp),
``har_table`` reproduces the HAR range regression (OLS on VAL, R2 / Spearman / within-coin Spearman on TEST) and
``interval_vs_baseline`` compares 80 % intervals after both are calibrated on VAL.
"""

from __future__ import annotations

import numpy as np

from nig_timenet_torch.stats import metrics

__all__ = ["window_ranges", "head_vs_baselines", "within_coin_spearman", "ols_fit", "har_table",
           "interval_vs_baseline", "Z80_LEVEL"]

Z80_LEVEL = 0.8
EPS = 1e-6


def window_ranges(x: np.ndarray, feature_order) -> dict[str, np.ndarray]:
    """Per-row range predictors (log units) from the input window ``x`` (n, T, F) whose channels are
    log(high / prev close), log(low / prev close), ...: ``bar1`` = last bar log(high/low), ``mean6`` / ``mean32`` =
    mean of the last 6 / all bars' ranges, ``atr14`` = mean true range of the last 14 bars. Raises when a bar has
    high < low (the channels are not raw log prices then)."""
    fo = list(feature_order)
    xh = np.asarray(x[:, :, fo.index("high")], dtype=np.float64)
    xl = np.asarray(x[:, :, fo.index("low")], dtype=np.float64)
    r = xh - xl
    if (r < -1e-9).any():
        raise ValueError("window_ranges: some bars have high < low; the input channels are not log(price/prev close)")
    tr = np.maximum(xh, 0.0) - np.minimum(xl, 0.0)
    return {"bar1": r[:, -1], "mean6": r[:, -6:].mean(axis=1), "mean32": r.mean(axis=1),
            "atr14": tr[:, -14:].mean(axis=1)}


def within_coin_spearman(y: np.ndarray, p: np.ndarray, coins: np.ndarray) -> float:
    """Spearman of the per-coin demeaned values: timing skill inside each coin, not the coin's volatility level."""
    y, p, c = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64), np.asarray(coins)
    _, inv = np.unique(c, return_inverse=True)

    def dm(a):
        return a - (np.bincount(inv, weights=a) / np.bincount(inv))[inv]
    return metrics.spearman(dm(y), dm(p))


def head_vs_baselines(pred: np.ndarray, fut: np.ndarray, baselines: dict, groups: np.ndarray, coins: np.ndarray,
                      sign: float = 1.0) -> dict:
    """Rank IC by timestamp of the model and of each baseline on the realised extreme ``fut`` (``sign`` -1 for the
    low head), the model's partial rank IC net of each baseline, and within-coin Spearman of all of them."""
    y, p = sign * np.asarray(fut, dtype=np.float64), sign * np.asarray(pred, dtype=np.float64)
    out = {"model": {"rank_ic": metrics.rank_ic(y, p, groups), "within_coin": within_coin_spearman(y, p, coins)}}
    for name, b in baselines.items():
        out[name] = {"rank_ic": metrics.rank_ic(y, b, groups), "within_coin": within_coin_spearman(y, b, coins),
                     "model_net_of_it": metrics.partial_rank_ic(y, p, b, groups)}
    return out


def ols_fit(features: dict, y: np.ndarray, names) -> np.ndarray:
    """Least-squares weights (intercept first) of ``y`` on the named feature columns."""
    a = np.column_stack([np.ones(len(y))] + [np.asarray(features[n], dtype=np.float64) for n in names])
    return np.linalg.lstsq(a, np.asarray(y, dtype=np.float64), rcond=None)[0]


def _ols_predict(features: dict, w: np.ndarray, names) -> np.ndarray:
    a = np.column_stack([np.ones(len(next(iter(features.values()))))] + [np.asarray(features[n]) for n in names])
    return a @ w


def har_table(feat_val: dict, y_val: np.ndarray, feat_test: dict, y_test: np.ndarray, coins_test: np.ndarray,
              sets: dict) -> dict:
    """For each predictor set {name: [feature names]}: OLS fitted on VAL, scored on TEST (R2, Spearman,
    within-coin Spearman). Features and targets are passed already transformed (e.g. log ranges)."""
    out = {}
    for name, names in sets.items():
        p = _ols_predict(feat_test, ols_fit(feat_val, y_val, names), names)
        out[name] = {"r2": metrics.r2(y_test, p), "spearman": metrics.spearman(y_test, p),
                     "within_coin": within_coin_spearman(y_test, p, coins_test)}
    return out


def _calibrated(center_v, scale_v, y_v, center_t, scale_t, y_t, level: float) -> dict:
    k = float(np.quantile(np.abs(y_v - center_v) / scale_v, level))
    half = k * scale_t
    return {"k": k, "coverage_val": metrics.interval_coverage(y_v, center_v - k * scale_v, center_v + k * scale_v),
            "coverage_test": metrics.interval_coverage(y_t, center_t - half, center_t + half),
            "mean_halfwidth_test": float(np.mean(half)), "median_halfwidth_test": float(np.median(half))}


def interval_vs_baseline(val: dict, test: dict, level: float = Z80_LEVEL) -> dict:
    """80 % intervals of the model and of a volatility-only model, both calibrated on VAL (one scale factor each, so
    VAL coverage is exactly ``level``) and scored on TEST: the narrower one at the same coverage is the sharper.
    ``val`` / ``test``: {"y": realised, "pred", "std": model centre and std, "base": {feature: array}} in return
    units. The baseline centre is OLS of y on the features, its scale OLS of |residual| on them (VAL fits)."""
    names = sorted(val["base"])
    w_c = ols_fit(val["base"], val["y"], names)
    c_v, c_t = _ols_predict(val["base"], w_c, names), _ols_predict(test["base"], w_c, names)
    w_s = ols_fit(val["base"], np.abs(val["y"] - c_v), names)
    s_v = np.maximum(_ols_predict(val["base"], w_s, names), EPS)
    s_t = np.maximum(_ols_predict(test["base"], w_s, names), EPS)
    model = _calibrated(val["pred"], val["std"], val["y"], test["pred"], test["std"], test["y"], level)
    base = _calibrated(c_v, s_v, val["y"], c_t, s_t, test["y"], level)
    model["share_std_below_move_test"] = metrics.width_below_share(test["std"], np.abs(test["pred"]))
    base["share_std_below_move_test"] = metrics.width_below_share(s_t, np.abs(c_t))
    return {"level": level, "model": model, "volatility_baseline": base,
            "halfwidth_ratio_model_over_baseline": model["mean_halfwidth_test"] / base["mean_halfwidth_test"]}
