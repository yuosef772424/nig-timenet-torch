"""T4 target report: the numbers of the ROADMAP "Targets" block (T-skill, T-conf, T-sharp + coverage) for one stage
(val or test) of one run, from a validated PredictionTable and its labels. Pure numpy; every metric lives in
``stats.metrics`` (C2) and every output key is derived from the head registry (G1).

Inputs: ``table`` (PredictionTable, ``core.prediction_schema.validate_table`` is applied), ``labels`` (label key ->
array aligned to the table rows: the encoded regression label and the class label of each head), ``anchors``
(price target -> own last price, aligned) and ``codec_state`` (the TargetCodec state the labels were encoded with).
T-conf is measured on the encoded scale, within coin; T-sharp compares widths in PRICE units (decoded)."""

import numpy as np
import pandas as pd

from nig_timenet_torch.core.prediction_schema import validate_table
from nig_timenet_torch.core.schema import TARGET_COLUMNS
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.stats import metrics
from nig_timenet_torch.stats.nig import nig_interval_halfwidth, nig_predictive_std

__all__ = ["target_report", "T_CONF_SPEARMAN_MAX", "T_SHARP_MIN_SHARE", "COVERAGE_RANGE"]

T_CONF_SPEARMAN_MAX = -0.8                  # ROADMAP T-conf
T_SHARP_MIN_SHARE = 0.80                    # ROADMAP T-sharp
COVERAGE_RANGE = (0.75, 0.85)               # ROADMAP T-sharp, coverage of the 80 % interval
INTERVAL_LEVEL = 0.8


def _col(table: pd.DataFrame, key: str) -> np.ndarray:
    if key not in table.columns:
        raise KeyError(f"table has no column {key!r}")
    return table[key].to_numpy(dtype=np.float64)


def _label(labels, key: str, n: int) -> np.ndarray:
    if key not in labels:
        raise KeyError(f"labels has no {key!r}")
    arr = np.asarray(labels[key], dtype=np.float64).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"label {key!r} has {arr.shape[0]} rows, the table has {n}")
    return arr


def _per_coin_spearman(y, p, assets) -> dict:
    vals = [metrics.spearman(y[assets == a], p[assets == a]) for a in np.unique(assets)]
    vals = np.asarray([v for v in vals if not np.isnan(v)])
    return {"mean": float(vals.mean()) if vals.size else float("nan"), "n_coins": int(vals.size),
            "share_positive": float((vals > 0).mean()) if vals.size else float("nan")}


def _skill(table, labels, spec_reg, spec_cls, seed: int) -> dict:
    n = len(table)
    y, pred = _label(labels, spec_reg.label_key, n), _col(table, spec_reg.output_keys["mu"])
    assets = table["asset"].to_numpy(dtype=object)
    out = {"rank_ic_by_timestamp": metrics.rank_ic(y, pred, groups=table["timestamp"].to_numpy()),
           "rank_ic_by_timestamp_shuffled_baseline": metrics.rank_ic(
               y, np.random.default_rng(seed).permutation(pred), groups=table["timestamp"].to_numpy()),
           "spearman_per_coin": _per_coin_spearman(y, pred, assets)}
    if spec_cls is not None and spec_cls.output_keys["logits"] in table.columns:
        yc, score = _label(labels, spec_cls.label_key, n), _col(table, spec_cls.output_keys["logits"])
        out["class"] = {"auc": metrics.auc(yc, score), "auc_constant_baseline": metrics.auc(yc, np.zeros(n)),
                        "auc_shuffled_baseline": metrics.auc(yc, np.random.default_rng(seed).permutation(score)),
                        "base_rate": float(np.nanmean(yc))}
    return out


def _conf(table, labels, spec, n_bins: int, seed: int) -> dict:
    n = len(table)
    err = np.abs(_col(table, spec.output_keys["mu"]) - _label(labels, spec.label_key, n))
    conf, assets = _col(table, spec.output_keys["confidence"]), table["asset"].to_numpy(dtype=object)
    rho = metrics.confidence_error_spearman(err, conf, assets, n_bins)
    gap, lo, hi = metrics.confidence_error_gap(err, conf, assets, n_bins, seed=seed)
    return {"decile_error_spearman": rho, "top_minus_bottom_error_gap": gap, "gap_ci95": [lo, hi],
            "constant_confidence_baseline": 0.0,
            "meets_target": bool(rho <= T_CONF_SPEARMAN_MAX and lo > 0.0)}


def _price_width(codec, state, mu, std, anchor) -> np.ndarray:
    return codec.decode(mu + std, state, anchor) - codec.decode(mu, state, anchor)


def _sharp(table, labels, specs, anchors, codec, state, level: float, name: str) -> dict:
    n, spec = len(table), specs[name]
    mu = _col(table, spec.output_keys["mu"])
    nu, alpha, beta = (_col(table, spec.output_keys[k]) for k in ("nu", "alpha", "beta"))
    std, half = nig_predictive_std(nu, alpha, beta), nig_interval_halfwidth(nu, alpha, beta, level)
    y = _label(labels, spec.label_key, n)
    cov = metrics.interval_coverage(y, mu - half, mu + half)
    in_range = bool(COVERAGE_RANGE[0] <= cov <= COVERAGE_RANGE[1])
    out = {"coverage": cov, "level": level, "coverage_within_target": in_range,
           "mean_interval_halfwidth": float(np.nanmean(half))}
    assets = table["asset"].to_numpy(dtype=object)
    per = [metrics.interval_coverage(y[assets == a], (mu - half)[assets == a], (mu + half)[assets == a])
           for a in np.unique(assets)]
    out["coverage_by_coin_min_max"] = [float(np.nanmin(per)), float(np.nanmax(per))]
    if {"high", "low"} <= set(specs):
        mus = {t: _col(table, specs[t].output_keys["mu"]) for t in ("high", "low")}
        rng_price = (codec.decode(mus["high"], state, np.asarray(anchors["high"], dtype=np.float64))
                     - codec.decode(mus["low"], state, np.asarray(anchors["low"], dtype=np.float64)))
        width = _price_width(codec, state, mu, std, np.asarray(anchors[name], dtype=np.float64))
        share = metrics.width_below_share(width, rng_price)
        out.update({"share_width_below_range": share, "share_target": T_SHARP_MIN_SHARE,
                    "zero_width_baseline_share": metrics.width_below_share(np.zeros(n), rng_price),
                    "meets_target": bool(share >= T_SHARP_MIN_SHARE and out["coverage_within_target"])})
    return out


def target_report(table: pd.DataFrame, labels: dict, *, anchors: dict, codec_state: dict, n_bins: int = 10,
                  seed: int = 0) -> dict:
    """{"n_rows", "targets": {target: {"skill", "conf", "sharp"}}}. A price target is reported when its NIG head
    columns are in the table; the class part of "skill" only when its class column is. T-sharp needs the high AND low
    heads (the range), otherwise "sharp" holds coverage only. An empty table raises."""
    validate_table(table)
    if len(table) == 0:
        raise ValueError("target_report: empty table")
    codec = TARGET_CODECS[codec_state["mode"]]()
    specs = {t: head_spec("nig_regression", t) for t in TARGET_COLUMNS
             if head_spec("nig_regression", t).output_keys["mu"] in table.columns}
    if not specs:
        raise ValueError("target_report: the table has no NIG regression head columns")
    out = {"n_rows": len(table), "targets": {}}
    for t, spec in specs.items():
        out["targets"][t] = {
            "skill": _skill(table, labels, spec, head_spec("binary_classification", t), seed),
            "conf": _conf(table, labels, spec, n_bins, seed),
            "sharp": _sharp(table, labels, specs, anchors, codec, codec_state, INTERVAL_LEVEL, t)}
    return out
