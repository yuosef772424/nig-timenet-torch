"""PURPOSE: One finished stage decoded and judged as the TF chicks code did: price, entry, NIG uncertainties,
    confidence, direction hit and error per head, and the decoding verification.
TAGS: chicks, decode stage, evaluate predictions, verify decoding, entry price, uncertainty real, nig bounded,
    direction correct, round trip, tf parity
PITFALLS: needs a codec whose decode is a price forecast (``forecasts_price``: not magnitude / entry_range /
    '+relative'); the entry of a head is the anchor the codec decodes it against (own last price for "return", the
    last close for "logret_close"); the true price is the dataset's future column (``future_close`` /
    ``future_high_max`` / ``future_low_min``), the decoded label only when that column is all NaN; the price slope
    (the TF ``iqr``) is a central difference of the codec's decode (exact for the linear codec, ~1e-11 for the log
    one); uncertainty is recomputed from nu / alpha / beta and capped at ``unc_max`` (20.0), as the TF evaluation did,
    not read from the table's own ``epistemic`` / ``aleatoric`` columns; the verification round trip compares the
    decoded price with its first-order expansion, so it is only meaningful for a linear codec (it reports 0 for a
    log one); a label's clip makes ``true_value_consistent`` tolerate 1 % of rows off by more than 1e-3; ``p_up`` (the
    class head's P(up)) is NaN unless ``class_head`` is set: the TF 'return' mode did not read it.
SOURCE: crypto-signal-prediction ``evaluation/decode.py`` (decode_predictions_v4), ``evaluation/metrics.py``
    (evaluate_predictions_v4) and ``evaluation/verify.py`` (verify_decoding); the model's raw outputs are the table's
    columns, the decode is the registered target codec (rule B4), so the TF ``base_params`` / ``TargetSpec`` are gone.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import FUTURE_OF, LAST_COLUMN_INDEX, TS_COL
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.stats import metrics
from nig_timenet_torch.stats.nig import NIG_UNC_MAX, nig_uncertainty_bounded

__all__ = ["StageData", "decode_stage", "slice_decoded", "evaluate_decoded", "verify_decoding", "nan_mean", "nan_max",
           "DIRECTION_TOLERANCE", "SLOPE_STEP"]

DIRECTION_TOLERANCE = 0.2     # the TF ``range_frac``: a prediction within this share of the true high - low is "within"
SLOPE_STEP = 1e-3             # half-width of the central difference that gives the price slope of the codec


@dataclass(frozen=True)
class StageData:
    """What one stage of a finished run holds for the chicks report: ``table`` (PredictionTable, one row per sample),
    ``labels`` (label key -> array aligned to the rows), ``anchors`` (target -> what the codec decodes against: (n,)
    prices, or (n, 2) rows), ``last_candles`` (the dataset's ``last_candles`` rows of the stage), ``codec_state``."""
    table: pd.DataFrame
    labels: dict
    anchors: dict
    last_candles: np.ndarray
    codec_state: dict
    targets: tuple

    def __post_init__(self):
        n = len(self.table)
        if len(self.last_candles) != n:
            raise ValueError(f"StageData: last_candles has {len(self.last_candles)} rows, the table {n}")
        for t in self.targets:
            if len(self.anchors[t]) != n:
                raise ValueError(f"StageData: the anchor of {t!r} has {len(self.anchors[t])} rows, the table {n}")
            key = head_spec("nig_regression", t).label_key
            if key not in self.labels or len(self.labels[key]) != n:
                raise ValueError(f"StageData: no label {key!r} aligned to the {n} rows")


def nan_mean(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    m = np.isfinite(x)
    return float(x[m].mean()) if m.any() else float("nan")


def nan_max(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    m = np.isfinite(x)
    return float(x[m].max()) if m.any() else float("nan")


def _column(table: pd.DataFrame, key: str) -> np.ndarray:
    if key not in table.columns:
        raise ValueError(f"the prediction table has no column {key!r}")
    return table[key].to_numpy(dtype=np.float64)


def _slope(codec, state: dict, raw: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """d price / d raw output of the codec at ``raw`` (the TF ``iqr``), by a central difference."""
    h = SLOPE_STEP
    return (codec.decode(raw + h, state, anchor) - codec.decode(raw - h, state, anchor)) / (2.0 * h)


def _decode_head(stage: StageData, codec, target: str, unc_max: float, class_head: bool) -> dict:
    state, table = stage.codec_state, stage.table
    reg = head_spec("nig_regression", target)
    raw = _column(table, reg.output_keys["mu"])
    anchor = np.asarray(stage.anchors[target], dtype=np.float64)
    entry = anchor if anchor.ndim == 1 else anchor[:, 0]
    pred = codec.decode(raw, state, anchor)
    slope = _slope(codec, state, raw, anchor)
    epi, ale, clipped = nig_uncertainty_bounded(*(_column(table, reg.output_keys[k]) for k in ("nu", "alpha", "beta")),
                                                unc_max)
    total = np.sqrt(epi ** 2 + ale ** 2)
    cls_key = head_spec("binary_classification", target).output_keys["logits"]
    lc = np.asarray(stage.last_candles, dtype=np.float64)
    future = lc[:, LAST_COLUMN_INDEX[FUTURE_OF[target]]]
    if np.all(np.isnan(future)):                   # a stage without future candles: the decoded label is the truth
        future = codec.decode(np.asarray(stage.labels[reg.label_key], dtype=np.float64), state, anchor)
    return {"kind": "continuous", "pred_scaled": raw, "pred_real": pred, "entry_price": entry,
            "change_from_entry": (pred - entry) / (np.abs(entry) + 1e-7) * 100.0, "target_real": future,
            "median": codec.decode(np.zeros_like(raw), state, anchor), "iqr": slope,
            "epistemic": epi * np.abs(slope), "aleatoric": ale * np.abs(slope),
            "uncertainty_real": total * np.abs(slope), "uncertainty_scaled": total,
            "confidence": _column(table, reg.output_keys["confidence"]), "unc_clipped": clipped,
            "p_up": _column(table, cls_key) if class_head and cls_key in table.columns else np.full(len(table), np.nan),
            "timestamp": lc[:, TS_COL], "y_scaled": np.asarray(stage.labels[reg.label_key], dtype=np.float64)}


def decode_stage(stage: StageData, unc_max: float = NIG_UNC_MAX, class_head: bool = False) -> dict:
    """{target: decoded fields over ALL rows of the stage} (the TF ``decode_predictions_v4`` of the pooled rows).
    ``class_head``: also read each head's class output as ``p_up`` (the TF code did it only for entry_range targets;
    without it ``p_up`` is NaN and pattern discovery / the calibration fallback do not see the class head)."""
    codec = TARGET_CODECS[stage.codec_state["mode"]]()
    if not codec.forecasts_price(stage.codec_state):
        name = stage.codec_state.get("target_mode", stage.codec_state["mode"])
        raise ValueError(f"chicks: the {name!r} target does not decode to a price forecast (no sign or no market "
                         "level), the report cannot read it")
    return {t: _decode_head(stage, codec, t, unc_max, class_head) for t in stage.targets}


def slice_decoded(decoded: dict, sl) -> dict:
    """The decoded fields of the rows ``sl`` (a slice or an index array) of every target."""
    return {t: {k: (v[sl] if isinstance(v, np.ndarray) else v) for k, v in r.items()} for t, r in decoded.items()}


def evaluate_decoded(decoded: dict, direction_tolerance: float = DIRECTION_TOLERANCE) -> dict:
    """The TF ``evaluate_predictions_v4`` of continuous targets: adds the true price, the validity mask, the absolute
    and percent error, the direction hit, the within-tolerance flag, the correlation of predicted and true change,
    the win rate and the share of errors within 1 / 2 predicted standard deviations (mutates and returns
    ``decoded``)."""
    high, low = decoded.get("high", {}).get("target_real"), decoded.get("low", {}).get("target_real")
    actual_range = None
    if high is not None and low is not None and not (np.all(np.isnan(high)) or np.all(np.isnan(low))):
        actual_range = np.maximum(high - low, 1e-7)
    for r in decoded.values():
        true_val = np.asarray(r["target_real"], dtype=np.float64)
        entry, pred_real = np.asarray(r["entry_price"]), np.asarray(r["pred_real"])
        change = np.asarray(r["change_from_entry"])
        valid = np.isfinite(true_val) & np.isfinite(pred_real) & np.isfinite(entry)
        abs_error = np.abs(true_val - pred_real)
        true_change_pct = (true_val - entry) / (np.abs(entry) + 1e-7) * 100.0
        direction_correct = ((np.sign(change) == np.sign(true_change_pct)) & valid).astype(int)
        tol = direction_tolerance * (actual_range if actual_range is not None else np.abs(true_val))
        corr = 0.0
        if valid.sum() > 2 and np.std(change[valid]) > 1e-7 and np.std(true_change_pct[valid]) > 1e-7:
            c = metrics.pearson(change[valid], true_change_pct[valid])
            corr = 0.0 if np.isnan(c) else c
        unc = np.asarray(r["uncertainty_real"])
        within_1 = within_2 = float("nan")
        if valid.any():
            within_1 = float(np.mean(abs_error[valid] <= unc[valid]) * 100)
            within_2 = float(np.mean(abs_error[valid] <= 2 * unc[valid]) * 100)
        r.update({"true_real": true_val, "has_truth": valid, "n_valid": int(valid.sum()),
                  "true_change_from_entry_pct": true_change_pct, "error_abs": abs_error,
                  "error_pct": abs_error / (np.abs(true_val) + 1e-7) * 100.0, "same_direction": direction_correct,
                  "within_tolerance": ((abs_error <= tol) & valid).astype(int), "direction_correct": direction_correct,
                  "correlation": corr, "win_rate": float(np.mean(direction_correct[valid]) * 100) if valid.any()
                  else float("nan"), "uncertainty_pct": unc / (np.abs(pred_real) + 1e-7) * 100.0,
                  "within_1_std": within_1, "within_2_std": within_2})
    return decoded


def verify_decoding(decoded: dict, atol: float = 1e-4) -> dict:
    """{target: report} of the decoding checks of the TF ``verify_decoding``, for already evaluated heads: the round
    trip of the price from the raw output (``median + iqr * raw``), the decoded label against the true price, the
    errors in the scaled and the price space and their ratio (about 1 when the decode is right)."""
    report = {}
    for name, d in decoded.items():
        rep = {"target": name, "kind": "continuous"}
        recomputed = d["median"] + d["iqr"] * d["pred_scaled"]
        linear = bool(np.all(np.abs(recomputed - d["pred_real"]) <= 1e-6 * (1.0 + np.abs(d["pred_real"]))))
        diff = np.abs(recomputed - d["pred_real"]) if linear else np.zeros(len(recomputed))
        rep["roundtrip_max_diff"] = float(diff.max())
        rep["roundtrip_ok"] = bool(np.all(diff <= atol))
        true_scaled, true_real = d["y_scaled"], d["target_real"]
        err_scaled = np.abs(d["pred_scaled"] - true_scaled)
        rep["mae_scaled"] = nan_mean(err_scaled)
        if not np.all(np.isnan(true_real)):
            from_scaled = d["median"] + d["iqr"] * true_scaled
            gap = np.abs(from_scaled - true_real)
            rep["true_value_consistency_max_diff"] = nan_max(gap)
            rel = gap / (np.abs(true_real) + 1e-12)
            finite = np.isfinite(rel)
            bad = float(np.mean(rel[finite] > 1e-3)) if finite.any() else 0.0
            rep["true_value_inconsistent_frac"] = bad
            rep["true_value_consistent"] = bool(bad <= 0.01)
            err_real = np.abs(d["pred_real"] - true_real)
            rep["mae_real"] = nan_mean(err_real)
            denom = err_scaled * np.abs(d["iqr"])
            rep["space_consistency_ratio_mean"] = nan_mean(err_real / np.where(denom < 1e-9, np.nan, denom))
        rep["status"] = "ok" if rep["roundtrip_ok"] and rep.get("true_value_consistent", True) else "FAILED"
        report[name] = rep
    return report
