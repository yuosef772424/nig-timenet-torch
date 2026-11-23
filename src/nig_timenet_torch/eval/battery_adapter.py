"""Battery adapter: a validated PredictionTable (+ labels, last close, codec state) -> the battery's tidy table

PURPOSE: Turns one stage of a run into the table the battery reads: close-only mu / sigma / p_dir / y_ret in return
    units, plus the high / low predicted and realised extremes the limit-order sections need.
TAGS: battery adapter, prediction table, close return, nig predictive std, limit order inputs, target codec
PITFALLS: needs a codec whose decode is a price forecast (``forecasts_price``: not magnitude / entry_range /
    '+relative', which raise); direction (p_dir, dir_true, mu sign) comes from the CLOSE head only; high / low enter
    as price levels, never as a direction (owner, rule D). sigma is the NIG predictive std, not the capped model
    uncertainty. A missing close class column falls back to a Gaussian P(return > 0), flagged in attrs.

Column mapping (return = price / last_close - 1, prices decoded with the head's OWN anchor via the TargetCodec):
step = 1 (one horizon per row); mu = decode(close mu) return; sigma = (decode(mu + std) - decode(mu)) / last_close
with std = nig_predictive_std(nu, alpha, beta); p_dir = the close class head output (a probability, see
model/heads.py) or the Gaussian fallback; y_ret = decoded close label return; dir_true = sign(y_ret); last_ret =
last_close / prev_close - 1 (only when ``prev_close`` is given); coin = asset; end_time = prediction_made_at;
last_close = the price the returns are over (price = last_close * (1 + return); the TF price-level
plots need it, older tables lack it); atr = the mean true range of the last 14 input bars (log units per bar), only
when the caller passes it (``workflow.battery_run.stage_inputs`` does).
When the high / low heads are in the table: pred_high / pred_low (decoded mu return), fut_high / fut_low (decoded
label return = the realised extreme over the horizon), sigma_high / sigma_low.
When the LS1 auxiliary close heads are in the table (model ``aux_heads``): pay_long / pay_short (each side's predicted
payoff after cost, return units, as the head outputs it) and q_up (the upper-quantile head, log -> return units). Both
assume a log-return close label (codec logret_close, the artifact's).
"""

import math

import numpy as np
import pandas as pd

from nig_timenet_torch.core.prediction_schema import validate_table
from nig_timenet_torch.core.schema import TARGET_COLUMNS
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.stats.nig import nig_predictive_std

__all__ = ["to_battery_table", "BATTERY_COLUMNS", "LIMIT_COLUMNS"]

BATTERY_COLUMNS = ("step", "mu", "sigma", "p_dir", "y_ret", "dir_true", "last_ret", "coin", "end_time",
                   "last_close")
LIMIT_COLUMNS = ("pred_high", "pred_low", "fut_high", "fut_low", "sigma_high", "sigma_low")


def _col(table: pd.DataFrame, key: str) -> np.ndarray:
    if key not in table.columns:
        raise KeyError(f"table has no column {key!r}")
    return table[key].to_numpy(dtype=np.float64)


def _label(labels: dict, key: str, n: int) -> np.ndarray:
    if key not in labels:
        raise KeyError(f"labels has no {key!r}")
    arr = np.asarray(labels[key], dtype=np.float64).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"label {key!r} has {arr.shape[0]} rows, the table has {n}")
    return arr


def _head(table, labels, anchors, last_close, codec, state, target: str):
    """(predicted return, realised return, predictive std in return units) of one NIG head."""
    spec = head_spec("nig_regression", target)
    n = len(table)
    mu = _col(table, spec.output_keys["mu"])
    std = nig_predictive_std(*(_col(table, spec.output_keys[k]) for k in ("nu", "alpha", "beta")))
    anchor = np.asarray(anchors[target], dtype=np.float64)   # (n,) prices, or (n, 2) rows of the target-mode codec
    if anchor.shape[0] != n:
        raise ValueError(f"anchor of {target!r} has {anchor.shape[0]} rows, the table has {n}")
    pred = codec.decode(mu, state, anchor) / last_close - 1.0
    real = codec.decode(_label(labels, spec.label_key, n), state, anchor) / last_close - 1.0
    width = (codec.decode(mu + std, state, anchor) - codec.decode(mu, state, anchor)) / last_close
    return pred, real, width


def _gaussian_p_up(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    z = mu / np.where(sigma > 0, sigma, np.nan)
    return np.array([0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) if np.isfinite(v) else 0.5 for v in z])


def to_battery_table(table: pd.DataFrame, labels: dict, *, anchors: dict, last_close, codec_state: dict,
                     prev_close=None, atr=None) -> pd.DataFrame:
    """The battery table of one stage. ``table`` is validated; ``labels`` (label key -> encoded array aligned to
    the rows), ``anchors`` (target -> own last price per row), ``last_close`` (per row, finite > 0) and
    ``codec_state`` are the same inputs ``eval.target_report.target_report`` takes. The close head is required;
    high / low are used when present. ``prev_close`` (the close before ``last_close``) adds ``last_ret``; ``atr`` (per
    row, the mean true range of the last input bars in log units, ``eval.vol_baseline.window_ranges``) adds ``atr``
    for the take-profit-at-ATR section."""
    codec = TARGET_CODECS[codec_state["mode"]]()
    if not codec.forecasts_price(codec_state):
        name = codec_state.get("target_mode", codec_state["mode"])
        raise ValueError(f"to_battery_table: the {name!r} target does not decode to a price forecast (no sign or no "
                         "market level), the trading rules cannot read it")
    validate_table(table)
    n = len(table)
    if n == 0:
        raise ValueError("to_battery_table: empty table")
    last_close = np.asarray(last_close, dtype=np.float64).reshape(-1)
    if last_close.shape[0] != n:
        raise ValueError(f"last_close has {last_close.shape[0]} rows, the table has {n}")
    if not np.isfinite(last_close).all() or (last_close <= 0).any():
        raise ValueError("last_close must be finite and > 0")
    if head_spec("nig_regression", "close").output_keys["mu"] not in table.columns:
        raise ValueError("to_battery_table: the table has no close NIG head (direction is judged on the close only)")
    mu, y_ret, sigma = _head(table, labels, anchors, last_close, codec, codec_state, "close")
    cls_key = head_spec("binary_classification", "close").output_keys["logits"]
    from_class = cls_key in table.columns
    p_dir = _col(table, cls_key) if from_class else _gaussian_p_up(mu, sigma)
    out = pd.DataFrame({"step": np.ones(n, dtype=np.int64), "mu": mu, "sigma": sigma, "p_dir": p_dir,
                        "y_ret": y_ret, "dir_true": np.sign(y_ret).astype(np.int64)})
    if prev_close is not None:
        prev = np.asarray(prev_close, dtype=np.float64).reshape(-1)
        if prev.shape[0] != n or not np.isfinite(prev).all() or (prev <= 0).any():
            raise ValueError("prev_close must be finite, > 0 and aligned to the rows")
        out["last_ret"] = last_close / prev - 1.0
    if atr is not None:
        a = np.asarray(atr, dtype=np.float64).reshape(-1)
        if a.shape[0] != n:
            raise ValueError(f"atr has {a.shape[0]} rows, the table has {n}")
        out["atr"] = a
    out["coin"] = table["asset"].astype(str).to_numpy()
    out["end_time"] = table["prediction_made_at"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    out["last_close"] = last_close
    for t in (c for c in TARGET_COLUMNS if c != "close"):
        if head_spec("nig_regression", t).output_keys["mu"] in table.columns:
            pred, real, width = _head(table, labels, anchors, last_close, codec, codec_state, t)
            out[f"pred_{t}"], out[f"fut_{t}"], out[f"sigma_{t}"] = pred, real, width
    side = head_spec("side_payoff", "close").output_keys     # LS1 auxiliary heads (owner 2026-11-19), when present
    if side["long"] in table.columns:
        out["pay_long"], out["pay_short"] = _col(table, side["long"]), _col(table, side["short"])
    upq = head_spec("upper_quantile", "close").output_keys["q"]
    if upq in table.columns:
        out["q_up"] = np.expm1(_col(table, upq))
    out.attrs["p_dir_source"] = "close class head" if from_class else "gaussian from close mu/sigma"
    return out
