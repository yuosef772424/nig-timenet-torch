"""A small synthetic world with a KNOWN answer for the integrated verification tests (no model, no artifact file).

A dataset dict as the artifact contract has it (daily bars, ``last_candles``, regression and class labels built by the
registered return codec), the row indices of three time splits with gaps, and the PredictionTables of a "model":
``skilled`` (the truth plus noise), ``unskilled`` (independent of the truth) or ``majority`` (always the train majority
class). Nothing here is a test."""

import numpy as np
import pandas as pd

from nig_timenet_torch.core.prediction_schema import PREDICTION_TABLE_COLUMNS, SCHEMA_VERSION
from nig_timenet_torch.core.schema import LAST_COLUMNS
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS, anchor_and_future
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.stats.nig import nig_predictive_std

TARGETS = ("high", "low", "close")
SCALE = 100.0
WINDOW, HORIZON = 8, 1
SD = 0.01                                    # sd of the close return and of the NIG predictive std (return units)
T0 = int(pd.Timestamp("2025-01-01").value)


def _splits(n_days: int, n_coins: int, val_start: int, test_start: int) -> dict:
    day = np.arange(n_days * n_coins) // n_coins
    return {"train": np.flatnonzero(day < n_days // 2),
            "val": np.flatnonzero((day >= val_start) & (day < n_days * 3 // 4)),
            "test": np.flatnonzero(day >= test_start)}


def make_dataset(seed: int = 0, n_days: int = 400, n_coins: int = 5, bar: str = "1D"):
    """-> (dataset, truth) with truth = {"r": close return, "cpos": close position in the last candle}. ``n_days`` is
    the number of bars of the base timeframe ``bar`` ("1D", or "6h": four bars per calendar day)."""
    bar_ns = int(pd.Timedelta(bar).value)
    rng = np.random.default_rng(seed)
    n = n_days * n_coins
    last_close = 100.0 * np.exp(rng.normal(0, 0.1, n))
    width, cpos = rng.uniform(0.01, 0.02, n), rng.uniform(0.0, 1.0, n)
    last_low = last_close * (1 - cpos * width)
    last_high = last_low + width * last_close
    r = rng.normal(0, SD, n)
    fut_close = last_close * (1 + r)
    fut_high = last_close * (1 + np.maximum(r, 0) + np.abs(rng.normal(0, 0.002, n)))
    fut_low = last_close * (1 + np.minimum(r, 0) - np.abs(rng.normal(0, 0.002, n)))
    ts = (T0 + (np.arange(n) // n_coins) * bar_ns).astype(np.float64)
    lc = np.zeros((n, len(LAST_COLUMNS)))
    columns = (last_high, last_low, last_close, ts, fut_close, fut_low, fut_high)
    for name, col in zip(LAST_COLUMNS, columns, strict=True):
        lc[:, LAST_COLUMNS.index(name)] = col
    ds = {"last_candles": lc, "base_tf": bar, "timeframes": [bar], "window_sizes": {bar: WINDOW}, "bar_ns": bar_ns,
          "forecast_horizon": HORIZON, "reg_target_mode": "return", "reg_target_scale": SCALE,
          "price_targets": list(TARGETS), f"X_{bar}": rng.normal(size=(n, WINDOW, 3)).astype(np.float32)}
    state = TARGET_CODECS["return"]().fit(ds)
    for t in TARGETS:
        raw = anchor_and_future(lc, t, state)
        ds[head_spec("nig_regression", t).label_key] = TARGET_CODECS["return"]().encode(raw, state).astype(np.float32)
        ds[head_spec("binary_classification", t).label_key] = (raw[:, 1] > raw[:, 0]).astype(np.float32)
    ds["n_coins"] = n_coins
    return ds, {"r": r, "cpos": cpos}


def make_idx(ds: dict, *, val_start: int = 212, test_start: int = 312) -> dict:
    n_days = len(ds["last_candles"]) // ds["n_coins"]
    return _splits(n_days, ds["n_coins"], val_start, test_start)


def make_table(ds: dict, rows, *, kind: str = "skilled", seed: int = 1, majority: dict | None = None) -> pd.DataFrame:
    """The PredictionTable of ``rows`` for a model of ``kind`` ("skilled" | "unskilled" | "majority"); ``majority``:
    {target: True when the train majority class is up} for the "majority" model, which always says that class."""
    rng = np.random.default_rng(seed)
    rows = np.asarray(rows)
    n = len(rows)
    ts = pd.to_datetime(np.asarray(ds["last_candles"])[rows, LAST_COLUMNS.index("timestamp")].astype(np.int64),
                        utc=True).as_unit("ns")
    cols = {"timestamp": ts, "asset": pd.array([f"C{i % ds['n_coins']}" for i in rows], dtype="str"),
            "horizon_ns": np.full(n, ds["bar_ns"], dtype=np.int64),
            "prediction_made_at": ts + pd.Timedelta(ds["bar_ns"]),
            "label_available_at": ts + pd.Timedelta(2 * ds["bar_ns"]),
            "identity_hash": pd.array(np.full(n, "h"), dtype="str"),
            "stage": pd.array(np.full(n, "val"), dtype="str"), "fold": np.zeros(n, dtype=np.int64),
            "train_end": pd.DatetimeIndex(np.full(n, 0)).tz_localize("UTC").as_unit("ns"),
            "schema_version": np.full(n, SCHEMA_VERSION, dtype=np.int64)}
    assert set(PREDICTION_TABLE_COLUMNS) <= set(cols)
    nu, alpha = 1.0, 5.0
    beta = (SD * SCALE) ** 2 * nu * (alpha - 1) / (1 + nu)               # predictive std = SD in label units
    assert abs(float(nig_predictive_std(nu, alpha, beta)) - SD * SCALE) < 1e-6
    for t in TARGETS:
        reg, cls = head_spec("nig_regression", t), head_spec("binary_classification", t)
        truth = np.asarray(ds[reg.label_key])[rows].astype(np.float64)
        if kind == "skilled":
            mu = truth + rng.normal(0, 0.5 * SD * SCALE, n)
        elif kind == "unskilled":
            mu = rng.normal(0, SD * SCALE, n) + np.median(truth)
        elif kind == "majority":
            mu = np.full(n, np.median(truth))
        else:
            raise ValueError(kind)
        p = (np.full(n, 0.9 if majority[t] else 0.1) if kind == "majority"
             else 1.0 / (1.0 + np.exp(-mu / (0.5 * SD * SCALE))))
        cols[reg.output_keys["mu"]] = mu
        cols[reg.output_keys["nu"]] = np.full(n, nu)
        cols[reg.output_keys["alpha"]] = np.full(n, alpha)
        cols[reg.output_keys["beta"]] = np.full(n, beta)
        cols[reg.output_keys["confidence"]] = np.clip(np.abs(mu - np.median(truth)) / (3 * SD * SCALE), 0, 1)
        cols[cls.output_keys["logits"]] = p
    return pd.DataFrame(cols)


def make_world(*, kind: str = "skilled", seed: int = 0, val_start: int = 212, test_start: int = 312,
               bar: str = "1D"):
    """-> (dataset, idx, tables) with tables = {"val": ..., "test": ...}."""
    ds, _ = make_dataset(seed, bar=bar)
    idx = make_idx(ds, val_start=val_start, test_start=test_start)
    maj = {t: bool(np.mean(ds[head_spec("binary_classification", t).label_key][idx["train"]] > 0) >= 0.5)
           for t in TARGETS}
    tables = {s: make_table(ds, idx[s], kind=kind, seed=seed + 1 + i, majority=maj)
              for i, s in enumerate(("val", "test"))}
    return ds, idx, tables
