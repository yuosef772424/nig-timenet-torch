"""Shared helpers of the E1 / E2 / E3 environment tests (HF-32: helpers live here, not in a test module)."""

from dataclasses import replace

import numpy as np
import pandas as pd

from nig_timenet_torch.core.prediction_schema import SCHEMA_VERSION
from nig_timenet_torch.core.schema import LAST_COLUMNS, TS_COL
from nig_timenet_torch.data_io.synthetic import SyntheticSettings, build_synthetic, oracle_keys
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS, anchor_and_future, codec_name
from nig_timenet_torch.model.heads import head_spec

#: small E1 for the fast suite: 32 coins (the protocol minimum) x 2000 bars = 32,000 rows
SMALL = SyntheticSettings(n_bars=2000)
#: a NIG with this alpha is a Student-t with 2e4 degrees of freedom: a normal with the oracle's sd
ORACLE_ALPHA = 1e4


def small(**changes) -> dict:
    return build_synthetic(replace(SMALL, **changes))


def assets(ds: dict) -> np.ndarray:
    return np.repeat([b["name"] for b in ds["asset_bounds"]], [b["end"] - b["start"] for b in ds["asset_bounds"]])


def oracle_table(ds: dict) -> tuple[pd.DataFrame, dict, dict, np.ndarray, dict]:
    """The oracle as a prediction table for eval.target_report: mu = oracle mean, NIG std = oracle sd (nu = 1, alpha
    large), confidence = -oracle sd. Returns (table, labels, anchors, last_close, codec_state)."""
    lc = np.asarray(ds["last_candles"])
    n = len(lc)
    t_idx = pd.to_datetime(lc[:, TS_COL].astype(np.int64), utc=True).as_unit("ns")
    cols = {"timestamp": t_idx, "asset": pd.array(assets(ds), dtype="str"),
            "horizon_ns": np.full(n, int(pd.Timedelta(ds["base_tf"]).value) * ds["forecast_horizon"], np.int64),
            "prediction_made_at": t_idx, "label_available_at": t_idx,
            "identity_hash": pd.array(np.full(n, "oracle"), dtype="str"),
            "stage": pd.array(np.full(n, "val"), dtype="str"), "fold": np.zeros(n, dtype=np.int64),
            "train_end": pd.DatetimeIndex(np.full(n, 0)).tz_localize("UTC").as_unit("ns"),
            "schema_version": np.full(n, SCHEMA_VERSION, dtype=np.int64)}
    state = TARGET_CODECS[codec_name(ds)]().fit(ds)
    labels, anchors = {}, {}
    for t in ds["price_targets"]:
        reg, cls = head_spec("nig_regression", t), head_spec("binary_classification", t)
        mean, sd = (np.asarray(ds[k]) for k in oracle_keys(t))
        cols[reg.output_keys["mu"]] = mean
        cols[reg.output_keys["nu"]] = np.ones(n)
        cols[reg.output_keys["alpha"]] = np.full(n, ORACLE_ALPHA)
        cols[reg.output_keys["beta"]] = sd ** 2 * (ORACLE_ALPHA - 1.0) / 2.0
        cols[reg.output_keys["confidence"]] = -sd
        labels[reg.label_key] = np.asarray(ds[reg.label_key])
        labels[cls.label_key] = np.asarray(ds[cls.label_key])
        anchors[t] = anchor_and_future(lc, t, state)[:, 0]
    return pd.DataFrame(cols), labels, anchors, lc[:, LAST_COLUMNS.index("last_close")], state
