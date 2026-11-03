"""Synthetic PredictionTable + labels with a KNOWN answer, for the T4 tests (no model involved)."""

import numpy as np
import pandas as pd

from nig_timenet_torch.core.prediction_schema import SCHEMA_VERSION
from nig_timenet_torch.core.schema import TARGET_COLUMNS
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS
from nig_timenet_torch.model.heads import head_spec

N_COINS, PER_COIN = 6, 1000
ANCHOR = 100.0


def codec_state() -> dict:
    return TARGET_CODECS["return"]().fit({})


def make(*, scale: float = 0.005, conf_informative: bool = True, skilled: bool = True, seed: int = 0,
         nu: float = 1.0, alpha: float = 5.0, signal_sd: float = 0.005, center: dict | None = None):
    """-> (table, labels, anchors).

    mu = centre + signal_sd * z (a different z per row). Skilled: truth = mu + noise, the noise being Student-t(2
    alpha) at NIG scale ``scale`` (so the 80 % interval from nu/alpha/beta must cover 80 %); unskilled: truth =
    centre + an independent signal + the same noise (mu knows nothing). With ``conf_informative`` the noise size is
    multiplied by (1.5 - u), u ~ U(0, 1) hidden, and confidence = u (bigger u = smaller error); otherwise the
    noise is homoscedastic and confidence is a constant. The class label is truth > centre, its score a sigmoid of
    the same mu offset."""
    rng = np.random.default_rng(seed)
    center = center or {"high": 0.02, "low": -0.02, "close": 0.0}
    n = N_COINS * PER_COIN
    asset = np.repeat([f"C{i}" for i in range(N_COINS)], PER_COIN)
    ts = np.tile(np.arange(PER_COIN), N_COINS).astype("int64") * 3_600_000_000_000
    u = rng.random(n)
    t_idx = pd.to_datetime(ts, utc=True).as_unit("ns")
    cols = {"timestamp": t_idx, "asset": pd.array(asset, dtype="str"),
            "horizon_ns": np.full(n, 3_600_000_000_000, dtype=np.int64), "prediction_made_at": t_idx,
            "label_available_at": t_idx, "identity_hash": pd.array(np.full(n, "h"), dtype="str"),
            "stage": pd.array(np.full(n, "val"), dtype="str"), "fold": np.zeros(n, dtype=np.int64),
            "train_end": pd.DatetimeIndex(np.full(n, 0)).tz_localize("UTC").as_unit("ns"),
            "schema_version": np.full(n, SCHEMA_VERSION, dtype=np.int64)}
    labels, anchors = {}, {}
    beta = scale**2 * alpha * nu / (1.0 + nu)           # NIG scale^2 = beta (1 + nu) / (nu alpha)
    amp = (1.5 - u) / 1.0 if conf_informative else np.ones(n)
    for t in TARGET_COLUMNS:
        reg, cls = head_spec("nig_regression", t), head_spec("binary_classification", t)
        mu = center[t] + signal_sd * rng.standard_normal(n)
        noise = rng.standard_t(2 * alpha, n) * scale * amp
        y = mu + noise if skilled else center[t] + signal_sd * rng.standard_normal(n) + noise
        cols[reg.output_keys["mu"]] = mu
        cols[reg.output_keys["nu"]] = np.full(n, nu)
        cols[reg.output_keys["alpha"]] = np.full(n, alpha)
        cols[reg.output_keys["beta"]] = np.full(n, beta)
        cols[reg.output_keys["confidence"]] = u if conf_informative else np.full(n, 0.5)
        cols[cls.output_keys["logits"]] = 1.0 / (1.0 + np.exp(-(mu - center[t]) / signal_sd))
        labels[reg.label_key], labels[cls.label_key] = y, (y > center[t]).astype(float)
        anchors[t] = np.full(n, ANCHOR)
    return pd.DataFrame(cols), labels, anchors
