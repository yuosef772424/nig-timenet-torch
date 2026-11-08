"""B-051 class guard: every run checks that its codec decodes the stored labels to the REAL future prices of
last_candles, so an artifact whose label definition differs from the codec (the TF logret_* case) is refused or
decoded right, never misread silently."""

import numpy as np
import pytest

from nig_timenet_torch.core.schema import LAST_COLUMNS
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.workflow import flat_run, run_common
from tests.contract_doc import DATASET_DIR

TARGETS = ("high", "low", "close")
FUT = {"high": "future_high_max", "low": "future_low_min", "close": "future_close"}


def logret_dataset(n: int = 400, seed: int = 0) -> dict:
    """A TF-style logret_std dataset: labels = log(future / last_close) for every head."""
    rng = np.random.default_rng(seed)
    c = np.exp(rng.normal(0, 1, n))
    lc = np.zeros((n, len(LAST_COLUMNS)))
    col = {k: i for i, k in enumerate(LAST_COLUMNS)}
    lc[:, col["last_close"]] = c
    lc[:, col["last_high"]] = c * np.exp(np.abs(rng.normal(0, 0.02, n)))
    lc[:, col["last_low"]] = c * np.exp(-np.abs(rng.normal(0, 0.02, n)))
    fc = c * np.exp(rng.normal(0, 0.02, n))
    lc[:, col["future_close"]] = fc
    lc[:, col["future_high_max"]] = np.maximum(fc, c) * np.exp(np.abs(rng.normal(0, 0.01, n)))
    lc[:, col["future_low_min"]] = np.minimum(fc, c) * np.exp(-np.abs(rng.normal(0, 0.01, n)))
    ds = {"last_candles": lc, "reg_target_mode": "return", "reg_target_scale": 1.0, "price_norm_mode": "logret_std"}
    for t in TARGETS:
        ds[f"y_{t}_reg"] = np.log(lc[:, col[FUT[t]]] / c).astype(np.float32)
    return ds


def test_the_fixture_artifact_decodes_to_its_future_prices():
    run_common.check_decode(npy_dir.load(DATASET_DIR), TARGETS)


def test_a_logret_artifact_decodes_with_the_log_close_codec():
    ds = logret_dataset()
    run_common.check_decode(ds, TARGETS)
    state = run_common.codec_state(ds)
    assert state["mode"] == "logret_close"
    a = flat_run.anchors_by_name(ds["last_candles"], np.arange(3), TARGETS, state)
    assert all(np.array_equal(a[t], ds["last_candles"][:3, LAST_COLUMNS.index("last_close")]) for t in TARGETS)


def test_a_label_definition_the_codec_does_not_know_is_refused():
    ds = logret_dataset()
    del ds["price_norm_mode"]  # what the loader saw before B-051: the own-anchor return codec on log labels
    with pytest.raises(ValueError, match="does not decode the 'high' labels"):
        run_common.check_decode(ds, TARGETS)
