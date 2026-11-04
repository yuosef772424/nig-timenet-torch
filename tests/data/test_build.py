"""build_dataset: orchestration rules (skips, sealed coins, reference coin, spill/workers equality, filters)."""

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.data.build import build_dataset, feature_order
from nig_timenet_torch.data.lock import is_sealed, load_lock
from nig_timenet_torch.data.sources import synthetic_market
from nig_timenet_torch.data.windows import sample_filter_mask
from tests.data.helpers import COINS, tiny_settings


def test_feature_order_is_the_production_set():
    assert feature_order(tiny_settings()) == [
        "close", "volume", "DMP_14", "DMN_14", "RSI_14", "NATR_14", "BBB_20_2.0", "MFI_14", "RANGE_rel", "POS_50",
        "VOLZ_20", "SUPERT_STRETCH_10", "VOL_TERM_3_30", "MKT_CORR_20", "MKT_CORR_60", "MKT_BETA_20",
        "MKT_BREADTH_24", "MOM_ORTH_NATR"]


def test_excluding_and_disabling_features_changes_the_columns():
    s = tiny_settings(**{"features.exclude_from_features": ("volume",), "features.breadth.enabled": False})
    cols = feature_order(s)
    assert "volume" not in cols and "MKT_BREADTH_24" not in cols
    ds = build_dataset(s)
    assert ds["X_1h"].shape[2] == len(cols) and ds["feature_order"] == cols


def test_a_too_short_coin_is_skipped_with_its_reason():
    market = synthetic_market(list(COINS), 40, 40)
    market["C3USDT"] = market["C3USDT"].iloc[:20]
    ds = build_dataset(tiny_settings(), loader=lambda n: market[n])
    assert [b["name"] for b in ds["asset_bounds"]] == list(COINS[:3])
    assert "too short" in ds["skipped_assets"]["C3USDT"]


def test_a_missing_file_is_a_skip_not_a_crash():
    market = synthetic_market(list(COINS), 40, 40)

    def loader(name):
        if name == "C2USDT":
            raise FileNotFoundError(name)
        return market[name]
    ds = build_dataset(tiny_settings(), loader=loader)
    assert "C2USDT" in ds["skipped_assets"] and len(ds["asset_bounds"]) == 3


def test_sealed_coins_are_never_loaded():
    lock = load_lock()
    sealed = lock.sealed[0]
    assert is_sealed(sealed, lock) and is_sealed(sealed[:-4].lower(), lock)
    asked = []
    market = synthetic_market(list(COINS), 40, 40)

    def loader(name):
        asked.append(name)
        return market[name]
    s = tiny_settings(**{"source.coins": (*COINS, sealed)})
    ds = build_dataset(s, loader=loader)
    assert sealed not in asked and ds["sealed_dropped"] == [sealed]


def test_a_missing_reference_coin_is_an_error_not_silent_zeros():
    s = tiny_settings(**{"features.market.reference_symbol": "BTCUSDT"})
    market = synthetic_market(list(COINS), 40, 40)

    def loader(name):
        if name == "BTCUSDT":
            raise FileNotFoundError(name)
        return market[name]
    with pytest.raises(ValueError, match="reference coin"):
        build_dataset(s, loader=loader)


def test_spill_and_workers_do_not_change_the_numbers(tmp_path):
    base = build_dataset(tiny_settings())
    for extra in ({"runtime.workers": 3}, {"runtime.spill_dir": str(tmp_path / "spill")}):
        other = build_dataset(tiny_settings(**extra))
        for k in ("X_1h", "y_close_reg", "last_candles", "base_params"):
            np.testing.assert_array_equal(base[k], other[k])
    assert not (tmp_path / "spill").exists()


def test_window_count_follows_stride_and_horizon():
    a = build_dataset(tiny_settings())
    b = build_dataset(tiny_settings(**{"windows.stride": 16}))
    c = build_dataset(tiny_settings(**{"targets.horizon": 3}))
    assert 1.8 < len(a["last_candles"]) / len(b["last_candles"]) < 2.2
    assert c["forecast_horizon"] == 3
    fut = c["last_candles"][:, 4]
    assert np.isfinite(fut).all() and (c["last_candles"][:, 6] >= c["last_candles"][:, 5]).all()


def test_no_window_crosses_a_gap():
    market = synthetic_market(list(COINS), 40, 40)
    market["C1USDT"] = market["C1USDT"].drop(market["C1USDT"].index[300:310])
    ds = build_dataset(tiny_settings(), loader=lambda n: market[n])
    c1 = next(b for b in ds["asset_bounds"] if b["name"] == "C1USDT")
    ts = pd.to_datetime(ds["last_candles"][c1["start"]:c1["end"], 3].astype("int64"), utc=True)
    gap_lo, gap_hi = market["C1USDT"].index[299], market["C1USDT"].index[300]
    first_input = ts - pd.Timedelta(hours=15)
    assert not ((first_input <= gap_lo) & (ts >= gap_hi)).any()


def test_sample_filters_by_feature_and_by_expression():
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0], "RSI_14": [10.0, 50.0, 90.0]})
    assert sample_filter_mask(df, [{"feature": "RSI_14", "op": ">", "value": 20}]).tolist() == [False, True, True]
    assert sample_filter_mask(df, [{"expr": "close < 3"}, {"feature": "RSI_14", "op": ">=", "value": 50}]
                              ).tolist() == [False, True, False]
    with pytest.raises(ValueError, match="unknown column or operator"):
        sample_filter_mask(df, [{"feature": "nope", "op": ">", "value": 1}])
    full = build_dataset(tiny_settings())
    some = build_dataset(tiny_settings(**{"windows.sample_filters": ({"feature": "RSI_14", "op": ">", "value": 50},)}))
    assert 0 < len(some["last_candles"]) < len(full["last_candles"])
