"""Target modes: hand-computed values, class rules, round trips, retargeting a built dataset."""

import numpy as np
import pytest

from nig_timenet_torch.data.build import build_dataset
from nig_timenet_torch.data.targets import TARGET_RAW, decode_reg, mode_names, parse_mode, retarget, target_arrays
from tests.data.helpers import tiny_settings

# last_high, last_low, last_close, timestamp, future_close, future_low_min, future_high_max
LAST = np.array([[110.0, 90.0, 100.0, 0, 105.0, 95.0, 120.0],
                 [50.0, 40.0, 45.0, 0, 40.5, 38.0, 46.0]])
BASES = np.array([[100.0, 10.0], [45.0, 5.0]])


def test_all_modes_are_registered():
    assert set(mode_names()) == {"return", "return_close", "window_scale", "scaled", "volnorm", "magnitude",
                                 "entry_range"}


def test_return_mode_values_and_classes():
    y, _ = target_arrays(LAST, BASES, "return", ("high", "low", "close"), scale=100.0)
    np.testing.assert_allclose(y["high_reg"], [(120 - 110) / 110 * 100, (46 - 50) / 50 * 100], rtol=1e-6)
    np.testing.assert_allclose(y["close_reg"], [5.0, -10.0], rtol=1e-6)
    assert y["low_class"].tolist() == [1.0, 0.0] and y["close_class"].tolist() == [1.0, 0.0]


def test_clip_is_applied_before_the_scale():
    last = np.array([[10.0, 10.0, 10.0, 0, 40.0, 10.0, 40.0]])
    y, info = target_arrays(last, None, "return_close", ("close",), scale=100.0, clip=1.0)
    assert y["close_reg"][0] == pytest.approx(100.0) and info.clip == 1.0


def test_scaled_modes_use_the_window_scale_and_are_unit_free():
    y, info = target_arrays(LAST, BASES, "scaled", ("close",), scale=100.0)
    np.testing.assert_allclose(y["close_reg"], [(105 - 100) / 10, (40.5 - 45) / 5], rtol=1e-6)
    assert info.scales["close"] == 1.0
    legacy, info2 = target_arrays(LAST, BASES, "window_scale", ("close",), scale=100.0)
    np.testing.assert_allclose(legacy["close_reg"], y["close_reg"] * 100.0, rtol=1e-6)
    assert info2.scales["close"] == 100.0


def test_modes_that_need_bases_say_so():
    with pytest.raises(ValueError, match="window bases"):
        target_arrays(LAST, None, "scaled", ("close",))


def test_magnitude_class_uses_the_threshold_it_is_given():
    y, info = target_arrays(LAST, None, "magnitude", ("close",))
    assert info.thresholds["close"] == pytest.approx(np.median([0.05, 0.1]))
    y2, _ = target_arrays(LAST, None, "magnitude", ("close",), thresholds={"close": 0.07})
    assert y2["close_class"].tolist() == [0.0, 1.0] and (y["close_reg"] >= 0).all()


def test_entry_range_is_non_negative_with_own_direction_classes():
    y, _ = target_arrays(LAST, None, "entry_range", ("high", "low", "close"))
    assert (y["high_reg"] >= 0).all() and (y["low_reg"] >= 0).all()
    assert y["high_class"].tolist() == [1.0, 0.0]            # own direction: 120 > 110, 46 < 50
    pos, info = target_arrays(LAST, None, "entry_range", ("close",), close_reg="range_pos", scale=100.0)
    assert info.scales["close"] == 1.0 and 0.0 <= pos["close_reg"].min() <= pos["close_reg"].max() <= 1.0


def test_relative_subtracts_the_median_of_the_same_time():
    last = LAST.copy()
    last[:, 3] = 7                                           # both rows share one timestamp
    y, info = target_arrays(last, None, "return_close+relative", ("close",), scale=1.0)
    r = np.array([0.05, -0.1])
    np.testing.assert_allclose(y["close_reg"], r - np.median(r), rtol=1e-6)
    assert info.group_size.tolist() == [2, 2]


def test_mode_names_are_validated():
    assert parse_mode("relative") == ("return", True) and parse_mode("scaled+relative") == ("scaled", True)
    with pytest.raises(ValueError, match="unknown target mode"):
        parse_mode("retrun")
    with pytest.raises(ValueError, match="not supported"):
        parse_mode("entry_range+relative")


@pytest.mark.parametrize("mode", ["return", "return_close", "window_scale", "scaled", "volnorm"])
def test_decode_is_the_inverse_of_the_encoding(mode):
    y, info = target_arrays(LAST, BASES, mode, ("high", "low", "close"), scale=100.0, clip=1e9)
    fut = {"high": LAST[:, 6], "low": LAST[:, 5], "close": LAST[:, 4]}
    for t in ("high", "low", "close"):
        back = decode_reg(y[f"{t}_reg"], LAST, BASES, mode, t, scale=100.0)
        np.testing.assert_allclose(back, fut[t], rtol=2e-5)
    assert info.mode == mode


def test_modes_without_a_price_say_so():
    for mode in ("magnitude", "entry_range", "return+relative"):
        with pytest.raises(ValueError, match="cannot be turned back"):
            decode_reg(np.zeros(2), LAST, BASES, mode, "close")


def test_registry_meta_is_complete():
    for name in mode_names():
        assert TARGET_RAW.entry(name).meta["clip"] > 0


def test_retarget_keeps_x_and_windows_and_stamps_the_dataset():
    ds = build_dataset(tiny_settings())
    new, report = retarget(ds, "scaled")
    assert new["X_1h"] is ds["X_1h"] and new["last_candles"] is ds["last_candles"]
    assert new["target_mode"] == "scaled" and new["reg_target_scale"] == 1.0 and report["clip"] == 10.0
    assert not np.array_equal(new["y_close_reg"], ds["y_close_reg"])
    again, _ = retarget(new, "return")
    np.testing.assert_allclose(again["y_close_reg"], ds["y_close_reg"], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(again["y_close_class"], ds["y_close_class"])
    assert again["reg_target_scale"] == ds["reg_target_scale"]


def test_retarget_magnitude_fits_the_threshold_on_the_train_rows_only():
    ds = build_dataset(tiny_settings())
    train = np.arange(0, 100)
    _, rep_train = retarget(ds, "magnitude", train_idx=train)
    _, rep_all = retarget(ds, "magnitude")
    assert rep_train["thresholds"] != rep_all["thresholds"]


def test_build_refuses_modes_that_need_the_whole_dataset():
    with pytest.raises(ValueError, match="whole dataset"):
        tiny_settings(**{"targets.reg_target_mode": "magnitude"})
