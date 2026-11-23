"""Target modes: hand-computed values, class rules, round trips, retargeting a built dataset."""

import numpy as np
import pytest

from nig_timenet_torch.data.build import build_dataset
from nig_timenet_torch.data.targets import (
    PIPELINE_TARGETS,
    TARGET_RAW,
    decode_affine,
    decode_reg,
    entry_range_to_prices,
    mode_names,
    parse_mode,
    retarget,
    retarget_splits,
    row_anchors,
    target_arrays,
)
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


@pytest.mark.slow  # rule E3: fast suite under 100 s on CI (owner 2026-11-19); still runs in CI
def test_retarget_magnitude_fits_the_threshold_on_the_train_rows_only():
    ds = build_dataset(tiny_settings())
    train = np.arange(0, 100)
    _, rep_train = retarget(ds, "magnitude", train_idx=train)
    _, rep_all = retarget(ds, "magnitude")
    assert rep_train["thresholds"] != rep_all["thresholds"]


def test_build_refuses_modes_that_need_the_whole_dataset():
    with pytest.raises(ValueError, match="whole dataset"):
        tiny_settings(**{"targets.reg_target_mode": "magnitude"})


# ----------------------------------------------------------------------------------------- the spec, by hand
def test_every_mode_follows_the_owner_spec_by_hand():
    """Hand-written formulas of the spec on LAST (row 0: entry 100, row 1: entry 45), scale 1, clip off."""
    kw = {"scale": 1.0, "clip": 1e9}
    hi, lo, cl = ("high", "low", "close")
    # return: each target against the last price of its OWN kind; class = return > 0
    y, _ = target_arrays(LAST, BASES, "return", (hi, lo, cl), **kw)
    np.testing.assert_allclose(y["high_reg"], [120 / 110 - 1, 46 / 50 - 1])
    np.testing.assert_allclose(y["low_reg"], [95 / 90 - 1, 38 / 40 - 1])
    # return_close: everything against the last close
    y, _ = target_arrays(LAST, BASES, "return_close", (hi, lo, cl), **kw)
    np.testing.assert_allclose(y["high_reg"], [120 / 100 - 1, 46 / 45 - 1])
    np.testing.assert_allclose(y["low_reg"], [95 / 100 - 1, 38 / 45 - 1])
    assert y["low_class"].tolist() == [0.0, 0.0] and y["high_class"].tolist() == [1.0, 1.0]
    # scaled: (future - own last) / IQR; class = future above the own last price
    y, _ = target_arrays(LAST, BASES, "scaled", (hi, lo, cl), **kw)
    np.testing.assert_allclose(y["high_reg"], [(120 - 110) / 10, (46 - 50) / 5])
    np.testing.assert_allclose(y["low_reg"], [(95 - 90) / 10, (38 - 40) / 5])
    assert y["high_class"].tolist() == [1.0, 0.0] and y["low_class"].tolist() == [1.0, 0.0]
    # magnitude: |close return|, max rise, max fall (from the entry); class = larger than the TRAIN median
    thr = {"high": 0.1, "low": 0.1, "close": 0.07}
    y, info = target_arrays(LAST, BASES, "magnitude", (hi, lo, cl), thresholds=thr, **kw)
    np.testing.assert_allclose(y["close_reg"], [0.05, 0.1])
    np.testing.assert_allclose(y["high_reg"], [0.2, 46 / 45 - 1])
    np.testing.assert_allclose(y["low_reg"], [0.05, 1 - 38 / 45])
    assert y["close_class"].tolist() == [0.0, 1.0] and y["high_class"].tolist() == [1.0, 0.0]
    # entry_range: high = max(high / P - 1, 0), low = max(1 - low / P, 0), close = |close / P - 1| or the range position
    y, _ = target_arrays(LAST, BASES, "entry_range", (hi, lo, cl), **kw)
    np.testing.assert_allclose(y["high_reg"], [0.2, 46 / 45 - 1])
    np.testing.assert_allclose(y["low_reg"], [0.05, 1 - 38 / 45])
    np.testing.assert_allclose(y["close_reg"], [0.05, 0.1])
    assert y["high_class"].tolist() == [1.0, 0.0] and y["close_class"].tolist() == [1.0, 0.0]   # own direction, 1/0
    pos, _ = target_arrays(LAST, BASES, "entry_range", (cl,), close_reg="range_pos", **kw)
    np.testing.assert_allclose(pos["close_reg"], [(105 - 95) / (120 - 95), (40.5 - 38) / (46 - 38)])


def test_a_gap_is_zero_not_negative_in_entry_range_and_magnitude():
    last = np.array([[10.0, 9.0, 10.0, 0, 10.5, 10.2, 11.0]])      # the horizon low (10.2) stayed above the entry
    for mode in ("entry_range", "magnitude"):
        y, _ = target_arrays(last, None, mode, ("low",), scale=1.0)
        assert y["low_reg"][0] == 0.0


def test_relative_class_is_beating_the_market_and_scaled_relative_is_in_volatility_units():
    last = np.tile(LAST[:1], (5, 1))
    last[:, 3] = 7                                                   # one timestamp, five coins
    last[:, 4] = [101.0, 102.0, 103.0, 104.0, 130.0]
    bases = np.tile(BASES[:1], (5, 1))
    y, info = target_arrays(last, bases, "scaled+relative", ("close",))
    r = (last[:, 4] - 100.0) / 10.0
    np.testing.assert_allclose(y["close_reg"], r - np.median(r), rtol=1e-6)
    assert y["close_class"].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]
    np.testing.assert_allclose(info.centers["close"], np.full(5, np.median(r)))
    rel, _ = target_arrays(last, bases, "relative", ("close",), scale=1.0)
    ret, _ = target_arrays(last, bases, "return+relative", ("close",), scale=1.0)
    np.testing.assert_array_equal(rel["close_reg"], ret["close_reg"])


# ----------------------------------------------------------------------------------------- decode for every mode
def _world(n_coins=6, n_times=40, seed=3):
    """A synthetic last_candles panel: n_coins coins x n_times timestamps (time-major), with gaps in high / low."""
    rng = np.random.default_rng(seed)
    n = n_coins * n_times
    ts = np.repeat(np.arange(n_times), n_coins) * 3_600_000_000_000 + 1_700_000_000_000_000_000
    close = 100 * np.exp(rng.normal(0, 0.02, n))
    fc = close * np.exp(rng.normal(0, 0.03, n))
    fh = np.maximum(fc, close) * (1 + np.abs(rng.normal(0, 0.01, n)))
    fl = np.minimum(fc, close) * (1 - np.abs(rng.normal(0, 0.01, n)))
    fl[3::7] = close[3::7] * 1.01                                     # horizon low above the entry (a gap)
    fh[3::7] = np.maximum(fh[3::7], fl[3::7] * 1.005)
    fh[5::11] = close[5::11] * 0.99                                   # horizon high below the entry (a gap)
    fl[5::11] = np.minimum(fl[5::11], fh[5::11] * 0.995)
    fc = np.clip(fc, fl, fh)
    last = np.stack([close * 1.004, close * 0.996, close, ts.astype("float64"), fc, fl, fh], axis=1)
    bases = np.stack([close, np.abs(rng.normal(2.0, 0.4, n)) + 0.5], axis=1).astype("float32")
    return last, bases


def _dataset(last, bases, scale=100.0):
    ds = {"last_candles": last, "base_params": bases, "price_targets": ("high", "low", "close"),
          "reg_target_scale": scale, "reg_target_mode": "return", "target_mode": "return"}
    pipe, _ = target_arrays(last, bases, "return", ds["price_targets"], scale=scale)
    ds.update({f"y_{k}": v for k, v in pipe.items()})
    return ds


FUTURE = {"high": 6, "low": 5, "close": 4}
ROUND_TRIP_MODES = ["return", "return_close", "window_scale", "scaled", "volnorm", "magnitude", "entry_range",
                    "return+relative", "return_close+relative", "scaled+relative", "volnorm+relative",
                    "magnitude+relative", "window_scale+relative", "relative"]


@pytest.mark.parametrize(("mode", "close_reg"), [(m, "abs_return") for m in ROUND_TRIP_MODES]
                         + [("entry_range", "range_pos")])
def test_decode_of_the_encoding_is_the_real_future_price_for_every_mode(mode, close_reg):
    """decode(encode(y)) == the future price: exact for every mode, given the true sign (magnitude / entry_range
    close head), the true centre ('+relative') or the decoded range (range_pos). Clip off, so nothing is lost."""
    last, bases = _world()
    y, info = target_arrays(last, bases, mode, ("high", "low", "close"), scale=100.0, clip=1e9, close_reg=close_reg)
    sign = np.where(last[:, 4] >= last[:, 2], 1.0, -1.0)
    dec = {}
    for t in ("high", "low", "close"):
        kw = {"sign": sign, "center": info.centers.get(t), "close_reg": close_reg}
        if close_reg == "range_pos" and t == "close":
            kw["range_prices"] = (last[:, 6], last[:, 5])        # the range the position is inside: the true one
        dec[t] = decode_reg(y[f"{t}_reg"], last, bases, mode, t, scale=info.scales[t], **kw)
        want = last[:, FUTURE[t]]
        if parse_mode(mode)[0] in ("magnitude", "entry_range") and t != "close":   # a gap is floored at the entry
            want = np.where(want >= last[:, 2], want, last[:, 2]) if t == "high" else np.where(
                want <= last[:, 2], want, last[:, 2])
        np.testing.assert_allclose(dec[t], want, rtol=3e-6, err_msg=f"{mode} {close_reg} {t}")
        a, slope = decode_affine(last, bases, mode, t, info.scales[t], **kw)       # the affine form: same numbers
        np.testing.assert_allclose(a + slope * y[f"{t}_reg"], dec[t], rtol=1e-9)
    assert info.mode == mode


def test_decode_raises_clearly_when_an_input_is_missing():
    last, bases = _world(2, 3)
    z = np.zeros(len(last))
    with pytest.raises(ValueError, match="no sign"):
        decode_reg(z, last, bases, "magnitude", "close", 100.0)
    with pytest.raises(ValueError, match="no sign"):
        decode_reg(z, last, bases, "entry_range", "close", 100.0)
    with pytest.raises(ValueError, match="centre"):
        decode_reg(z, last, bases, "return+relative", "close", 100.0)
    with pytest.raises(ValueError, match="range_pos"):
        decode_reg(z, last, bases, "entry_range", "close", 1.0, close_reg="range_pos")
    with pytest.raises(ValueError, match="close_reg"):
        decode_reg(z, last, bases, "entry_range", "close", 1.0, close_reg="bogus")
    # high / low need nothing extra
    np.testing.assert_allclose(decode_reg(z + 5.0, last, bases, "magnitude", "high", 100.0), last[:, 2] * 1.05)
    np.testing.assert_allclose(decode_reg(z + 5.0, last, bases, "entry_range", "low", 100.0), last[:, 2] * 0.95)


def test_the_sign_picks_the_direction_of_the_magnitude_close():
    last, bases = _world(1, 2)
    up = decode_reg(np.full(2, 3.0), last, bases, "magnitude", "close", 100.0, sign=np.array([1, 1]))
    down = decode_reg(np.full(2, 3.0), last, bases, "magnitude", "close", 100.0, sign=np.array([-1, -1]))
    np.testing.assert_allclose(up, last[:, 2] * 1.03)
    np.testing.assert_allclose(down, last[:, 2] * 0.97)


def test_entry_range_to_prices_is_the_old_function():
    P = np.array([100.0, 50.0])
    out = entry_range_to_prices(P, high=[0.1, 0.2], low=[0.05, 0.1], close=[0.02, 0.04], p_close_up=[0.9, 0.2])
    np.testing.assert_allclose(out["high"], [110.0, 60.0])
    np.testing.assert_allclose(out["low"], [95.0, 45.0])
    np.testing.assert_allclose(out["close_up"], [102.0, 52.0])
    np.testing.assert_allclose(out["close_down"], [98.0, 48.0])
    np.testing.assert_allclose(out["close"], [102.0, 48.0])                 # p >= 0.5 -> up, else down
    assert "close" not in entry_range_to_prices(P, close=[0.02, 0.04])
    pos = entry_range_to_prices(P, high=[0.1, 0.2], low=[0.05, 0.1], close=[0.5, 1.5], close_reg="range_pos")
    np.testing.assert_allclose(pos["close"], [102.5, 60.0])                  # low + clip(pos, 0, 1) (high - low)
    assert "close" not in entry_range_to_prices(P, close=[0.5, 0.5], close_reg="range_pos")
    with pytest.raises(ValueError, match="close_reg"):
        entry_range_to_prices(P, close=[0.1, 0.1], close_reg="x")


# ----------------------------------------------------------------------------------------- retarget_splits
def _splits(n, cut=(0.6, 0.8)):
    a, b = int(n * cut[0]), int(n * cut[1])
    idx = np.arange(n)
    return idx[:a], idx[a:b], idx[b:]


def _y(ds):
    return {k: v for k, v in ds.items() if k.startswith("y_") or k.startswith("target_center_")}


STAMPS = ("target_mode", "reg_target_scales", "reg_target_clip", "return_target_scale", "reg_target_scale",
          "reg_target_mode", "entry_close_reg")


def test_retarget_splits_follows_target_arrays_on_each_split_and_stamps_the_dataset():
    last, bases = _world()
    ds = _dataset(last, bases)
    tr, va, te = _splits(len(last))
    new, rep = retarget_splits(ds, (tr, va, te), "scaled+relative")
    assert new["target_mode"] == "scaled+relative" and new["reg_target_mode"] == "scaled+relative"
    assert new["reg_target_scale"] == 1.0 and new["return_target_scale"] == 100.0 and new["reg_target_clip"] == 10.0
    assert new["reg_target_scales"] == {"high": 1.0, "low": 1.0, "close": 1.0} and "entry_close_reg" not in new
    for idx in (tr, va, te):                                   # every split on its own rows
        y, info = target_arrays(last[idx], bases[idx], "scaled+relative", ("high", "low", "close"))
        for k, v in y.items():
            np.testing.assert_array_equal(new[f"y_{k}"][idx], v)
        np.testing.assert_array_equal(new["target_center_close"][idx], info.centers["close"])
    assert set(rep["share_small_groups"]) == {"train", "val", "test"}
    assert new["last_candles"] is ds["last_candles"] and ds["target_mode"] == "return"   # the original is untouched


def test_magnitude_threshold_is_fitted_on_train_only_and_reused_on_val_and_test():
    last, bases = _world()
    ds = _dataset(last, bases)
    tr, va, te = _splits(len(last))
    new, rep = retarget_splits(ds, (tr, va, te), "magnitude")
    r = np.abs(last[:, FUTURE["close"]] / last[:, 2] - 1.0)
    thr = float(np.median(r[tr]))
    assert rep["thresholds"]["close"] == pytest.approx(thr) and thr != pytest.approx(float(np.median(r)))
    np.testing.assert_array_equal(new["y_close_class"][va], (r[va] > thr).astype("float32"))   # TRAIN median on val
    np.testing.assert_array_equal(new["y_close_class"][te], (r[te] > thr).astype("float32"))
    np.testing.assert_allclose(new["y_close_reg"][va], np.clip(r[va], -1, 1) * 100.0, rtol=1e-6)
    # a different future in val / test does not move the threshold
    last2 = last.copy()
    last2[va[0]:, 4] *= 1.5
    _, rep2 = retarget_splits(_dataset(last2, bases), (tr, va, te), "magnitude")
    assert rep2["thresholds"] == rep["thresholds"]


def test_relative_centre_is_the_median_of_the_same_split_at_the_same_time():
    last, bases = _world(4, 6)
    ds = _dataset(last, bases)
    n = len(last)
    tr, va = np.arange(0, n // 2 + 1), np.arange(n // 2 + 1, n)        # one timestamp has coins in both splits
    new, _ = retarget_splits(ds, (tr, va, np.array([], dtype=int)), "return_close+relative")
    r = last[:, 4] / last[:, 2] - 1.0
    ts = last[:, 3]
    for idx in (tr, va):
        for t in np.unique(ts[idx]):
            rows = idx[ts[idx] == t]
            np.testing.assert_allclose(new["target_center_close"][rows], np.median(r[rows]))
            np.testing.assert_allclose(new["y_close_reg"][rows], (r[rows] - np.median(r[rows])) * 100.0, rtol=1e-5)


def test_any_switching_order_gives_the_same_result_as_going_straight_to_the_last_mode():
    last, bases = _world()
    ds = _dataset(last, bases)
    parts = _splits(len(last))
    paths = (["scaled", "magnitude", "return_close+relative"], ["entry_range", "volnorm", "magnitude"],
             ["return", "volnorm", "entry_range", "scaled", "magnitude+relative"], [None, "scaled+relative"],
             ["scaled+relative", None, "return_close"])
    for path in paths:
        cur = ds
        for m in path:
            cur, _ = retarget_splits(cur, parts, m)
        direct, _ = retarget_splits(ds, parts, path[-1])
        assert set(_y(cur)) == set(_y(direct)), path
        for k in _y(direct):
            np.testing.assert_array_equal(_y(cur)[k], _y(direct)[k], err_msg=f"{path} {k}")
        for k in STAMPS:
            assert cur.get(k) == direct.get(k), (path, k)


def test_mode_none_returns_the_pipeline_targets_and_stamps():
    last, bases = _world()
    ds = _dataset(last, bases)
    ds["entry_close_reg"] = "abs_return"                               # a stamp of the pipeline that a mode drops
    parts = _splits(len(last))
    new, _ = retarget_splits(ds, parts, "relative")
    assert new["target_mode"] == "relative" and "target_center_close" in new
    back, rep = retarget_splits(new, parts, None)
    assert rep["mode"] is None
    for k in _y(ds):
        np.testing.assert_array_equal(back[k], ds[k])
    assert not any(k.startswith("target_center_") for k in back)
    for k in STAMPS:
        assert back.get(k) == ds.get(k) and (k in back) == (k in ds), k
    assert back[PIPELINE_TARGETS]["y"]["y_close_reg"] is ds["y_close_reg"]       # carried by reference, not copied
    again, _ = retarget_splits(back, parts, None)                                  # idempotent
    np.testing.assert_array_equal(again["y_close_reg"], ds["y_close_reg"])


def test_return_through_retarget_splits_equals_the_pipeline_targets():
    last, bases = _world()
    ds = _dataset(last, bases)
    new, _ = retarget_splits(ds, _splits(len(last)), "return")
    for k in _y(ds):
        np.testing.assert_allclose(new[k], ds[k], rtol=1e-6)


def test_retarget_splits_is_the_same_component_as_retarget():
    last, bases = _world()
    ds = _dataset(last, bases)
    parts = _splits(len(last))
    whole, _ = retarget(ds, "magnitude", train_idx=parts[0])
    split, _ = retarget_splits(ds, parts, "magnitude")
    for k in _y(whole):
        np.testing.assert_array_equal(whole[k], split[k])         # a per-row mode: whole dataset == per split
    with pytest.raises(ValueError, match="train, val, test"):
        retarget_splits(ds, parts[:2], "return")


def test_a_relative_retarget_then_a_plain_one_leaves_no_centre_behind():
    last, bases = _world()
    ds = _dataset(last, bases)
    parts = _splits(len(last))
    rel, _ = retarget_splits(ds, parts, "return_close+relative")
    assert {"target_center_high", "target_center_low", "target_center_close"} <= set(rel)
    plain, _ = retarget_splits(rel, parts, "return_close")
    assert not any(k.startswith("target_center_") for k in plain)
    with pytest.raises(ValueError, match="not supported"):
        retarget_splits(ds, parts, "entry_range+relative")


def test_entry_range_stamps_its_close_definition_and_scale():
    last, bases = _world()
    ds = _dataset(last, bases)
    new, _ = retarget_splits(ds, _splits(len(last)), "entry_range", close_reg="range_pos")
    assert new["entry_close_reg"] == "range_pos"
    assert new["reg_target_scales"] == {"high": 100.0, "low": 100.0, "close": 1.0}
    assert 0.0 <= new["y_close_reg"].min() and new["y_close_reg"].max() <= 1.0
    other, _ = retarget_splits(new, _splits(len(last)), "scaled")
    assert "entry_close_reg" not in other


# ----------------------------------------------------------------------------------------- the codec of the modes
SIGNED = ("return", "return_close", "window_scale", "scaled", "volnorm")


@pytest.mark.parametrize("mode", [m for m in ROUND_TRIP_MODES if m != "return"])
def test_the_target_mode_codec_decodes_the_retargeted_dataset_to_the_real_prices(mode):
    from nig_timenet_torch.data_io.target_codec import TARGET_CODECS, codec_name
    last, bases = _world()
    parts = _splits(len(last))
    new, _ = retarget_splits(_dataset(last, bases), parts, mode, clip=1e9)
    assert codec_name(new) == "target_raw"
    codec = TARGET_CODECS["target_raw"]()
    state = codec.fit(new)
    idx = parts[1]
    truth = row_anchors(new, idx, ("high", "low", "close"), truth=True)
    for t in ("high", "low", "close"):
        dec = codec.decode(new[f"y_{t}_reg"][idx].astype("float64"), state, truth[t])
        want = last[idx, FUTURE[t]]
        if parse_mode(mode)[0] in ("magnitude", "entry_range") and t != "close":   # a gap is floored at the entry
            want = np.maximum(want, last[idx, 2]) if t == "high" else np.minimum(want, last[idx, 2])
        np.testing.assert_allclose(dec, want, rtol=3e-6, err_msg=f"{mode} {t}")
    base, relative = parse_mode(mode)
    assert codec.forecasts_price(state) is (base in SIGNED and not relative)
    ev = row_anchors(new, idx, ("high", "low", "close"))       # evaluation anchors: no truth
    assert all(np.isfinite(ev[t]).all() for t in ev)


def test_row_anchors_for_evaluation_use_no_truth():
    last, bases = _world()
    ds = _dataset(last, bases)
    parts = _splits(len(last))
    rel, _ = retarget_splits(ds, parts, "return_close+relative")
    a = row_anchors(rel, parts[1], ("close",))["close"]
    np.testing.assert_allclose(a[:, 0], last[parts[1], 2])                       # centre 0: the entry itself
    mag, _ = retarget_splits(ds, parts, "magnitude")
    up = row_anchors(mag, parts[1], ("close",))["close"]
    np.testing.assert_allclose(up[:, 1], last[parts[1], 2] / 100.0)               # decoded UP: + entry / scale per unit
    rp, _ = retarget_splits(ds, parts, "entry_range", close_reg="range_pos")
    assert np.isnan(row_anchors(rp, parts[1], ("close",))["close"]).all()         # needs the predicted range
    assert np.isfinite(row_anchors(rp, parts[1], ("high", "low"))["high"]).all()


def test_a_return_retarget_keeps_the_own_anchor_return_codec():
    from nig_timenet_torch.data_io.target_codec import codec_name
    last, bases = _world()
    new, _ = retarget_splits(_dataset(last, bases), _splits(len(last)), "return")
    assert codec_name(new) == "return" and codec_name({**new, "reg_target_mode": "return_close"}) == "target_raw"
