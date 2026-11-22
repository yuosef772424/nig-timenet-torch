"""HB5: the shared helpers the data tools use (channel maths, candle cache, label time, coin names, label keys, the
npy_dir writer, the archive URLs) behave as the copies they replaced."""

import numpy as np
import pandas as pd

from nig_timenet_torch.core import schema
from nig_timenet_torch.core.coins import norm_coin
from nig_timenet_torch.core.label_time import label_bar_close, label_bar_open
from nig_timenet_torch.data import align, channel_math, derive, fetch, indicator_channels, sources
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.data_io.lock import is_sealed, load_lock
from nig_timenet_torch.model.heads import head_spec


def test_label_keys_and_head_names_are_the_head_registry_strings():
    for t in schema.TARGET_COLUMNS:
        assert schema.reg_label_key(t) == head_spec("nig_regression", t).label_key
        assert schema.class_label_key(t) == head_spec("binary_classification", t).label_key
    assert schema.head_names(["high"]) == ["high_class", "high_reg"] == [
        head_spec("binary_classification", "high").name, head_spec("nig_regression", "high").name]


def test_ratio_indicators_equal_the_bar_indicators_on_closes_only():
    rng = np.random.default_rng(0)
    c = pd.Series(np.exp(np.cumsum(0.02 * rng.standard_normal(200))))
    g = pd.DataFrame({"high": c, "low": c, "close": c, "quote_volume": rng.uniform(1, 9, 200)})
    ch, _, _ = indicator_channels.bar_indicators(g)
    k, d = channel_math.stoch_slow(c)
    np.testing.assert_array_equal(ch[:, 0], channel_math.rsi_scaled(c))
    np.testing.assert_array_equal(ch[:, 1], k)
    np.testing.assert_array_equal(ch[:, 2], d)
    np.testing.assert_array_equal(ch[:, 3], channel_math.volume_rel(g["quote_volume"]).to_numpy())


def test_to_grid_puts_missing_bars_as_nan_and_drops_off_step_bars():
    idx = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 08:00", "2024-01-01 09:00"], tz="UTC")
    c = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    first, g = channel_math.to_grid(c, pd.Timedelta("4h").value)
    assert first == idx[0].value and g["close"].tolist()[0] == 1.0 and np.isnan(g["close"][1]) and g["close"][2] == 2.0
    assert len(g) == 3
    first2, g2 = channel_math.to_grid(c, pd.Timedelta("4h").value,
                                      first=first, size=5)
    assert len(g2) == 5 and np.isnan(g2["close"][4])


def test_candle_cache_refetches_a_narrow_or_stale_file(tmp_path):
    idx = pd.date_range("2024-01-01", periods=10, freq="4h", tz="UTC")
    calls = []

    def fetch_(symbol, tf, start, end):
        calls.append((start, end))
        return pd.DataFrame({"high": 2.0, "low": 1.0, "close": 1.5, "quote_volume": 5.0}, index=idx)

    t0, t1 = idx[0], idx[-1]
    narrow = ("high", "low", "close")
    derive.coin_candles("X", "4h", t0, t1, tmp_path, fetch_, narrow)
    assert len(calls) == 1
    derive.coin_candles("X", "4h", t0, t1, tmp_path, fetch_, narrow)            # covered: cache
    assert len(calls) == 1
    out = derive.coin_candles("X", "4h", t0, t1, tmp_path, fetch_)             # wants quote_volume: refetch
    assert len(calls) == 2 and "quote_volume" in out.columns
    derive.coin_candles("X", "4h", t0, t1, tmp_path, fetch_)                    # now wide enough: cache
    assert len(calls) == 2
    derive.coin_candles("X", "4h", t0 - pd.Timedelta("1D"), t1, tmp_path, fetch_)   # wider range: refetch
    assert len(calls) == 3


def test_label_time_open_and_close_agree_on_the_seal_for_bar_aligned_data():
    bar, seal = 4 * 3600 * 10**9, 100 * 4 * 3600 * 10**9
    ts = np.arange(90, 110) * bar
    assert (label_bar_close(ts, 2, bar) - label_bar_open(ts, 2, bar) == bar).all()
    assert ((label_bar_open(ts, 2, bar) < seal) == (label_bar_close(ts, 2, bar) <= seal)).all()


def test_norm_coin_and_the_reference_coin_without_usdt():
    assert norm_coin(" btcusdt ") == norm_coin("BTC") == "BTC" and norm_coin("USDT") == "USDT"
    sealed = load_lock().sealed[0]
    assert is_sealed(sealed, load_lock()) and is_sealed(norm_coin(sealed), load_lock())


def test_build_compares_the_reference_coin_with_the_shared_normaliser():
    from nig_timenet_torch.data import build

    assert build.norm_coin is norm_coin     # D-23: "BTC" and "BTCUSDT" are one coin to the builder too


def test_one_ohlcv_agg_and_the_open_time_column():
    assert align.OHLCV_AGG is schema.OHLCV_AGG is sources.OHLCV_AGG
    assert schema.OPEN_TIME_COL == "open_time" and schema.QUOTE_VOLUME_COL in fetch.CSV_HEADER


def test_archive_urls_are_the_ones_both_downloaders_used():
    m = pd.Period("2024-01", "M")
    assert fetch.monthly_url("BTCUSDT", "4h", m) == (
        "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/4h/BTCUSDT-4h-2024-01.zip")
    assert fetch.daily_url("BTCUSDT", "4h", m, 5) == (
        "https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/4h/BTCUSDT-4h-2024-01-05.zip")


def test_npy_dir_save_round_trips_and_writes_the_manifest_last(tmp_path):
    ds = {"a": np.arange(6).reshape(3, 2), "meta": {"k": 1}, "b": np.ones(3, dtype="float32")}
    seen = {}
    man = npy_dir.save(ds, tmp_path, export=lambda files: seen.update(files) or {"files": files})
    got = npy_dir.load(tmp_path)
    assert list(got) == ["a", "meta", "b"] and got["meta"] == {"k": 1}
    np.testing.assert_array_equal(got["a"], ds["a"])
    assert man["export"]["files"] == seen and set(seen) == {"a.npy", "b.npy", "meta.pkl"}
    assert seen["a.npy"] == npy_dir.sha256_file(tmp_path / "a.npy")
