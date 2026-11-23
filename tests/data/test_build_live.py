"""data/build.py::build_live and data/windows.py::prepare_asset(last_n=): the live windows are the dataset windows."""

import numpy as np
import pytest

from nig_timenet_torch.data.build import build_dataset, build_live
from nig_timenet_torch.data.sources import synthetic_market
from nig_timenet_torch.data_io.lock import load_lock
from tests.data.helpers import COINS, tiny_settings

pytestmark = pytest.mark.slow     # rule E3: 8 s of dataset builds, like tests/data/test_build.py

LAST = {"last_close": 2, "ts": 3, "future_close": 4, "future_low": 5, "future_high": 6}


def _rows(ds: dict) -> dict:
    """{(coin, bar open ns): row} of a dataset-shaped dict."""
    out = {}
    for b in ds["asset_bounds"]:
        for i in range(b["start"], b["end"]):
            out[(b["name"], int(ds["last_candles"][i, LAST["ts"]]))] = i
    return out


@pytest.mark.parametrize("two_tf", [False, True])
def test_live_windows_equal_the_dataset_windows_at_the_same_bars(two_tf):
    s = tiny_settings(two_tf=two_tf)
    market = synthetic_market(list(COINS), 40, 40)
    ds = build_dataset(s, loader=lambda n: market[n])
    lv = build_live(s, lambda n: market[n], list(COINS), last_n=300)
    a, b = _rows(ds), _rows(lv)
    common = sorted(set(a) & set(b))
    assert len(common) > 100                                          # many bars are on both grids
    for tf in s.windows.tf_order:
        x_ds, x_lv = ds[f"X_{tf}"], lv[f"X_{tf}"]
        assert all(np.array_equal(x_ds[a[k]], x_lv[b[k]]) for k in common)
    for key in ("last_candles", "base_params", "y_close_reg", "y_close_class"):
        k = key if key in ds else next(c for c in ds if c.endswith(key.split("_", 1)[1]))
        assert all(np.array_equal(ds[k][a[c]], lv[k][b[c]], equal_nan=True) for c in common[:200]), key


def test_the_newest_bar_is_live_and_has_no_future_or_label():
    s = tiny_settings()
    market = synthetic_market(list(COINS), 40, 40)
    lv = build_live(s, lambda n: market[n], list(COINS), last_n=5)
    assert [b["end"] - b["start"] for b in lv["asset_bounds"]] == [5] * 4
    for b in lv["asset_bounds"]:
        newest = lv["last_candles"][b["end"] - 1]
        assert int(newest[LAST["ts"]]) == int(market[b["name"]].index[-1].value)       # the very last bar of the coin
        assert np.isnan(newest[4:7]).all()                                              # no future yet
        assert np.isfinite(lv["last_candles"][b["start"]:b["end"] - 1, 4:7]).all()      # horizon 1: all but the newest
    y = [k for k in lv if k.startswith("y_") and k.endswith("_reg")][0]
    assert np.isnan(lv[y][[b["end"] - 1 for b in lv["asset_bounds"]]]).all()


def test_a_short_history_gives_the_same_window():
    s = tiny_settings()
    market = synthetic_market(list(COINS), 40, 40)
    full = build_live(s, lambda n: market[n], list(COINS), last_n=3)
    short = build_live(s, lambda n: market[n].iloc[-200:], list(COINS), last_n=3)
    assert np.allclose(full["X_1h"], short["X_1h"], atol=1e-3)


def test_a_gap_in_the_newest_window_skips_the_coin_instead_of_predicting_on_bad_data():
    s = tiny_settings()
    market = synthetic_market(list(COINS), 40, 40)
    holed = market["C2USDT"].drop(market["C2USDT"].index[-3])       # a missing bar inside the newest window
    lv = build_live(s, lambda n: holed if n == "C2USDT" else market[n], list(COINS), last_n=4)
    assert "gap: 1 raw bar(s) missing" in lv["skipped_assets"]["C2USDT"]
    assert [b["name"] for b in lv["asset_bounds"]] == ["C0USDT", "C1USDT", "C3USDT"]


def test_sealed_coins_are_dropped_and_an_empty_result_raises():
    sealed = load_lock().sealed[0]
    market = synthetic_market(list(COINS), 40, 40)
    lv = build_live(tiny_settings(), lambda n: market[n], [*COINS, sealed], last_n=2)
    assert lv["sealed_dropped"] == [sealed] and len(lv["asset_bounds"]) == 4
    with pytest.raises(ValueError, match="every coin is sealed"):
        build_live(tiny_settings(), lambda n: market[n], [sealed], last_n=2)
    with pytest.raises(ValueError, match="no coin produced a live window"):
        build_live(tiny_settings(), lambda n: market[n].iloc[:10], list(COINS), last_n=2)
    with pytest.raises(ValueError, match="last_n"):
        build_live(tiny_settings(), lambda n: market[n], list(COINS), last_n=0)
