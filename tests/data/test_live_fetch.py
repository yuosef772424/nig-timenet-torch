"""data/fetch.py::recent_klines and data/sources.py::frames_loader: the live candles, closed bars only, no network."""

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.data.fetch import Http, recent_klines
from nig_timenet_torch.data.settings import SourceSettings
from nig_timenet_torch.data.sources import frames_loader

H = 3_600_000
T0 = int(pd.Timestamp("2026-11-20", tz="UTC").timestamp() * 1000)


def _kline(i: int, price: float) -> list:
    """A Binance kline: [openTime, o, h, l, c, v, closeTime, quoteVol, trades, takerBase, takerQuote, ignore]."""
    t = T0 + i * H
    return [t, str(price), str(price + 1), str(price - 1), str(price + 0.5), "10", t + H - 1, "1000", 7, "5", "500",
            "0"]


class FakeApi(Http):
    def __init__(self, n_bars: int, fail_first: int = 0, cap: int = 1500):
        self.rows, self.calls, self.fail, self.cap = [_kline(i, 100 + i) for i in range(n_bars)], [], fail_first, cap

    def get_json(self, url, params=None):
        self.calls.append(dict(params))
        if self.fail:
            self.fail -= 1
            raise ConnectionError("boom")
        lo, hi = params["startTime"], params.get("endTime", 2 ** 62)
        return [r for r in self.rows if lo <= r[0] <= hi][:min(params["limit"], self.cap)]


def test_only_closed_bars_the_last_n_in_time_order():
    api = FakeApi(60)
    now = pd.Timestamp(T0 + 50 * H + 20 * 60_000, unit="ms", tz="UTC")     # bar 50 opened 20 minutes ago: still open
    got = recent_klines("BTCUSDT", "1h", 24, api, now=now)
    assert len(got) == 24 and got.index[-1] == pd.Timestamp(T0 + 49 * H, unit="ms", tz="UTC")
    assert got.index.is_monotonic_increasing and got.index.tz is not None
    assert list(got.columns[:5]) == ["open", "high", "low", "close", "volume"] and got["close"].iloc[-1] == 149.5
    exact = recent_klines("BTCUSDT", "1h", 3, api, now=pd.Timestamp(T0 + 50 * H, unit="ms", tz="UTC"))
    assert exact.index[-1] == pd.Timestamp(T0 + 49 * H, unit="ms", tz="UTC")   # bar 49 closes exactly at now: closed


def test_pages_beyond_the_api_limit_and_retries_a_failed_request():
    api = FakeApi(3200, fail_first=2)
    got = recent_klines("ETHUSDT", "1h", 3100, api, now=pd.Timestamp(T0 + 3200 * H, unit="ms", tz="UTC"), pause=0.0)
    assert len(got) == 3100 and not got.index.duplicated().any() and sum("limit" in c for c in api.calls) >= 4
    with pytest.raises(ConnectionError):
        recent_klines("ETHUSDT", "1h", 5, FakeApi(10, fail_first=9), tries=2, pause=0.0)


def test_errors_are_loud():
    with pytest.raises(ValueError, match="bars"):
        recent_klines("X", "1h", 0, FakeApi(1))
    far = pd.Timestamp(T0 - 10 * H, unit="ms", tz="UTC")
    with pytest.raises(ValueError, match="no (closed )?candle"):
        recent_klines("X", "1h", 5, FakeApi(3), now=far)


def test_frames_loader_cleans_like_a_file_source_and_raises_the_skip_error():
    api = FakeApi(30)
    frame = recent_klines("BTCUSDT", "1h", 20, api, now=pd.Timestamp(T0 + 30 * H, unit="ms", tz="UTC"))
    load = frames_loader({"BTCUSDT": frame}, SourceSettings(kind="synthetic", interval="1h"))
    got = load("BTCUSDT")
    assert list(got.columns) == ["open", "high", "low", "close", "volume"] and got.dtypes.eq(np.float64).all()
    assert got.index.tz is not None and len(got) == 20
    with pytest.raises(FileNotFoundError, match="no candles in memory"):
        load("NOPEUSDT")


def test_a_klines_url_other_than_the_futures_api_is_asked():
    seen = []

    class Spy(FakeApi):
        def get_json(self, url, params=None):
            seen.append(url)
            return super().get_json(url, params)

    now = pd.Timestamp(T0 + 30 * H, unit="ms", tz="UTC")
    recent_klines("BTCUSDT", "1h", 3, Spy(30), now=now)
    recent_klines("BTCUSDT", "1h", 3, Spy(30), now=now, url="https://data-api.binance.vision/api/v3/klines")
    assert seen[0].endswith("/fapi/v1/klines") and seen[-1] == "https://data-api.binance.vision/api/v3/klines"
    assert len(set(seen)) == 2


def test_an_endpoint_that_caps_its_pages_below_the_limit_never_loses_the_newest_bars():
    """Found against the real spot mirror (pages of 1000, 1500 asked): the loop took a short page for the end."""
    api = FakeApi(3200, cap=1000)
    now = pd.Timestamp(T0 + 3200 * H, unit="ms", tz="UTC")
    got = recent_klines("BTCUSDT", "1h", 2500, api, now=now)
    assert len(got) == 2500 and got.index[-1] == pd.Timestamp(T0 + 3199 * H, unit="ms", tz="UTC")
    assert not got.index.duplicated().any() and got.index.is_monotonic_increasing
