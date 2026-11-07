"""Raw sources: the reference project's file layout is read and written, downloads run against a fake network."""

import csv
import gzip
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.data.fetch import (
    CSV_HEADER,
    FAPI,
    VISION,
    Http,
    fetch_all,
    kline_row,
    raw_path,
    read_registry,
    update_registry,
)
from nig_timenet_torch.data.settings import SourceSettings
from nig_timenet_torch.data.sources import make_loader, resolve_coins, synthetic_market

HOUR = 3_600_000
T0 = int(pd.Timestamp("2025-03-01", tz="UTC").timestamp() * 1000)


def klines(n, start=T0, step=HOUR):
    return [[start + i * step, f"{100 + i}", f"{101 + i}", f"{99 + i}", f"{100.5 + i}", "10",
             start + (i + 1) * step - 1, "1005", 7, "5", "502", "0"] for i in range(n)]


class FakeHttp(Http):
    """/fapi pages of 1500 rows from a fixed list, and archive zips for the months/days it knows."""

    def __init__(self, rows, zips=None):
        self.rows, self.zips, self.calls = rows, zips or {}, []

    def get_json(self, url, params=None):
        self.calls.append((url, dict(params or {})))
        lo, hi = params["startTime"], params.get("endTime", 2 ** 62)
        return [r for r in self.rows if lo <= r[0] <= hi][:params["limit"]]

    def get_bytes(self, url):
        self.calls.append((url, None))
        return self.zips.get(url)


def zip_of(rows, header=True):
    buf = io.StringIO()
    w = csv.writer(buf)
    if header:
        w.writerow(["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count",
                    "taker_buy_volume", "taker_buy_quote_volume", "ignore"])
    w.writerows(rows)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("data.csv", buf.getvalue())
    return out.getvalue()


def settings(tmp_path, **kw):
    return SourceSettings(root=str(tmp_path), coins=("AAAUSDT",), **kw)


def test_files_layout_uses_the_reference_names(tmp_path):
    s = settings(tmp_path, fetch="futures")
    assert raw_path(s, "AAAUSDT") == tmp_path / "history_1h" / "AAAUSDT.csv"
    fetch_all(s, ["AAAUSDT"], FakeHttp(klines(30)))
    with open(tmp_path / "history_1h" / "AAAUSDT.csv", newline="") as f:
        assert next(csv.reader(f)) == CSV_HEADER
    reg = read_registry(s)
    assert reg[0]["name"] == "AAAUSDT" and reg[0]["file"] == "AAAUSDT.csv" and reg[0]["n_candles"] == "30"
    assert (tmp_path / "crypto_data" / "asset_registry.csv").exists()


def test_loader_reads_csv_gz_and_parquet_and_both_time_columns(tmp_path):
    s = settings(tmp_path, fetch="futures")
    fetch_all(s, ["AAAUSDT"], FakeHttp(klines(30)))
    plain = make_loader(SourceSettings(root=str(tmp_path), coins=("AAAUSDT",)))("AAAUSDT")
    assert plain.index.tz is not None and list(plain.columns) == ["open", "high", "low", "close", "volume"]
    assert len(plain) == 30 and plain.index.is_monotonic_increasing
    folder = tmp_path / "history_1h"
    with open(folder / "AAAUSDT.csv", "rb") as f, gzip.open(folder / "BBBUSDT.csv.gz", "wb") as g:
        g.write(f.read())
    gz = make_loader(SourceSettings(root=str(tmp_path), coins=("BBBUSDT",)))("BBBUSDT")      # .gz fallback
    pd.testing.assert_frame_equal(plain, gz)
    pd.DataFrame(klines(5), columns=range(12)).iloc[:, :6].set_axis(
        ["timestamp", "open", "high", "low", "close", "volume"], axis=1).to_csv(folder / "CCCUSDT.csv", index=False)
    only_ts = make_loader(SourceSettings(root=str(tmp_path), coins=("CCCUSDT",)))("CCCUSDT")   # ms timestamps only
    assert only_ts.index[0] == pd.Timestamp(T0, unit="ms", tz="UTC")
    plain.to_parquet(folder / "ohlc_DDDUSDT_1h.parquet")
    pq = make_loader(SourceSettings(root=str(tmp_path), coins=("DDDUSDT",), pattern="ohlc_{name}_{interval}.parquet"))
    assert len(pq("DDDUSDT")) == 30


def test_missing_file_and_folder_are_clear_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="candle folder"):
        make_loader(SourceSettings(root=str(tmp_path / "nowhere")))
    (tmp_path / "history_1h").mkdir()
    with pytest.raises(FileNotFoundError, match="NOPEUSDT"):
        make_loader(SourceSettings(root=str(tmp_path)))("NOPEUSDT")


def test_tail_start_end_bounds(tmp_path):
    s = settings(tmp_path, fetch="futures")
    fetch_all(s, ["AAAUSDT"], FakeHttp(klines(48)))
    cut = make_loader(SourceSettings(root=str(tmp_path), start="2025-03-01 10:00", end="2025-03-02 00:00"))("AAAUSDT")
    assert cut.index[0] == pd.Timestamp("2025-03-01 10:00", tz="UTC") and len(cut) == 14


def test_futures_download_pages_and_resumes(tmp_path):
    rows = klines(3200)
    s = settings(tmp_path, fetch="futures", start="2025-03-01")
    http = FakeHttp(rows[:2000])
    assert fetch_all(s, ["AAAUSDT"], http) == {"AAAUSDT": 2000}
    assert [c[0] for c in http.calls] == [f"{FAPI}/fapi/v1/klines"] * 2                 # two pages
    http2 = FakeHttp(rows)
    assert fetch_all(s, ["AAAUSDT"], http2) == {"AAAUSDT": 1200}                           # only the new tail
    assert http2.calls[0][1]["startTime"] == rows[1999][0] + 1
    assert len(make_loader(SourceSettings(root=str(tmp_path)))("AAAUSDT")) == 3200
    assert read_registry(s)[0]["n_gaps"] == "0"


def test_vision_download_monthly_then_daily(tmp_path):
    jan = [r for r in klines(24 * 31, start=int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000))]
    feb_day1 = klines(24, start=int(pd.Timestamp("2024-02-01", tz="UTC").timestamp() * 1000))
    base = f"{VISION}/data/futures/um"
    zips = {f"{base}/monthly/klines/AAAUSDT/1h/AAAUSDT-1h-2024-01.zip": zip_of(jan),
            f"{base}/daily/klines/AAAUSDT/1h/AAAUSDT-1h-2024-02-01.zip": zip_of(feb_day1, header=False)}
    s = settings(tmp_path, fetch="vision", start="2024-01-01", end="2024-02-02")
    assert fetch_all(s, ["AAAUSDT"], FakeHttp([], zips)) == {"AAAUSDT": 24 * 31 + 24}


def test_registry_rows_are_rebuilt_from_files_and_other_rows_kept(tmp_path):
    s = settings(tmp_path, fetch="futures")
    fetch_all(s, ["AAAUSDT"], FakeHttp(klines(10)))
    s2 = SourceSettings(root=str(tmp_path), coins=("BBBUSDT",), fetch="futures")
    fetch_all(s2, ["BBBUSDT"], FakeHttp(klines(12)))
    assert {r["name"]: r["n_candles"] for r in read_registry(s)} == {"AAAUSDT": "10", "BBBUSDT": "12"}
    assert update_registry(s, ["AAAUSDT"]) == 2


def test_all_global_means_the_whole_registry(tmp_path):
    s = settings(tmp_path, fetch="futures")
    fetch_all(s, ["AAAUSDT"], FakeHttp(klines(10)))
    assert resolve_coins(SourceSettings(root=str(tmp_path), categories=("all_global",))) == ["AAAUSDT"]


def test_resolve_coins_rules():
    assert resolve_coins(SourceSettings(coins=("A", "B"), exclude=("B",))) == ["A"]
    assert "BTCUSDT" in resolve_coins(SourceSettings(categories=("Large_Caps",)))
    with pytest.raises(ValueError, match="twice"):
        resolve_coins(SourceSettings(coins=("A", "A")))
    with pytest.raises(ValueError, match="unknown coin categories"):
        resolve_coins(SourceSettings(categories=("Nope",)))


def test_kline_row_and_synthetic_source_are_deterministic():
    assert kline_row(klines(1)[0])[:2] == [T0, "2025-03-01 00:00:00"]
    a, b = synthetic_market(["X", "Y"], 3, 5), synthetic_market(["X", "Y"], 3, 5)
    pd.testing.assert_frame_equal(a["Y"], b["Y"])
    assert not np.allclose(a["X"]["close"].to_numpy(), a["Y"]["close"].to_numpy())
    assert len(a["X"]) == 72


def test_normalize_tf_is_the_one_timeframe_normaliser():
    """'1d' -> '1D' (pandas 3 deprecates the lowercase day/week spellings); anything else unchanged."""
    from nig_timenet_torch.data.sources import normalize_tf
    assert normalize_tf("1d") == "1D" and normalize_tf("2w") == "2W"
    assert normalize_tf("4h") == "4h" and normalize_tf("1D") == "1D" and normalize_tf("15min") == "15min"
    with pytest.raises(TypeError, match="timeframe must be a string"):
        normalize_tf(4)


def test_resample_ohlcv_combines_bars_and_keeps_only_full_ones():
    """1h -> 4h the old project's way: open first / high max / low min / close last / volume sum, and
    only groups with the FULL count of base bars survive (a partial trailing group or a gap is dropped)."""
    from nig_timenet_torch.data.sources import resample_ohlcv
    hours = pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC")  # two full groups + a 2-bar partial
    df = pd.DataFrame(
        {"open": [1, 2, 2, 2, 4, 5, 5, 5, 7, 8], "high": [3, 5, 4, 4, 6, 8, 7, 7, 9, 9],
         "low": [0.5, 1, 1, 1, 3, 4, 4, 4, 6, 6], "close": [2, 2, 2, 4, 5, 5, 5, 7, 8, 8],
         "volume": [2, 3, 2, 3, 4, 5, 4, 5, 6, 6]}, index=hours)
    out = resample_ohlcv(df, "4h")
    assert len(out) == 2 and list(out.index) == [hours[0], hours[4]]  # the 2-bar partial group is gone
    first = out.iloc[0]
    assert first["open"] == 1 and first["high"] == 5 and first["low"] == 0.5
    assert first["close"] == 4 and first["volume"] == 10
    assert resample_ohlcv(df, "1h") is df  # the base timeframe is a no-op, same object
    with pytest.raises(ValueError, match="cannot be split"):
        resample_ohlcv(df, "30min")  # a target finer than the base can never be built
