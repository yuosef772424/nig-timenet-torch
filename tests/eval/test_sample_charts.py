"""eval/sample_charts.py + data.fetch.vision_klines without the network: a fake archive serves monthly zips of a
coin that pumps 3x on the event day; the facts must see the pump, the chart must be written, a missing coin is a row
with an error, not a crash."""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.data.fetch import vision_klines
from nig_timenet_torch.eval import sample_charts as sc

DAY = pd.Timestamp("2026-04-18", tz="UTC")


def _klines(interval_h: int) -> list[list]:
    t = pd.date_range(DAY - pd.Timedelta(days=50), DAY + pd.Timedelta(days=3), freq=f"{interval_h}h", inclusive="left")
    close = np.full(len(t), 1.0)
    pump = (t >= DAY + pd.Timedelta(hours=8)) & (t < DAY + pd.Timedelta(days=1))
    close[pump] = np.linspace(1.0, 3.0, pump.sum())
    close[t >= DAY + pd.Timedelta(days=1)] = 3.0
    open_ = np.r_[1.0, close[:-1]]
    vol = np.where(pump, 1e6, 1e3)
    ms = t.as_unit("ms").asi8
    return [[int(m), o, max(o, c) * 1.01, min(o, c) * 0.99, c, v, int(m) + 1, v * c, 10, v / 2, v * c / 2, 0]
            for m, o, c, v in zip(ms, open_, close, vol, strict=True)]


class FakeHttp:
    def __init__(self, coins=("PUMPUSDT",)):
        self.coins = coins

    def get_bytes(self, url: str):
        coin, interval = url.split("/klines/")[1].split("/")[:2]
        if coin not in self.coins or "/monthly/" not in url:
            return None
        month = url.rsplit("-", 2)[-2] + "-" + url.rsplit("-", 1)[-1][:2]
        rows = [r for r in _klines(int(interval[:-1])) if pd.Timestamp(r[0], unit="ms").strftime("%Y-%m") == month]
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("k.csv", "open_time,open\n" + "\n".join(",".join(map(str, r)) for r in rows))
        return buf.getvalue()


def test_vision_klines_reads_the_archive_between_the_bounds():
    f = vision_klines("PUMPUSDT", "4h", DAY - pd.Timedelta(days=2), DAY, FakeHttp())
    assert len(f) == 12 and f.index[0] == DAY - pd.Timedelta(days=2) and f.index.is_monotonic_increasing
    assert list(f.columns) == ["open", "high", "low", "close", "volume", "quote_volume", "trades"]


def test_facts_see_the_pump_and_the_chart_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    s = sc.Sample("PUMPUSDT", "2026-04-18", "short", -172.0, "test", "pump")
    candles = sc.load_candles(s, FakeHttp())
    f = sc.facts(s, candles["4h"])
    assert f["max_up_2bar_pct"] > 20 and f["event_return_pct"] == pytest.approx(200, rel=0.05)
    assert f["volume_vs_30d_median"] > 100 and f["window_flat_bars"] == 0
    assert sc.plot_sample(s, candles, tmp_path).name == "PUMPUSDT_20260418.png"


def test_run_writes_facts_and_records_a_missing_coin(tmp_path):
    pytest.importorskip("matplotlib")
    csv = tmp_path / "s.csv"
    pd.DataFrame({"coin": ["PUMPUSDT", "GONEUSDT"], "date": ["2026-04-18", "2026-04-18"], "side": ["short", ""],
                  "pnl": ["-172", ""], "source": ["a", "b"], "note": ["", ""], "time": ["", "true"]}).to_csv(csv,
                                                                                                          index=False)
    samples = sc.read_samples(csv)
    assert samples[1].time and np.isnan(samples[1].pnl) and samples[1].tag == "GONEUSDT_202604180000"
    table = sc.run(samples, tmp_path / "out", FakeHttp())
    assert (tmp_path / "out" / "facts.csv").is_file() and len(table) == 2
    assert table.loc[0, "chart"] == "PUMPUSDT_20260418.png" and table.loc[1, "chart"] == ""
