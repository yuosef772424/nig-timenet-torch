"""data_io/vision_slice.py (no network): candles -> the contract artifact's definition, checked by brute force."""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX, TS_COL
from nig_timenet_torch.data.artifact_writer import write_artifact
from nig_timenet_torch.data_io.loading import LoaderSettings, load_dataset
from nig_timenet_torch.data_io.vision_slice import build_logret_dataset, fetch_coin, fetch_month, load_frames
from nig_timenet_torch.doctor.data_checks import alignment

BAR = int(pd.Timedelta("4h").value)
T0 = int(pd.Timestamp("2024-01-01", tz="UTC").value)
WINDOW, HORIZON = 8, 2


def _frame(n, seed, hole=None):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * np.exp(rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * np.exp(-rng.uniform(0, 0.01, n))
    df = pd.DataFrame({"open_time": T0 + np.arange(n) * BAR, "open": open_, "high": high, "low": low, "close": close,
                       "volume": rng.uniform(1, 2, n)})
    if hole is not None:
        df = df.drop(index=range(*hole)).reset_index(drop=True)
    return df


@pytest.fixture(scope="module")
def frames():
    return {"TSTAUSDT": _frame(120, 1), "TSTBUSDT": _frame(90, 2, hole=(40, 44))}


@pytest.fixture(scope="module")
def ds(frames):
    return build_logret_dataset(frames, window=WINDOW, horizon=HORIZON)


def _brute(frames, ds, r):
    """Recompute row r from the candles by open time (a missing bar would KeyError: nothing crosses a hole)."""
    coin = next(b["name"] for b in ds["asset_bounds"] if b["start"] <= r < b["end"])
    df = frames[coin].set_index("open_time")
    ts = int(ds["last_candles"][r, TS_COL])
    bar = lambda k: df.loc[ts + k * BAR]  # noqa: E731
    x = np.array([[np.log(bar(k)[c] / bar(k - 1)["close"]) for c in ("high", "low", "close")]
                  for k in range(-WINDOW + 1, 1)])
    c0 = bar(0)["close"]
    nxt = [bar(k) for k in range(1, HORIZON + 1)]
    return x, {"close": np.log(nxt[-1]["close"] / c0), "high": np.log(max(b["high"] for b in nxt) / c0),
               "low": np.log(min(b["low"] for b in nxt) / c0)}, ts


def test_shapes_and_keys(ds):
    n = len(ds["last_candles"])
    assert ds["X_4h"].shape == (n, WINDOW, 3) and ds["X_4h"].dtype == np.float32
    for t in ("high", "low", "close"):
        assert ds[f"y_{t}_reg"].shape == (n,) and ds[f"y_{t}_class"].shape == (n,)
    assert ds["asset_bounds"][0] == {"name": "TSTAUSDT", "start": 0, "end": 120 - WINDOW - HORIZON}
    assert ds["asset_bounds"][1] == {"name": "TSTBUSDT", "start": 110, "end": 110 + (40 - WINDOW - HORIZON)
                                     + (46 - WINDOW - HORIZON)}      # one run of 40 bars, one of 46: two segments
    assert ds["asset_bounds"][-1]["end"] == n
    assert ds["price_norm_state"] == "unfitted" and ds["forecast_horizon"] == HORIZON


def test_rows_match_brute_force(frames, ds):
    n = len(ds["last_candles"])
    for r in sorted({0, 1, 5, n // 2, ds["asset_bounds"][0]["end"] - 1, ds["asset_bounds"][1]["start"], n - 1}):
        x, y, ts = _brute(frames, ds, r)
        np.testing.assert_allclose(ds["X_4h"][r], x, atol=1e-6)
        for t in ("close", "high", "low"):
            assert float(ds[f"y_{t}_reg"][r]) == pytest.approx(y[t], abs=1e-6)
            assert ds[f"y_{t}_class"][r] == float(y[t] > 0) or t != "close"
        assert ts == int(ds["last_candles"][r, TS_COL])


def test_a_hole_is_never_crossed(frames, ds):
    # TSTBUSDT lost bars 40..43: every row of it must still be fully rebuildable from consecutive bars
    b = ds["asset_bounds"][1]
    ts = ds["last_candles"][b["start"]:b["end"], TS_COL].astype(np.int64)
    missing = set(range(T0 + 40 * BAR, T0 + 44 * BAR, BAR))
    for r in range(b["start"], b["end"]):
        _brute(frames, ds, r)                                    # KeyError if a needed bar were absent
    for e in ts:
        used = set(range(int(e) - (WINDOW) * BAR, int(e) + (HORIZON + 1) * BAR, BAR))
        assert not used & missing


def test_time_seal_drops_late_rows(frames):
    seal = pd.Timestamp(T0 + 60 * BAR, tz="UTC")
    full = build_logret_dataset(frames, window=WINDOW, horizon=HORIZON)
    sealed = build_logret_dataset(frames, window=WINDOW, horizon=HORIZON, time_seal=seal)
    ts = sealed["last_candles"][:, TS_COL].astype(np.int64)
    assert len(ts) < len(full["last_candles"])
    assert ts.max() + HORIZON * BAR + (HORIZON + 1) * BAR <= seal.value
    assert sealed["last_candles"][:, TS_COL].max() < full["last_candles"][:, TS_COL].max()


def test_future_candles_do_not_change_earlier_rows(frames, ds):
    t_cut = T0 + 70 * BAR
    changed = {}
    for coin, df in frames.items():
        d = df.copy()
        late = d["open_time"] > t_cut
        factor = np.exp(np.random.default_rng(9).normal(0, 0.3, int(late.sum())))
        d.loc[late, ["open", "high", "low", "close"]] = d.loc[late, ["open", "high", "low", "close"]].to_numpy() \
            * factor[:, None]
        changed[coin] = d
    ds2 = build_logret_dataset(changed, window=WINDOW, horizon=HORIZON)
    assert len(ds2["last_candles"]) == len(ds["last_candles"])
    ts = ds["last_candles"][:, TS_COL].astype(np.int64)
    safe = ts + HORIZON * BAR <= t_cut                                 # the label's last bar is at or before t
    assert 10 < safe.sum() < len(ts)
    assert not np.array_equal(ds["last_candles"], ds2["last_candles"])  # the perturbation did something
    for k in ("X_4h", "y_high_reg", "y_low_reg", "y_close_reg", "y_close_class", "base_params", "last_candles"):
        assert np.array_equal(ds[k][safe][:, ...], ds2[k][safe][:, ...]), k
    col = [LAST_COLUMN_INDEX[c] for c in ("last_high", "last_low", "last_close", "timestamp")]
    assert np.array_equal(ds["last_candles"][safe][:, col], ds2["last_candles"][safe][:, col])


def _zip(rows, header=False):
    csv = "\n".join(([("open_time,open,high,low,close,volume,close_time")] if header else [])
                    + [f"{int(t // 1_000_000)},{o},{h},{lo},{c},{v},0" for t, o, h, lo, c, v in rows])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.csv", csv)
    return buf.getvalue()


def test_fetch_month_and_load_frames(tmp_path, frames):
    raw, calls = tmp_path / "raw", []
    df = frames["TSTAUSDT"]
    rows = list(df.itertuples(index=False, name=None))
    jan = [r for r in rows if pd.Timestamp(r[0], tz="UTC").month == 1]
    feb = [r for r in rows if pd.Timestamp(r[0], tz="UTC").month == 2]

    def get(url):
        calls.append(url)
        if url.endswith("2024-01.zip"):
            return _zip(jan)
        if url.endswith("2024-02.zip"):
            return _zip(feb, header=True)
        return None

    p = fetch_month("TSTAUSDT", "4h", pd.Period("2024-01", "M"), raw, get)
    assert p == raw / "TSTAUSDT" / "TSTAUSDT-4h-2024-01.csv" and p.exists()
    assert fetch_month("TSTAUSDT", "4h", pd.Period("2024-01", "M"), raw, get) == p and len(calls) == 1  # kept
    assert fetch_month("TSTAUSDT", "4h", pd.Period("2024-03", "M"), raw, get) is None
    assert (raw / "TSTAUSDT" / "TSTAUSDT-4h-2024-03.missing").exists()
    n = len(calls)
    assert fetch_month("TSTAUSDT", "4h", pd.Period("2024-03", "M"), raw, get) is None and len(calls) == n
    assert fetch_coin("TSTAUSDT", "4h", "2024-01", "2024-03", raw, get) == 2
    out = load_frames(raw, ["TSTAUSDT", "NOPEUSDT"], "4h")
    assert list(out) == ["TSTAUSDT"]
    got = out["TSTAUSDT"]
    assert len(got) == len(jan) + len(feb)
    assert got["open_time"].iloc[0] == df["open_time"].iloc[0]
    np.testing.assert_allclose(got["close"].to_numpy(), df["close"].to_numpy()[: len(got)])


def test_artifact_round_trip_aligned(tmp_path, ds):
    write_artifact(ds, tmp_path / "art", source_commit="a" * 40, overwrite=True)
    loaded = load_dataset(LoaderSettings(name=str(tmp_path / "art"), mmap=False, local_dir="")).dataset
    assert len(loaded["last_candles"]) == len(ds["last_candles"])
    rep = alignment(loaded)
    assert rep is not None and rep["pairs"] > 0
    assert rep["window_err"] <= 2e-5 and max(rep["label_err"].values()) <= 2e-5
