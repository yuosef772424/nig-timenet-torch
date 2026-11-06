"""The ar_windows unit contract: raw candles -> gap-free (context + horizon) windows, any pair of
lengths, free coin selection, and the timestamp-aligned panel grid."""

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.data.ar_windows import (
    FEATURE_NAMES,
    RET_COL,
    build_windows,
    candle_features,
    contiguous_runs,
    fit_volume_median,
    panel_windows,
    raw_candles,
    resolve_coin_list,
    split_panel,
    split_windows,
)


def _frame(n: int = 200, freq: str = "1h", gaps: tuple[int, ...] = ()) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    if gaps:  # drop a few bars: the index has holes
        idx = idx.delete(list(gaps))
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    return pd.DataFrame({"open": close * (1 + rng.normal(0, 0.002, len(idx))),
                         "high": close * (1 + np.abs(rng.normal(0, 0.003, len(idx)))),
                         "low": close * (1 - np.abs(rng.normal(0, 0.003, len(idx)))),
                         "close": close, "volume": rng.uniform(10, 1000, len(idx))}, index=idx)


def test_features_shape_and_first_bar():
    f = candle_features(_frame(), volume_median=100.0)
    assert f.shape == (200, len(FEATURE_NAMES)) and f.dtype == np.float32
    assert f[0, RET_COL] == 0.0  # the first bar has no earlier close to return against
    assert np.isfinite(f).all()


def test_features_causal_past_does_not_see_future():
    df = _frame()
    f1 = candle_features(df, 100.0)
    df.iloc[100:] = df.iloc[100:] * 1.5  # shock the future bars only
    f2 = candle_features(df, 100.0)
    assert np.allclose(f1[:100], f2[:100])


def test_volume_median_uses_only_train_head():
    df = _frame(100)
    assert fit_volume_median(df, 0.7) == pytest.approx(np.median(df["volume"].to_numpy()[:70]))


def test_runs_and_windows_skip_gaps():
    df = _frame(200, gaps=(50, 51, 120))
    runs = contiguous_runs(df.index, "1h")
    assert [r.stop - r.start for r in runs] == [50, 68, 79]  # 200 bars - 3 dropped
    feats = candle_features(df, 100.0)
    w = build_windows(feats, runs, context_len=32, horizon_len=16)
    assert w.X.shape == (w.n, 32, 5) and w.Y.shape == (w.n, 16, 5) and w.dirs.shape == (w.n, 16)
    # 48-bar windows only, one per possible end position inside a run: 3 + 21 + 32 = 56 windows
    assert w.n == 56


def test_window_alignment_and_dirs():
    feats = np.arange(48 * 5, dtype="float32").reshape(48, 5)
    w = build_windows(feats, [slice(0, 48)], context_len=32, horizon_len=16)
    assert np.allclose(w.X[0], feats[:32]) and np.allclose(w.Y[0], feats[32:])
    assert set(np.unique(w.dirs)) <= {-1.0, 0.0, 1.0}
    assert w.trend_dir.shape == (1,) and w.last_ret[0] == w.X[0, -1, RET_COL]


def test_any_horizon_length():
    feats = np.random.default_rng(1).normal(size=(100, 5)).astype("float32")
    for c, h in ((32, 4), (32, 8), (16, 2), (8, 1)):
        w = build_windows(feats, [slice(0, 100)], context_len=c, horizon_len=h)
        assert w.X.shape == (100 - c - h + 1, c, 5) and w.Y.shape == (100 - c - h + 1, h, 5)


def test_split_windows_time_ordered_and_keys():
    coins = {"A": build_windows(np.random.default_rng(2).normal(size=(120, 5)).astype("float32"),
                                [slice(0, 120)], 32, 16)}
    d = split_windows(coins, 0.15, 0.15)
    assert set(d) == {f"{s}_{k}" for s in ("train", "val", "test")
                      for k in ("X", "Y", "dirs", "trend_dir", "last_ret", "coins")}
    assert d["train_X"].shape[0] + d["val_X"].shape[0] + d["test_X"].shape[0] == coins["A"].n
    # train rows are the coin's first windows in time: their first context row matches
    assert np.allclose(d["train_X"][0], coins["A"].X[0])
    # the coin labels ride along (one name per window) and end times only when an index was given
    assert list(d["train_coins"]) == ["A"] * d["train_X"].shape[0]
    assert "train_end_times" not in d  # built without an index: no timestamps to carry


def test_raw_candles_synthetic_and_unknown():
    frames = raw_candles("synthetic", ["AAA"], 10, seed=3)
    assert list(frames) == ["AAA"] and len(frames["AAA"]) == 240
    with pytest.raises(ValueError, match="unknown source"):
        raw_candles("nope", ["AAA"], 10)


def test_raw_candles_drive_reads_the_crypto_signal_layout(tmp_path):
    """SOURCE="drive" reads `<root>/history_1h/<SYMBOL>.csv` (the crypto-signal-prediction Drive layout)
    via the SAME files loader the data pipeline uses, keeps the newest `days` days, and refuses with a
    clear error when a file is missing (no silent synthetic fallback: the owner asked for THAT data)."""
    root = tmp_path / "drive"
    (root / "history_1h").mkdir(parents=True)
    idx = pd.date_range("2026-01-01", periods=24 * 30, freq="1h", tz="UTC")  # 30 days of hourly bars
    close = 100.0 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, len(idx))))
    pd.DataFrame({"datetime_utc": idx.strftime("%Y-%m-%d %H:%M:%S"), "open": close, "high": close * 1.01,
                  "low": close * 0.99, "close": close, "volume": 50.0}).to_csv(
        root / "history_1h" / "BTCUSDT.csv", index=False)
    frames = raw_candles("drive", ["BTCUSDT"], 10, root=str(root))  # the newest 10 days
    assert list(frames) == ["BTCUSDT"]
    assert len(frames["BTCUSDT"]) == 241  # 10 days x 24 bars + the boundary bar (trim is inclusive)
    assert list(frames["BTCUSDT"].columns) == ["open", "high", "low", "close", "volume"]
    # everything is kept when days <= 0
    assert len(raw_candles("drive", ["BTCUSDT"], 0, root=str(root))["BTCUSDT"]) == 24 * 30
    # a missing coin is a CLEAR error naming the searched path, never a silent fallback
    with pytest.raises(FileNotFoundError, match="ETHUSDT"):
        raw_candles("drive", ["ETHUSDT"], 10, root=str(root))


def test_end_times_are_recorded_only_with_an_index():
    feats = np.random.default_rng(4).normal(size=(60, 5)).astype("float32")
    idx = pd.date_range("2026-01-01", periods=60, freq="1h", tz="UTC")
    w = build_windows(feats, [slice(0, 60)], 32, 16, index=idx)
    assert w.end_times is not None and len(w.end_times) == w.n
    assert int(w.end_times[0]) == int(idx[47].as_unit("ns").asm8.view("i8"))  # first window's last candle
    assert build_windows(feats, [slice(0, 60)], 32, 16).end_times is None


def test_panel_windows_aligns_coins_on_shared_end_times():
    coins = {}
    for name, seed, drop in (("A", 1, ()), ("B", 2, (30, 31)), ("C", 3, ())):
        n = 200 - len(drop)
        idx = pd.date_range("2026-01-01", periods=200, freq="1h", tz="UTC").delete(list(drop))
        feats = np.random.default_rng(seed).normal(size=(n, 5)).astype("float32")
        coins[name] = build_windows(feats, contiguous_runs(idx, "1h"), 32, 16, index=idx)
    panel = panel_windows(coins)
    assert panel["X"].shape[1] == 3 and panel["X"].shape[2] == 32 and panel["Y"].shape[2] == 16
    assert list(panel["coin_names"]) == ["A", "B", "C"]
    # B's two dropped bars remove its windows around the gap, so the shared grid is smaller than A's
    assert 0 < panel["n"] < coins["A"].n
    sets = {n: set(w.end_times.tolist()) for n, w in coins.items()}
    for t in panel["end_times"]:
        assert all(t in s for s in sets.values())  # every grid position is a window of EVERY coin


def test_panel_windows_needs_end_times_and_common_stamps():
    feats = np.random.default_rng(5).normal(size=(60, 5)).astype("float32")
    no_times = {"A": build_windows(feats, [slice(0, 60)], 32, 16)}
    with pytest.raises(ValueError, match="no end times"):
        panel_windows(no_times)
    idx_a = pd.date_range("2026-01-01", periods=60, freq="1h", tz="UTC")
    idx_b = pd.date_range("2027-01-01", periods=60, freq="1h", tz="UTC")  # disjoint: no shared stamp
    disj = {"A": build_windows(feats, [slice(0, 60)], 32, 16, index=idx_a),
            "B": build_windows(feats, [slice(0, 60)], 32, 16, index=idx_b)}
    with pytest.raises(ValueError, match="no timestamp"):
        panel_windows(disj)


def test_split_panel_time_ordered():
    feats = np.random.default_rng(6).normal(size=(200, 5)).astype("float32")
    idx = pd.date_range("2026-01-01", periods=200, freq="1h", tz="UTC")
    coins = {"A": build_windows(feats, [slice(0, 200)], 32, 16, index=idx),
             "B": build_windows(feats, [slice(0, 200)], 32, 16, index=idx)}
    panel = panel_windows(coins)
    d = split_panel(panel, 0.15, 0.15)
    assert d["train_X"].shape[0] + d["val_X"].shape[0] + d["test_X"].shape[0] == panel["n"]
    assert d["train_X"].shape[1:] == panel["X"].shape[1:]  # coins and candles keep the grid shape
    assert d["train_end_times"][-1] < d["val_end_times"][0] < d["test_end_times"][0]  # time-ordered
    with pytest.raises(ValueError, match="empty"):
        split_panel(panel, 0.0, 0.0)  # no val or test rows left: a loud error, not a silent empty split


def test_resolve_coin_list_all_explicit_and_refused():
    assert len(resolve_coin_list("all")) >= 6          # the registry's default category
    assert resolve_coin_list(("BTCUSDT", "ETHUSDT")) == ["BTCUSDT", "ETHUSDT"]
    with pytest.raises(ValueError, match="unknown coin selection"):
        resolve_coin_list("everything")
    with pytest.raises(ValueError, match="empty"):
        resolve_coin_list([])
